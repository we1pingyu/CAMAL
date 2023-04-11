import logging
import time
import numpy as np
import pandas as pd
import xgboost as xgb
import copy
import random
import sys
import os
import yaml
from sklearn.model_selection import KFold

sys.path.append('./lrkv')
from runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.cost_function import CostFunction
from utils.model_lr import (
    get_tier_cost,
    get_cache_uniform,
    traverse_for_T,
    traverse_for_h,
)
from utils.distribution import dist_regression, generate_key_log
from utils.lsm import *

workloads = [
    (0.25, 0.25, 0.25, 0.25),
    (0.97, 0.01, 0.01, 0.01),
    (0.01, 0.97, 0.01, 0.01),
    (0.01, 0.01, 0.97, 0.01),
    (0.01, 0.01, 0.01, 0.97),
    (0.49, 0.49, 0.01, 0.01),
    (0.49, 0.01, 0.49, 0.01),
    (0.49, 0.01, 0.01, 0.49),
    (0.01, 0.49, 0.49, 0.01),
    (0.01, 0.49, 0.01, 0.49),
    (0.01, 0.01, 0.49, 0.49),
    (0.33, 0.33, 0.33, 0.01),
    (0.33, 0.33, 0.01, 0.33),
    (0.33, 0.01, 0.33, 0.33),
    (0.01, 0.33, 0.33, 0.33),
]

M = 2147483648  # 256MB
n_estimators = 100
N = 1e7
queries = 200000
fold = 10


class LevelCost(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('rlt_logger')

    def single_run(
        self,
        workload,
        size_ratio,
        ratio,
        n,
        buffer,
        bpe,
        dist,
        skew,
        cache_cap,
        queries,
        key_log,
    ):
        z0, z1, q, w = workload
        self.logger.info(f'Workload : {z0},{z1},{q},{w}')
        self.logger.info(f'Building DB at size : {n}')
        row = self.config['lsm_tree_config'].copy()

        row['db_name'] = 'level_cost'
        row['path_db'] = self.config['app']['DATABASE_PATH']
        row['T'] = size_ratio
        row['N'] = n
        row['M'] = buffer + (bpe * n)
        row['h'] = bpe
        row['dist'] = dist
        row['skew'] = skew
        row['cache_cap'] = cache_cap
        row['is_leveling_policy'] = False
        row['queries'] = queries
        row['mbuf'] = buffer / 8
        row['z0'] = z0
        row['z1'] = z1
        row['q'] = q
        row['w'] = w
        db = RocksDB(self.config)

        self.logger.info('Running workload')
        row['key_log'] = key_log
        results = db.run(
            row['db_name'],
            row['path_db'],
            row['h'],
            row['T'],
            row['N'],
            row['E'],
            row['M'],
            z0,
            z1,
            q,
            w,
            dist,
            skew,
            queries,
            is_leveling_policy=row['is_leveling_policy'],
            cache_cap=cache_cap,
            key_log=key_log,
        )

        for key, val in results.items():
            self.logger.info(f'{key} : {val}')
            row[f'{key}'] = val
        cf = CostFunction(
            row['N'],
            row['phi'],
            row['s'],
            row['B'],
            row['E'],
            row['M'],
            row['is_leveling_policy'],
            z0,
            z1,
            q,
            w,
        )
        row['L'] = cf.L(row['h'], row['T'])
        row['z0'] = z0
        row['z1'] = z1
        row['q'] = q
        row['w'] = w
        row['ratio'] = ratio
        row['write_io'] = (
            row['bytes_written']
            + row['compact_read']
            + row['compact_write']
            + row['flush_written']
        ) / 4096
        self.logger.info('write_io: {}'.format(row['write_io']))
        row['read_model_io'] = queries * cf.calculate_read_cost(row['h'], row['T'])
        row['write_model_io'] = queries * cf.calculate_write_cost(row['h'], row['T'])
        row['model_io'] = row['read_model_io'] + row['write_model_io']
        self.logger.info('mbuf: {}'.format(row['mbuf']))
        self.logger.info('read_model_io: {}'.format(row['read_model_io']))
        self.logger.info('write_model_io: {}'.format(row['write_model_io']))
        self.logger.info('model_io: {}'.format(row['model_io']))
        return row

    def sample_around_x0(self, x0, h, lower_bound, upper_bound):
        lower_bound = lower_bound
        upper_bound = upper_bound
        if x0 < lower_bound:
            x0 = lower_bound
        if x0 > upper_bound:
            x0 = upper_bound
        samples = []
        left_offset = min(h // 2, x0 - lower_bound)
        right_offset = h - left_offset - 1
        for i in range(-left_offset, right_offset + 1):
            value = x0 + i
            if lower_bound <= value <= upper_bound:
                samples.append(value)
        while len(samples) < h and (upper_bound - lower_bound + 1) >= h:
            right_offset += 1
            value = x0 + right_offset
            if lower_bound <= value <= upper_bound:
                samples.append(value)
            left_offset += 1
            value = x0 - left_offset
            if lower_bound <= value <= upper_bound:
                samples.append(value)
        return samples

    def run(self):
        start_time = time.time()
        df = []
        key_path = 'key_log_al_level_cost'
        if not os.path.exists(key_path):
            os.makedirs(key_path)
        step = 0
        eps = 1e-7
        num_samples = 3
        X = []
        Y = []
        Xc = []
        Yc = []
        for workload in workloads:
            z0, z1, q, w = workload
            # Train and search optimal size ratio
            min_err = 1e9
            for T in range(2, estimate_T(N, M / 2 / 8, 1) + 1):
                err = T_level_equation(T, q, w)
                if err < min_err:
                    min_err = err
                    temp = T
            T_list = self.sample_around_x0(temp, 3, 2, estimate_T(N, M / 2 / 8, 1))
            print(T_list)
            # continue
            z0, z1, q, w = workload
            ratio = 1.0
            dist = 'uniform'
            skew = 0.0
            bpe = 10
            buffer = ratio * M - bpe * N
            cache_cap = (1 - ratio) * (M - bpe * N) / 8
            for size_ratio in T_list:
                key_log = key_path + '/{}.dat'.format(step)
                row = self.single_run(
                    workload,
                    size_ratio,
                    ratio,
                    N,
                    buffer,
                    bpe,
                    dist,
                    skew,
                    cache_cap,
                    queries,
                    key_log,
                )
                # print(row)
                df.append(row)
                pd.DataFrame(df).to_csv('raw_data/al_xgb_level_cost_ckpt.csv')
                step += 1

            # iter model
            for sample in df:
                xc = get_cache_uniform(
                    sample['T'],
                    sample['h'],
                    sample['ratio'],
                    sample['z0'],
                    sample['z1'],
                    sample['q'],
                    sample['w'],
                )
                Xc.append(xc)
                Yc.append(np.log(sample['cache_hit_rate'] + eps))
                X.append(
                    get_tier_cost(
                        sample['T'],
                        sample['h'],
                        sample['ratio'],
                        sample['z0'],
                        sample['z1'],
                        sample['q'],
                        sample['w'],
                        sample['cache_hit_rate'],
                    )
                )
                Y.append(sample['total_latency'] / sample['queries'])
            _Xc = np.array(Xc)
            _Yc = np.array(Yc)
            Wc = np.linalg.lstsq(_Xc, _Yc, rcond=-1)[0]
            _X = np.array(X)
            _Y = np.array(Y)
            W = np.linalg.lstsq(_X, _Y, rcond=-1)[0]
            candidates = traverse_for_T([W], [Wc], z0, z1, q, w, n=1, policy='tier')
            T0 = candidates[0][0]

            min_err = 1e9
            for h in range(1, 17):
                err = h_mbuf_level_equation(h * N, z0, z1, q, w, T0)
                if err < min_err:
                    min_err = err
                    temp = h
            h_list = self.sample_around_x0(temp, 3, 1, 16)
            print(h_list)
            for h in h_list:
                buffer = ratio * (M - h * N)
                cache_cap = 0
                size_ratio = T0
                key_log = key_path + '/{}.dat'.format(step)
                row = self.single_run(
                    workload,
                    size_ratio,
                    ratio,
                    N,
                    buffer,
                    h,
                    dist,
                    skew,
                    cache_cap,
                    queries,
                    key_log,
                )
                # print(row)
                df.append(row)
                pd.DataFrame(df).to_csv('raw_data/al_xgb_level_cost_ckpt.csv')
                step += 1
            # iter model
            X = []
            Y = []
            for sample in df:
                xc = get_cache_uniform(
                    sample['T'],
                    sample['h'],
                    sample['ratio'],
                    sample['z0'],
                    sample['z1'],
                    sample['q'],
                    sample['w'],
                )
                Xc.append(xc)
                Yc.append(np.log(sample['cache_hit_rate'] + eps))
                X.append(
                    get_tier_cost(
                        sample['T'],
                        sample['h'],
                        sample['ratio'],
                        sample['z0'],
                        sample['z1'],
                        sample['q'],
                        sample['w'],
                        sample['cache_hit_rate'],
                    )
                )
                Y.append(sample['total_latency'] / sample['queries'])
            _Xc = np.array(Xc)
            _Yc = np.array(Yc)
            Wc = np.linalg.lstsq(_Xc, _Yc, rcond=-1)[0]
            _X = np.array(X)
            _Y = np.array(Y)
            W = np.linalg.lstsq(_X, _Y, rcond=-1)[0]
            candidates = traverse_for_h([W], [Wc], z0, z1, q, w, n=1, policy='tier')
            h0 = candidates[0][1]

            min_err = 1e9
            for ratio in [0.8, 0.9]:
                buffer = ratio * (M - h0 * N)
                h0 = ratio * h0
                cache_cap = (1 - ratio) * M / 8
                size_ratio = T0
                key_log = key_path + '/{}.dat'.format(step)
                row = self.single_run(
                    workload,
                    size_ratio,
                    ratio,
                    N,
                    buffer,
                    h0,
                    dist,
                    skew,
                    cache_cap,
                    queries,
                    key_log,
                )
                # print(row)
                df.append(row)
                pd.DataFrame(df).to_csv('raw_data/al_xgb_level_cost_ckpt.csv')
                step += 1

        self.logger.info('Exporting data from active learning level cost')
        pd.DataFrame(df).to_csv('raw_data/camal_lr_tier2.csv')
        self.logger.info(f'Finished al_level_cost, use {time.time()-start_time}s\n')


if __name__ == "__main__":

    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join('lrkv/config/config.yaml')

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = Runner(config)
    driver.run(LevelCost(config))
