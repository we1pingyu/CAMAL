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
import ast
from sklearn.model_selection import KFold

sys.path.append('./lrkv')
from runner import Runner
from lsm_tree.PySim import RocksDB
from lsm_tree.cost_function import CostFunction
from utils.model_lr import (
    get_level_cost,
    get_cache_uniform,
    traverse_for_T,
    traverse_for_h,
)
from utils.distribution import dist_regression, generate_key_log
from utils.lsm import *

workloads = [
    # (0.25, 0.25, 0.25, 0.25),
    # (0.97, 0.01, 0.01, 0.01),
    # (0.01, 0.97, 0.01, 0.01),
    # (0.01, 0.01, 0.97, 0.01),
    # (0.01, 0.01, 0.01, 0.97),
    # (0.49, 0.49, 0.01, 0.01),
    # (0.49, 0.01, 0.49, 0.01),
    # (0.49, 0.01, 0.01, 0.49),
    # (0.01, 0.49, 0.49, 0.01),
    # (0.01, 0.49, 0.01, 0.49),
    # (0.01, 0.01, 0.49, 0.49),
    # (0.33, 0.33, 0.33, 0.01),
    # (0.33, 0.33, 0.01, 0.33),
    # (0.33, 0.01, 0.33, 0.33),
    # (0.01, 0.33, 0.33, 0.33),
    (0.0, 0.0, 1.0, 0.0)
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
        kv_size=8192,
        scaling=1.0,
        path_db='/tmp',
    ):
        z0, z1, q, w = workload
        self.logger.info(f'Workload : {z0},{z1},{q},{w}')
        self.logger.info(f'Building DB at size : {n}')
        row = self.config['lsm_tree_config'].copy()

        row['db_name'] = 'level_cost'
        row['path_db'] = path_db
        buffer = buffer
        row['T'] = size_ratio
        row['N'] = n
        row['M'] = buffer * scaling + (bpe * n)
        row['h'] = bpe
        row['dist'] = dist
        row['skew'] = skew
        row['cache_cap'] = cache_cap * scaling
        row['is_leveling_policy'] = True
        row['queries'] = queries
        row['mbuf'] = buffer * scaling
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
            kv_size,
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
            scaling=scaling,
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
        files_per_level = ast.literal_eval(row['files_per_level'])
        row['actual_runs'] = len([i for i in files_per_level if i != 0])
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
        self.logger.info('actual_runs: {}'.format(row['actual_runs']))
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

        for step in range(100):
            for workload in workloads:
                # z0, z1, q, w = workload
                size_ratio = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 32, 64])
                ratio = random.uniform(0.5, 1)
                dist = 'uniform'
                skew = 0.0
                bpe = 10
                buffer = ratio * (M - bpe * N)
                cache_cap = 0.0
                key_log = key_path + '/{}.dat'.format(step)
                # original
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
                pd.DataFrame(df).to_csv(
                    "raw_data/samples_sim_lr_level_uniform_range_ckpt.csv"
                )

                # scaling
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
                    kv_size=8 * 8,
                    scaling=32 / 1052,
                    path_db='/dev/shm',
                )
                # print(row)
                df.append(row)
                pd.DataFrame(df).to_csv(
                    "raw_data/samples_sim_lr_level_uniform_range_ckpt.csv"
                )

        self.logger.info('Exporting data from active learning level cost')
        pd.DataFrame(df).to_csv("raw_data/samples_sim_lr_level_uniform_range_final.csv")
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
