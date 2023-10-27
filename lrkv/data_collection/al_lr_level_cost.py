import logging
import time
import numpy as np
import pandas as pd
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
from utils.distribution import dist_regression, generate_key_log
from utils.model_lr import (
    get_level_cost,
    get_cache_uniform,
    get_candidate_simulated_annealing,
    traverse_candidate,
)

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
    (0.33, 0.01, 0.33, 0.01),
    (0.01, 0.33, 0.33, 0.33),
]

M = 2147483648  # 256MB
n_estimators = 100
n = 1e7
queries = 200000
fold = 10
eps = 1e-8


class LevelCost(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('rlt_logger')

    def single_run(
        self,
        workload,
        ratio,
        size_ratio,
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
        row['ratio'] = ratio
        row['T'] = size_ratio
        row['N'] = n
        row['M'] = buffer + (bpe * n)
        row['h'] = bpe
        row['dist'] = dist
        row['skew'] = skew
        row['cache_cap'] = cache_cap
        row['is_leveling_policy'] = True
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

    def run(self):
        start_time = time.time()
        df = []
        key_path = 'key_log_al_lr_level_cost'
        if not os.path.exists(key_path):
            os.makedirs(key_path)
        step = 0
        # collect init data
        # workloads = [[0.97, 0.01, 0.01, 0.01]]
        for workload in workloads:
            for _ in range(3):
                ratio = random.uniform(0.25, 1.0)
                # dist = np.random.choice(['uniform', 'zipfian'], p=[0.25, 0.75])
                dist = 'uniform'
                skew = random.random()
                bpe = random.choice(range(2, 15))
                buffer = ratio * (M - bpe * n)
                cache_cap = (1 - ratio) * (M - bpe * n) / 8
                size_ratio = random.choice(range(2, 17))
                key_log = key_path + '/{}.dat'.format(step)
                row = self.single_run(
                    workload,
                    ratio,
                    size_ratio,
                    n,
                    buffer,
                    bpe,
                    dist,
                    skew,
                    cache_cap,
                    queries,
                    key_log,
                )
                df.append(row)
                pd.DataFrame(df).to_csv('raw_data/al_lr_level_cost_ckpt.csv')
                step += 1
            # iter model
            X = []
            Y = []
            Xc = []
            Yc = []
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
                    get_level_cost(
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

            candidates = traverse_candidate(
                [Wc],
                [W],
                workload[0],
                workload[1],
                workload[2],
                workload[3],
            )
            for c in range(3):
                candidate = candidates[c]
                print(candidate)
                size_ratio, bpe, ratio, _, _ = candidate
                buffer = ratio * (M - bpe * n)
                cache_cap = (1 - ratio) * (M - bpe * n) / 8
                row = self.single_run(
                    workload,
                    ratio,
                    size_ratio,
                    n,
                    buffer,
                    bpe,
                    dist,
                    skew,
                    cache_cap,
                    queries,
                    key_log,
                )
                df.append(row)
                pd.DataFrame(df).to_csv('raw_data/al_lr_level_cost_ckpt.csv')
                step += 1

            # iter model
            X = []
            Y = []
            Xc = []
            Yc = []
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
                    get_level_cost(
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

            candidates = traverse_candidate(
                [Wc],
                [W],
                workload[0],
                workload[1],
                workload[2],
                workload[3],
            )
            for c in range(3):
                candidate = candidates[c]
                print(candidate)
                size_ratio, bpe, ratio, _, _ = candidate
                buffer = ratio * (M - bpe * n)
                cache_cap = (1 - ratio) * (M - bpe * n) / 8
                row = self.single_run(
                    workload,
                    ratio,
                    size_ratio,
                    n,
                    buffer,
                    bpe,
                    dist,
                    skew,
                    cache_cap,
                    queries,
                    key_log,
                )
                df.append(row)
                pd.DataFrame(df).to_csv('raw_data/al_lr_level_cost_ckpt.csv')
                step += 1
        self.logger.info('Exporting data from active learning level cost')
        pd.DataFrame(df).to_csv('raw_data/al_lr_level_cost.csv')
        self.logger.info(f'Finished al_lr_level_cost, use {time.time()-start_time}s\n')


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
