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
from utils.model_xgb import get_cost_uniform, traverse_for_T, traverse_for_h
from utils.distribution import dist_regression, generate_key_log
from utils.lsm import *
from utils.model_xgb import get_candidate_simulated_annealing, get_cost, get_cache
from bayes_opt import BayesianOptimization

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
        self.sampels = 3

    def single_run(
        self,
        size_ratio,
        ratio,
        bpe,
    ):
        z0, z1, q, w = self.workload
        buffer = ratio * (M - bpe * N)
        cache_cap = (1 - ratio) * (M - bpe * N) / 8
        self.logger.info(f'Workload : {z0},{z1},{q},{w}')
        self.logger.info(f'Building DB at size : {N}')
        row = self.config['lsm_tree_config'].copy()

        row['db_name'] = 'level_cost'
        row['path_db'] = self.config['app']['DATABASE_PATH']
        row['T'] = size_ratio
        row['N'] = N
        row['M'] = buffer + (bpe * N)
        row['h'] = bpe
        row['dist'] = self.dist
        row['skew'] = self.skew
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
        row['key_log'] = self.key_log
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
            self.dist,
            self.skew,
            queries,
            is_leveling_policy=row['is_leveling_policy'],
            cache_cap=cache_cap,
            key_log=self.key_log,
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
        self.df.append(row)
        pd.DataFrame(self.df).to_csv('raw_data/bo_level_uniform_ckpt.csv')
        return row

    def objective(
        self,
        size_ratio,
        ratio,
        bpe,
    ):
        row = self.single_run(
            int(size_ratio),
            ratio,
            int(bpe),
        )
        return row['total_latency']

    def run(self):
        start_time = time.time()
        self.df = []
        key_path = 'key_log_al_level_cost'
        if not os.path.exists(key_path):
            os.makedirs(key_path)
        step = 0
        num_samples = 3
        X = []
        Y = []
        for workload in workloads:
            z0, z1, q, w = workload
            # Train and search optimal size ratio
            self.key_log = key_path + '/{}.dat'.format(step)
            self.queries = queries
            self.dist = 'uniform'
            self.skew = 0.0
            self.workload = workload
            bo_search = BayesianOptimization(
                self.objective,
                {
                    'size_ratio': (2, estimate_T(N, M / 2 / 8, 1) + 1),
                    'bpe': (2, 17),
                    'ratio': (0.7, 1.0),
                },
            )
            bo_search.maximize(n_iter=9)
            step += 1

        self.logger.info('Exporting data from active learning level cost')
        pd.DataFrame(self.df).to_csv('raw_data/bo_level_uniform_final.csv')
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
