import logging
import time
import numpy as np
import pandas as pd
import copy
import random
import sys
import os
import yaml

sys.path.append('./lrkv')
from runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.cost_function import NominalCostFunction
from utils.lsm import level_grid_sampling, level_min_cost_sampling

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

size_ratios = [2, 4, 6, 8, 10, 12, 14, 16]
M = 2147483648  # 256MB
n = 1e7


class LevelCost(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('rlt_logger')

    def run(self):
        start_time = time.time()
        df = []
        queries = 200000
        iter = 0
        for workload in workloads:
            candidates = level_min_cost_sampling(workload, 32)
            for candidate in candidates:
                size_ratio, bpe, ratio = candidate
                dist = 'unform'
                skew = 0
                buffer = ratio * (M - bpe * n)
                cache_cap = (1 - ratio) * (M - bpe * n) / 8
                z0, z1, q, w = workload
                self.logger.info(f'Workload : {z0},{z1},{q},{w}')
                key_path = 'key_log_level_cost_uniform'
                if not os.path.exists(key_path):
                    os.makedirs(key_path)
                self.logger.info(f'Building DB at size : {n}')
                row = self.config['lsm_tree_config'].copy()

                row['group'] = iter
                row['db_name'] = 'level_cost'
                row['ratio'] = ratio
                row['path_db'] = self.config['app']['DATABASE_PATH']
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
                key_log = key_path + '/{}.dat'.format(iter)
                row = copy.deepcopy(row)
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
                cf = NominalCostFunction(
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
                row['read_model_io'] = queries * cf.calculate_read_cost(
                    row['h'], row['T']
                )
                row['write_model_io'] = queries * cf.calculate_write_cost(
                    row['h'], row['T']
                )
                row['model_io'] = row['read_model_io'] + row['write_model_io']
                self.logger.info('mbuf: {}'.format(row['mbuf']))
                self.logger.info('read_model_io: {}'.format(row['read_model_io']))
                self.logger.info('write_model_io: {}'.format(row['write_model_io']))
                self.logger.info('model_io: {}'.format(row['model_io']))
                # print(row)
                df.append(row)
                pd.DataFrame(df).to_csv('raw_data/level_cost_grid_uniform_ckpt.csv')
                iter += 1

        self.logger.info('Exporting data from level cost')
        pd.DataFrame(df).to_csv('raw_data/level_cost_grid_uniform_min.csv')
        self.logger.info(
            f'Finished level_cost_uniform, use {time.time()-start_time}s\n'
        )


if __name__ == "__main__":

    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join('lrkv/config/robust-lsm-trees.yaml')

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = Runner(config)
    driver.run(LevelCost(config))
