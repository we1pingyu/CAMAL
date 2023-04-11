import logging

import numpy as np
import pandas as pd
import random
import sys
import os
import yaml

sys.path.append('./endure')
from runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.cost_function import CostFunction
from data.data_provider import DataProvider
from data.data_exporter import DataExporter


class LevelCostDist(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('rlt_logger')
        self.dp = DataProvider(config)
        self.de = DataExporter(config)

    def run(self):
        df = []
        num_queries = 100000
        for group in range(512):
            n = int(1e7 + (2e7 - 1e7) * np.random.random())
            # n = 1e7
            size_ratio = random.choice(range(4, 11))
            # size_ratio = 5
            dist = np.random.choice(['uniform', 'zipfian'], p=[0.3, 0.7])
            # dist = 'uniform'
            skew = random.random()
            # skew = 0.9
            bpe_budget = random.choice(range(1, 13))
            # bpe_budget = 10
            buffer = random.choice(range(1, 17))
            # buffer = 1
            buffer = buffer * 8 * 1024 * 1024 * 8  # *8MB
            cache_cap = random.choice(range(1, 17))
            # cache_cap = 7
            cache_cap = cache_cap * 8 * 1024 * 1024  # *8MB
            z0 = random.random()
            z1 = random.random()
            q = random.random()
            w = random.random()
            sum = z0 + z1 + q + w
            workload = [z0 / sum, z1 / sum, q / sum, w / sum]
            z0 = int(num_queries * workload[0])
            z1 = int(num_queries * workload[1])
            q = int(num_queries * workload[2])
            w = int(num_queries * workload[3])
            self.logger.info(f'Workload : {z0},{z1},{q},{w}')
            key_path = 'key_log'
            if not os.path.exists(key_path):
                os.makedirs(key_path)
            self.logger.info(f'Building DB at size : {n}')
            row, settings = (
                self.config['lsm_tree_config'].copy(),
                self.config['lsm_tree_config'].copy(),
            )
            settings['db_name'] = 'level_cost'
            settings['path_db'] = self.config['app']['DATABASE_PATH']
            settings['group'] = row['group'] = group
            settings['T'] = row['T'] = size_ratio
            settings['N'] = row['N'] = n
            settings['M'] = row['M'] = buffer + (bpe_budget * n)
            settings['h'] = row['h'] = bpe_budget
            settings['dist'] = row['dist'] = dist
            settings['skew'] = row['skew'] = skew
            settings['cache_cap'] = row['cache_cap'] = cache_cap
            settings['is_leveling_policy'] = row['is_leveling_policy'] = True
            row['mbuf'] = buffer / 8
            row['z0'] = z0
            row['z1'] = z1
            row['q'] = q
            row['w'] = w
            cf = CostFunction(
                settings['N'],
                settings['phi'],
                settings['s'],
                settings['B'],
                settings['E'],
                settings['M'],
                settings['is_leveling_policy'],
                z0,
                z1,
                q,
                w,
            )
            db = RocksDB(self.config)
            _ = db.init_database(**settings, bulk_stop_early=False)
            for step in range(100):
                self.logger.info('Running workload')
                key_log = key_path + '/{}_{}.dat'.format(group, step)
                settings['key_log'] = row['key_log'] = key_log
                results = db.run(
                    z0,
                    z1,
                    q,
                    w,
                    dist,
                    skew,
                    prime=100,
                    cache_cap=cache_cap,
                    key_log=key_log,
                )
                for key, val in results.items():
                    self.logger.info(f'{key} : {val}')
                    row[f'{key}'] = val
                row['N'] += w
                row['mbuf'] = buffer / 8
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
                row['read_model_io'] = cf.calculate_read_cost(
                    settings['h'], settings['T']
                )
                row['write_model_io'] = cf.calculate_write_cost(
                    settings['h'], settings['T']
                )
                row['read_io'] = row['blocks_read']
                row['model_io'] = row['read_model_io'] + row['write_model_io']
                self.logger.info('mbuf: {}'.format(row['mbuf']))
                self.logger.info('read_model_io: {}'.format(row['read_model_io']))
                self.logger.info('write_model_io: {}'.format(row['write_model_io']))
                self.logger.info('model_io: {}'.format(row['model_io']))
                df.append(row)
                self.de.export_csv_file(
                    pd.DataFrame(df),
                    'level_cost_ckpt.csv',
                )

        self.logger.info('Exporting data from level cost')
        df = pd.DataFrame(df)
        self.de.export_csv_file(df, 'level_cost.csv')
        self.logger.info('Finished level_cost\n')


if __name__ == "__main__":

    # Load configuration
    if len(sys.argv) > 1:
        config_yaml_path = sys.argv[1]
    else:
        config_yaml_path = os.path.join('endure/config/robust-lsm-trees.yaml')

    with open(config_yaml_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Start driver
    driver = Runner(config)
    driver.run(LevelCostDist(config))
