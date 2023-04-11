"""
Experiment 04
Scaling database experiment

OUTLINE:
    1. Fix database tuning
    2. Fix workload distribution
    3. Run workload on database tuning over different values of N (# of elment in DB)

"""
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
        num_queries = 100000
        # workload = (0.25, 0.25, 0.25, 0.25)
        workload = (0.0, 0.0, 0.0, 0.0)

        # z0 = int(np.ceil(num_queries * workload[0]))
        # z1 = int(np.ceil(num_queries * workload[1]))
        # q = int(np.ceil(num_queries  * workload[2]))
        # w = int(np.ceil(num_queries  * workload[3]))

        num_entries = (1e5, 3e5, 6e5, 1e6, 3e6, 6e6, 1e7, 3e7, 6e7, 1e8, 2e8)
        n = 1e6
        bpe_budget = 0.0
        buffer = 8 * 1024 * 1024 * 8  # MiB in bits
        size_ratio = 5

        df = []

        for i in range(512):
            # n = random.choice([1e6, 3e6, 5e6])
            # n = 3e6
            n = int(1e7 + (2e7 - 1e7) * np.random.random())
            # n = 14862756
            size_ratio = random.choice(range(4, 11))
            # size_ratio = 10
            z0 = random.random()
            z1 = random.random()
            q = random.random()
            w = random.random()
            sum = z0 + z1 + q + w
            workload = [z0 / sum, z1 / sum, q / sum, w / sum]
            # workload = [0.06, 0.39, 0.30, 0.25]
            dist = np.random.choice(['uniform', 'zipfian'], p=[0.3, 0.7])
            # dist = 'uniform'
            skew = random.random()
            # skew = 0.36
            bpe_budget = random.choice(range(1, 13))
            # bpe_budget = 4
            buffer = random.choice(range(1, 17))
            # buffer = 14
            buffer = buffer * 8 * 1024 * 1024 * 8  # *8MB
            # buffer = buffer * 1024 * 1024 * 2  # * 256KB
            cache_cap = random.choice(range(1, 17))
            # cache_cap = 16
            cache_cap = cache_cap * 8 * 1024 * 1024  # *8MB
            z0 = int(np.ceil(num_queries * workload[0]))
            z1 = int(np.ceil(num_queries * workload[1]))
            q = int(np.ceil(num_queries * workload[2]))
            w = int(np.ceil(num_queries * workload[3]))
            self.logger.info(
                f'Workload : {workload[0]},{workload[1]},{workload[2]},{workload[3]}'
            )
            key_path = 'key_log_tier_buffer'
            if not os.path.exists(key_path):
                os.makedirs(key_path)
            key_log = key_path + '/{}.dat'.format(i)
            self.logger.info(f'Building DB at size : {n}')
            row, settings = (
                self.config['lsm_tree_config'].copy(),
                self.config['lsm_tree_config'].copy(),
            )
            settings['db_name'] = 'tier_buffer_cost'
            settings['path_db'] = self.config['app']['DATABASE_PATH']
            settings['T'] = row['T'] = size_ratio
            settings['N'] = row['N'] = n
            settings['M'] = row['M'] = buffer + (bpe_budget * n)
            settings['h'] = row['h'] = bpe_budget
            settings['dist'] = row['dist'] = dist
            settings['skew'] = row['skew'] = skew
            settings['cache_cap'] = row['cache_cap'] = cache_cap
            settings['is_leveling_policy'] = row['is_leveling_policy'] = True
            settings['key_log'] = row['key_log'] = key_log

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
            print(cf.L(settings['h'], settings['T']))
            db = RocksDB(self.config)
            _ = db.init_database(**settings, bulk_stop_early=False)

            self.logger.info('Running workload')
            results = db.run(
                z0,
                z1,
                q,
                w,
                dist,
                skew,
                prime=10000,
                cache_cap=cache_cap,
                key_log=key_log,
            )
            for key, val in results.items():
                self.logger.info(f'{key} : {val}')
                row[f'{key}'] = val
            row['mbuf'] = buffer / 8
            row['L'] = cf.L(row['h'], row['T'])
            row['z0'] = workload[0]
            row['z1'] = workload[1]
            row['q'] = workload[2]
            row['w'] = workload[3]
            row['write_io'] = (
                row['bytes_written']
                + row['compact_read']
                + row['compact_write']
                + row['flush_written']
            ) / 4096
            self.logger.info('write_io: {}'.format(row['write_io']))
            row['read_model_io'] = cf.calculate_read_cost(settings['h'], settings['T'])
            row['write_model_io'] = cf.calculate_write_cost(
                settings['h'], settings['T']
            )
            row['model_io'] = row['read_model_io'] + row['write_model_io']
            self.logger.info('mbuf: {}'.format(row['mbuf']))
            self.logger.info('read_model_io: {}'.format(row['read_model_io']))
            self.logger.info('write_model_io: {}'.format(row['write_model_io']))
            self.logger.info('model_io: {}'.format(row['model_io']))
            df.append(row)
            self.de.export_csv_file(
                pd.DataFrame(df),
                'tier_cost_buffer_ckpt.csv',
            )

        self.logger.info('Exporting data from level cost')
        df = pd.DataFrame(df)
        self.de.export_csv_file(df, 'tier_cost_buffer.csv')
        self.logger.info('Finished tier_cost_cache\n')


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
