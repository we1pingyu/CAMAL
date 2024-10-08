import numpy as np
import pandas as pd
import sys
import logging
import os
import yaml
import copy
import random
import pickle as pkl

sys.path.append('./lrkv')
from runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.cost_function import CostFunction
from lsm_tree.tunner import NominalWorkloadTuning
from utils.model_lr import (
    simulated_annealing,
    traverse_optimizer,
    traverse_var_optimizer_uniform,
)
from utils.lsm import filt_memory_level_equation

np.set_printoptions(suppress=True)

N = 1e7
E = 1024
Q = 200000
B = 4
S = 2
M = 2147483648 * 2  # 512MB
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
# workloads = [
#     (0.01, 0.01, 0.01, 0.97),
# ]

dists = ['uniform']


class Optimizer(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('rlt_logger')

    def run(self):
        i = -1
        df = []
        # workloads = [(0.01, 0.01, 0.97, 0.01), (0.01, 0.01, 0.01, 0.97)]
        for workload in workloads:
            i += 1
            # workload = [random.random() for _ in range(4)]
            # workload = [w / sum(workload) for w in workload]
            dist = 'uniform'
            skew = 0.0
            # nominal optimizer
            row = self.config['lsm_tree_config'].copy()
            row['optimizer'] = 'nominal'
            row['db_name'] = 'level_optimizer'
            row['path_db'] = self.config['app']['DATABASE_PATH']
            z0, z1, q, w = workload
            cf = CostFunction(
                N,
                1,
                0.0000002,
                B,
                E,
                M,
                True,
                z0,
                z1,
                q,
                w,
            )
            nominal = NominalWorkloadTuning(cf)
            nominal_design = nominal.get_nominal_design(is_leveling_policy=None)
            print(f'nominal_optimizer: {nominal_design}')
            row['T'] = int(nominal_design['T'])
            row['N'] = N
            row['queries'] = Q
            row['M'] = M
            row['h'] = int(nominal_design['M_filt'] / N)
            row['dist'] = dist
            row['skew'] = skew
            row['cache_cap'] = 0.0
            row['is_leveling_policy'] = nominal_design['is_leveling_policy']
            row['mbuf'] = nominal_design['M_buff'] / 8
            row['z0'] = z0
            row['z1'] = z1
            row['q'] = q
            row['w'] = w
            key_path = 'key_log_lr_optimizer_w'
            if not os.path.exists(key_path):
                os.makedirs(key_path)
            key_log = key_path + '/{}.dat'.format(i)
            row['key_log'] = key_log
            self.logger.info(f'Building DB at size : {N}')
            self.config = config
            db = RocksDB(self.config)
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
                Q,
                is_leveling_policy=row['is_leveling_policy'],
                cache_cap=0,
                key_log=key_log,
            )
            for key, val in results.items():
                self.logger.info(f'{key} : {val}')
                row[f'{key}'] = val
            row['write_io'] = (
                row['bytes_written']
                + row['compact_read']
                + row['compact_write']
                + row['flush_written']
            ) / 4096
            self.logger.info('write_io: {}'.format(row['write_io']))
            row['read_model_io'] = Q * cf.calculate_read_cost(row['h'], row['T'])
            row['write_model_io'] = Q * cf.calculate_write_cost(row['h'], row['T'])
            row['model_io'] = row['read_model_io'] + row['write_model_io']
            self.logger.info('mbuf: {}'.format(row['mbuf']))
            self.logger.info('read_model_io: {}'.format(row['read_model_io']))
            self.logger.info('write_model_io: {}'.format(row['write_model_io']))
            self.logger.info('model_io: {}'.format(row['model_io']))
            # print(row)
            df.append(row)
            pd.DataFrame(df).to_csv('optimizer_data/lr_optimizer_uniform_ckpt.csv')

            # nominal + default cache optimizer
            row = copy.deepcopy(row)
            row['optimizer'] = 'nominal_cache'
            row['db_name'] = 'level_optimizer'
            row['path_db'] = self.config['app']['DATABASE_PATH']
            z0, z1, q, w = workload
            cache_cap = 16 * 1024 * 1024 * 8
            cf = CostFunction(
                N,
                1,
                0.0000002,
                B,
                E,
                M - cache_cap,
                True,
                z0,
                z1,
                q,
                w,
            )
            nominal = NominalWorkloadTuning(cf)
            nominal_design = nominal.get_nominal_design(is_leveling_policy=None)
            print(f'nominal_cache_optimizer: {nominal_design}')
            row['T'] = int(nominal_design['T'])
            row['N'] = N
            row['queries'] = Q
            row['M'] = M - cache_cap
            row['h'] = int(nominal_design['M_filt'] / N)
            row['dist'] = dist
            row['skew'] = skew
            row['cache_cap'] = cache_cap / 8
            row['is_leveling_policy'] = nominal_design['is_leveling_policy']
            row['mbuf'] = nominal_design['M_buff'] / 8
            row['z0'] = z0
            row['z1'] = z1
            row['q'] = q
            row['w'] = w
            key_path = 'key_log_xgb_optimizer'
            if not os.path.exists(key_path):
                os.makedirs(key_path)
            key_log = key_path + '/{}.dat'.format(i)
            row['key_log'] = key_log
            self.logger.info(f'Building DB at size : {N}')
            self.config = config
            db = RocksDB(self.config)
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
                Q,
                is_leveling_policy=row['is_leveling_policy'],
                cache_cap=0,
                key_log=key_log,
            )
            for key, val in results.items():
                self.logger.info(f'{key} : {val}')
                row[f'{key}'] = val
            row['write_io'] = (
                row['bytes_written']
                + row['compact_read']
                + row['compact_write']
                + row['flush_written']
            ) / 4096
            self.logger.info('write_io: {}'.format(row['write_io']))
            row['read_model_io'] = Q * cf.calculate_read_cost(row['h'], row['T'])
            row['write_model_io'] = Q * cf.calculate_write_cost(row['h'], row['T'])
            row['model_io'] = row['read_model_io'] + row['write_model_io']
            self.logger.info('mbuf: {}'.format(row['mbuf']))
            self.logger.info('read_model_io: {}'.format(row['read_model_io']))
            self.logger.info('write_model_io: {}'.format(row['write_model_io']))
            self.logger.info('model_io: {}'.format(row['model_io']))
            # print(row)
            df.append(row)
            pd.DataFrame(df).to_csv('optimizer_data/lr_optimizer_uniform_ckpt.csv')

            # rocksdb default setting
            row = copy.deepcopy(row)
            row['optimizer'] = 'rocksdb'
            row['is_leveling_policy'] = True
            row['T'] = 10
            row['h'] = 10
            best_h = 10
            row['cache_cap'] = 16 * 1024 * 1024
            best_ratio = 1 - 16 * 1024 * 1024 * 8 / (M - best_h * N)
            row['M'] = 10 * N + best_ratio * (M - best_h * N)
            self.logger.info(f'Building DB at size : {N}')
            db = RocksDB(self.config)
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
                Q,
                is_leveling_policy=row['is_leveling_policy'],
                cache_cap=row['cache_cap'],
                key_log=key_log,
            )
            for key, val in results.items():
                self.logger.info(f'{key} : {val}')
                row[f'{key}'] = val
            row['write_io'] = (
                row['bytes_written']
                + row['compact_read']
                + row['compact_write']
                + row['flush_written']
            ) / 4096
            self.logger.info('write_io: {}'.format(row['write_io']))
            self.logger.info('mbuf: {}'.format(row['mbuf']))
            # print(row)
            df.append(row)
            pd.DataFrame(df).to_csv('optimizer_data/xgb_optimizer_ckpt.csv')

            # learned optimizer
            for num_samples in [20]:
                row = copy.deepcopy(row)
                row['optimizer'] = 'lr'
                level_cost_models = pkl.load(
                    open(f"model/level_cost_lr_uniform_{num_samples}.pkl", "rb")
                )
                level_cache_models = pkl.load(
                    open(f"model/level_cache_lr_uniform_{num_samples}.pkl", "rb")
                )
                tier_cost_models = pkl.load(
                    open(f"model/tier_cost_lr_uniform_{num_samples}.pkl", "rb")
                )
                tier_cache_models = pkl.load(
                    open(f"model/tier_cache_lr_uniform_{num_samples}.pkl", "rb")
                )
                (
                    best_T,
                    best_h,
                    best_ratio,
                    best_var,
                    best_cost,
                ) = traverse_var_optimizer_uniform(
                    level_cache_models, level_cost_models, z0, z1, q, w, N=N
                )
                row['is_leveling_policy'] = True

                (
                    tier_best_T,
                    tier_best_h,
                    tier_best_ratio,
                    tier_best_var,
                    tier_best_cost,
                ) = traverse_var_optimizer_uniform(
                    tier_cache_models,
                    tier_cost_models,
                    z0,
                    z1,
                    q,
                    w,
                    policy='tier',
                    N=N,
                )
                print(
                    f'level_optimizer: best_T: {best_T}, best_h: {best_h}, best_ratio: {best_ratio},best_var: {best_var}, best_cost:{best_cost}'
                )
                print(
                    f'tier_optimizer: best_T: {tier_best_T}, best_h: {tier_best_h}, best_ratio: {tier_best_ratio}, best_var: {tier_best_var}, best_cost:{tier_best_cost}'
                )
                if tier_best_cost < best_cost:
                    row['is_leveling_policy'] = False
                    best_T, best_h, best_ratio, best_cost = (
                        tier_best_T,
                        tier_best_h,
                        tier_best_ratio,
                        tier_best_cost,
                    )
                policy = row['is_leveling_policy']
                print(
                    f'lr_optimizer: best_T: {best_T}, best_h: {best_h}, best_ratio: {best_ratio}, is_leveling_policy: {policy}, best_cost:{best_cost}'
                )
                min_err = 1e9
                for h in range(2, 20):
                    err = filt_memory_level_equation(h, best_h, 2)
                    if err < min_err:
                        min_err = err
                        temp = h
                best_h = temp    
                row['T'] = int(best_T)
                row['h'] = int(best_h)
                row['M'] = best_h * N + best_ratio * (M - best_h * N)
                row['cache_cap'] = (1 - best_ratio) * (M - best_h * N) / 8
                self.logger.info(f'Building DB at size : {N}')
                db = RocksDB(self.config)
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
                    Q,
                    is_leveling_policy=row['is_leveling_policy'],
                    cache_cap=row['cache_cap'],
                    key_log=key_log,
                )
                for key, val in results.items():
                    self.logger.info(f'{key} : {val}')
                    row[f'{key}'] = val
                row['write_io'] = (
                    row['bytes_written']
                    + row['compact_read']
                    + row['compact_write']
                    + row['flush_written']
                ) / 4096
                self.logger.info('write_io: {}'.format(row['write_io']))
                self.logger.info('mbuf: {}'.format(row['mbuf']))
                # print(row)
                df.append(row)
                pd.DataFrame(df).to_csv('optimizer_data/lr_optimizer_uniform_ckpt.csv')

        self.logger.info('Exporting data from lr optimizer')
        pd.DataFrame(df).to_csv('optimizer_data/lr_optimizer_uniform_512.csv')
        self.logger.info('Finished optimizer\n')


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
    driver.run(Optimizer(config))
