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
from scipy.special import rel_entr
from runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.cost_function import CostFunction
from lsm_tree.tunner import NominalWorkloadTuning
from lsm_tree.workload_uncertainty import WorkloadUncertainty
from utils.model_lr import (
    traverse_var_optimizer_uniform,
    traverse_var_optimizer_uniform_uncertainty,
)
from utils.distribution import dist_regression

np.set_printoptions(suppress=True)

N = 1e7
E = 1024
Q = 200000
B = 4
S = 2
M = 2147483648  # 256MB
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

dists = ['uniform']


class Optimizer(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('rlt_logger')

    def run(self):
        i = -1
        df = []
        rhos = np.arange(0, 4.5, 0.5)
        # workloads = [(0.01, 0.01, 0.97, 0.01), (0.01, 0.01, 0.01, 0.97)]
        for workload in workloads:
            for rho in rhos:
                i += 1
                while True:
                    uncertain_workload = [random.random() for _ in range(4)]
                    uncertain_workload = [
                        w / sum(uncertain_workload) for w in uncertain_workload
                    ]
                    if (
                        min(0, rho - 0.5)
                        < np.sum(rel_entr(uncertain_workload, workload))
                        < rho + 0.5
                    ):
                        break

                dist = 'uniform'
                skew = 0.0
                z0, z1, q, w = uncertain_workload

                row = self.config['lsm_tree_config'].copy()
                row['db_name'] = 'level_optimizer'
                row['path_db'] = self.config['app']['DATABASE_PATH']
                row['rho'] = rho
                # robust optimizer
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
                row['optimizer'] = 'robust'
                robust = WorkloadUncertainty(cf)
                tiering_design = robust.get_robust_tiering_design(
                    rho, nominal_design=None
                )
                leveling_design = robust.get_robust_leveling_design(
                    rho, nominal_design=None
                )
                if tiering_design['obj'] < leveling_design['obj']:
                    robust_design = tiering_design
                else:
                    robust_design = leveling_design
                print(f'robust_optimizer: {robust_design}')
                row['T'] = int(robust_design['T'])
                row['N'] = N
                row['queries'] = Q
                row['M'] = M
                row['h'] = int(robust_design['M_filt'] / N)
                row['dist'] = dist
                row['skew'] = skew
                row['cache_cap'] = 0.0
                row['is_leveling_policy'] = robust_design['is_leveling_policy']
                row['mbuf'] = robust_design['M_buff'] / 8
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

                # learned optimizer
                sampling_workloads = []
                while len(sampling_workloads) < 10:
                    sampling_workload = [random.random() for _ in range(4)]
                    sampling_workload = [
                        w / sum(sampling_workload) for w in sampling_workload
                    ]
                    if (
                        min(0, rho - 0.5)
                        < np.sum(rel_entr(sampling_workload, workload))
                        < rho + 0.5
                    ):
                        sampling_workloads.append(sampling_workload)

                for num_samples in [20]:
                    temp_z0, temp_z1, temp_q, temp_w = workload
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
                        var,
                        best_cost,
                    ) = traverse_var_optimizer_uniform(
                        level_cache_models,
                        level_cost_models,
                        temp_z0,
                        temp_z1,
                        temp_q,
                        temp_w,
                        N=N,
                    )
                    row['is_leveling_policy'] = True

                    (
                        tier_best_T,
                        tier_best_h,
                        tier_best_ratio,
                        var,
                        tier_best_cost,
                    ) = traverse_var_optimizer_uniform(
                        tier_cache_models,
                        tier_cost_models,
                        temp_z0,
                        temp_z1,
                        temp_q,
                        temp_w,
                        policy='tier',
                        N=N,
                    )
                    print(
                        f'level_optimizer: best_T: {best_T}, best_h: {best_h}, best_ratio: {best_ratio}, best_cost:{best_cost}'
                    )
                    print(
                        f'tier_optimizer: best_T: {tier_best_T}, best_h: {tier_best_h}, best_ratio: {tier_best_ratio}, best_cost:{tier_best_cost}'
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
                    pd.DataFrame(df).to_csv(
                        'optimizer_data/lr_optimizer_uniform_ckpt.csv'
                    )

                    # learned roubust optimizer
                    row = copy.deepcopy(row)
                    row['optimizer'] = 'lr_robust'
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
                        best_cost,
                    ) = traverse_var_optimizer_uniform_uncertainty(
                        level_cache_models, level_cost_models, sampling_workloads, N=N
                    )
                    row['is_leveling_policy'] = True

                    (
                        tier_best_T,
                        tier_best_h,
                        tier_best_ratio,
                        tier_best_cost,
                    ) = traverse_var_optimizer_uniform_uncertainty(
                        tier_cache_models,
                        tier_cost_models,
                        sampling_workloads,
                        policy='tier',
                        N=N,
                    )
                    print(
                        f'level_optimizer: best_T: {best_T}, best_h: {best_h}, best_ratio: {best_ratio}, best_cost:{best_cost}'
                    )
                    print(
                        f'tier_optimizer: best_T: {tier_best_T}, best_h: {tier_best_h}, best_ratio: {tier_best_ratio}, best_cost:{tier_best_cost}'
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
                        f'lr_robust_optimizer: best_T: {best_T}, best_h: {best_h}, best_ratio: {best_ratio}, is_leveling_policy: {policy}, best_cost:{best_cost}'
                    )
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
                    pd.DataFrame(df).to_csv(
                        'optimizer_data/lr_optimizer_uniform_ckpt.csv'
                    )

        self.logger.info('Exporting data from lr optimizer')
        pd.DataFrame(df).to_csv('optimizer_data/lr_optimizer_uniform_uncertainty.csv')
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
