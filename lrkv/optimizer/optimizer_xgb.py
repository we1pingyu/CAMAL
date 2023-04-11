import numpy as np
import pandas as pd
import xgboost as xgb
import sys
import logging
import os
import yaml
import copy
import random
import pickle

sys.path.append('./lrkv')
from runner import Runner
from lsm_tree.PyRocksDB import RocksDB
from lsm_tree.cost_function import NominalCostFunction, LRCostFunction
from lsm_tree.tunner import NominalWorkloadTuning, LinearTuning
from utils.model_xgb import load_models, get_cache, get_cost
from utils.lsm import estimate_level
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
workloads = [
    (0.01, 0.01, 0.01, 0.97),
]

dists = ['zipfian', 'uniform']


def ensemble_infer(x_cache, cost_models):
    y_costs = []
    for cost_model in cost_models:
        y_cost = cost_model.predict([x_cache])[0]
        y_costs.append(y_cost)
    var = np.var(y_costs)
    y_hat = np.mean(y_costs)
    return y_hat, var


def simulated_annealing(
    # cache_models,
    cost_models,
    init_T,
    init_h,
    init_ratio,
    alpha,
    c,
    z0,
    z1,
    q,
    w,
    temperature=100,
    cooling_rate=0.99,
):
    current_T, current_h, current_ratio = [init_T, init_h, init_ratio]
    x_cache = get_cache(current_T, current_h, current_ratio, alpha, c, z0, z1, q, w)
    # y_cache = cache_model.predict([x_cache])[0]
    # x_cost = get_cost(
    # current_T, current_h, current_ratio, alpha, c, z0, z1, q, w, y_cache
    # )
    current_cost, _ = ensemble_infer(x_cache, cost_models)
    best_T, best_h, best_ratio = current_T, current_h, current_ratio
    best_cost = current_cost
    while temperature > 1e-8:
        new_T = current_T + random.choice([-1, 0, 1])
        new_h = current_h + random.choice([-1, 0, 1])
        new_ratio = current_ratio + random.choice([-0.1, 0, 0.1])
        if 2 <= new_T <= 16 and 8 <= new_h <= 13 and 0.25 <= new_ratio < 1.0:
            x_cache = get_cache(new_T, new_h, new_ratio, alpha, c, z0, z1, q, w)
            # y_cache = cache_model.predict([x_cache])[0]
            # x_cost = get_cost(new_T, new_h, new_ratio, alpha, c, z0, z1, q, w, y_cache)
            # new_cost = cost_model.predict([x_cost])[0]
            new_cost, _ = ensemble_infer(x_cache, cost_models)
            delta_cost = new_cost - current_cost
            if delta_cost < 0:
                current_T, current_h, current_ratio = new_T, new_h, new_ratio
                current_cost = new_cost
                if current_cost < best_cost:
                    best_T, best_h, best_ratio = current_T, current_h, current_ratio
                    best_cost = current_cost
            elif np.exp(-delta_cost / temperature) > random.uniform(0, 1):
                current_T, current_h, current_ratio = new_T, new_h, new_ratio
                current_cost = new_cost
            # print(f'T: {current_T}, h: {current_h}, ratio: {current_ratio}, cost: {current_cost}')
            temperature *= cooling_rate
    return best_T, best_h, best_ratio, best_cost


class Optimizer(object):
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger('rlt_logger')

    def run(self):
        i = 0
        df = []
        for work in workloads:
            for dist in dists:
                if dist == 'zipfian':
                    skews = [0.01, 0.33, 0.66, 0.99]
                else:
                    skews = [0.0]
                for skew in skews:
                    # nominal optimizer
                    row = self.config['lsm_tree_config'].copy()
                    row['optimizer'] = 'nominal'
                    row['db_name'] = 'level_optimizer'
                    row['path_db'] = self.config['app']['DATABASE_PATH']
                    z0, z1, q, w = work
                    cf = NominalCostFunction(
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
                    key_path = 'key_log_xgb_optimizer'
                    if not os.path.exists(key_path):
                        os.makedirs(key_path)
                    key_log = key_path + '/{}.dat'.format(i)
                    row['key_log'] = key_log
                    i += 1
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
                    row['read_model_io'] = Q * cf.calculate_read_cost(
                        row['h'], row['T']
                    )
                    row['write_model_io'] = Q * cf.calculate_write_cost(
                        row['h'], row['T']
                    )
                    row['model_io'] = row['read_model_io'] + row['write_model_io']
                    self.logger.info('mbuf: {}'.format(row['mbuf']))
                    self.logger.info('read_model_io: {}'.format(row['read_model_io']))
                    self.logger.info('write_model_io: {}'.format(row['write_model_io']))
                    self.logger.info('model_io: {}'.format(row['model_io']))
                    # print(row)
                    df.append(row)
                    pd.DataFrame(df).to_csv('optimizer_data/xgb_optimizer_ckpt.csv')

                    # learned optimizer
                    row = copy.deepcopy(row)
                    row['optimizer'] = 'al_xgb'
                    alpha ,c = dist_regression(row)
                    
                    # level_cache_models = load_models(
                    #     'model/al_level_cache_xgb_holder.pkl'
                    # )
                    level_cost_models = load_models(
                        'model/al_level_cost_xgb_holder.pkl'
                    )
                    # tier_cache_models = load_models(
                    #     'model/al_level_cache_xgb_holder.pkl'
                    # )
                    tier_cost_models = load_models('model/al_level_cost_xgb_holder.pkl')
                    best_T, best_h, best_ratio, best_cost = simulated_annealing(
                        # level_cache_models,
                        level_cost_models,
                        10,
                        10,
                        0.5,
                        alpha,
                        c,
                        z0,
                        z1,
                        q,
                        w,
                    )
                    row['is_leveling_policy'] = True

                    (
                        tier_best_T,
                        tier_best_h,
                        tier_best_ratio,
                        tier_best_cost,
                    ) = simulated_annealing(
                        # tier_cache_models,
                        tier_cost_models,
                        10,
                        10,
                        0.5,
                        alpha,
                        c,
                        z0,
                        z1,
                        q,
                        w,
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
                        f'xgb_optimizer: best_T: {best_T}, best_h: {best_h}, best_ratio: {best_ratio}, is_leveling_policy: {policy}, best_cost:{best_cost}'
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
                    pd.DataFrame(df).to_csv('optimizer_data/xgb_optimizer_ckpt.csv')

        self.logger.info('Exporting data from optimizer')
        pd.DataFrame(df).to_csv('optimizer_data/xgb_optimizer.csv')
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
