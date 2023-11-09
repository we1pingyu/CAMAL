import numpy as np
import pandas as pd
import pickle as pkl
import sys
import time
import yaml
import os
from sklearn.model_selection import KFold

sys.path.append('./lrkv')
from utils.lsm import estimate_fpr, estimate_level
from utils.distribution import dist_regression
from utils.model_lr import get_cache_uniform, get_level_cost, get_tier_cost
from sklearn.linear_model import LinearRegression

np.set_printoptions(suppress=True)

config_yaml_path = os.path.join('lrkv/config/config.yaml')
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
level_data = config['samples_path']['xgb_level_final']
tier_data = config['samples_path']['xgb_tier_final']
fold = 15
for num_sample in [100]:
    start_time = time.time()
    all_samples = pd.read_csv(level_data)
    # all_samples = all_samples.sample(frac=1)
    all_samples = all_samples[: num_sample * 15]
    Xc = []
    Yc = []
    X = []
    Y = []

    eps = 1e-8
    print('Start training')
    for _, sample in all_samples.iterrows():
        if sample['read_io'] + sample['write_io'] == 0:
            continue
        # print(1 - sample['cache_cap'] * 8 / M)
        # if 'ratio' not in sample:
        #     sample['ratio'] = 1 - sample['cache_cap'] * 8 / M
        xc = get_cache_uniform(
            sample['T'],
            sample['h'],
            sample['ratio'],
            sample['z0'],
            sample['z1'],
            sample['q'],
            sample['w'],
            sample['E'] / 8,
            sample['M'] / sample['ratio'],
            sample['N'],
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
                sample['cache_hit_rate'] + eps,
                sample['E'] / 8,
                sample['M'] / sample['ratio'],
                sample['N'],
            )
        )
        y = (sample['total_latency']) / sample['queries']
        Y.append(y)

    Xc = np.array(Xc)
    Yc = np.array(Yc)
    # print(Xc.shape)s
    Wc = np.linalg.lstsq(Xc, Yc, rcond=-1)[0]

    X = np.array(X)
    Y = np.array(Y)
    W = np.linalg.lstsq(X, Y, rcond=-1)[0]

    Wcs = []
    Ws = []
    kf = KFold(n_splits=fold)
    errors = []
    rerrors = []
    for train_index, test_index in kf.split(X):
        Xc_train = Xc[train_index]
        Yc_train = Yc[train_index]
        X_train = X[train_index]
        Y_train = Y[train_index]
        weights = 1 / Y_train
        Wc = np.linalg.lstsq(Xc_train, Yc_train, rcond=-1)[0]
        W = np.linalg.lstsq(
            X_train * weights[:, np.newaxis], Y_train * weights, rcond=-1
        )[0]
        # W = np.linalg.lstsq(X_train, Y_train, rcond=-1)[0]
        Wcs.append(Wc)
        Ws.append(W)
        Xc_test = Xc[test_index]
        yc = np.clip(np.exp(np.dot(Xc_test, Wc)), 0, 1)
        X_test = X[test_index]
        X_test[:, -1] = yc
        y_hat = np.dot(X_test, W)
        Y_test = Y[test_index]
        error = abs(y_hat - Y_test)
        rerror = abs(y_hat - Y_test) / Y_test
        for x, _y_hat, _y, _error, _rerror in zip(X_test, y_hat, Y_test, error, rerror):
            # print('=' * 50)
            # print(x)
            # print(_y_hat, _y)
            # print(_error, _rerror)
            errors.append(_error)
            rerrors.append(_rerror)
    print('=' * 50)
    print(np.mean(errors), np.mean(rerrors))
    pkl.dump(Wcs, open(f'model/level_lr_sim_cache_model.pkl', "wb"))
    pkl.dump(Ws, open(f'model/level_lr_sim_cost_model.pkl', "wb"))
    print(time.time() - start_time)

    all_samples = pd.read_csv(tier_data)
    all_samples = all_samples.sample(frac=1)
    all_samples = all_samples[: num_sample * 15]

    Xc = []
    Yc = []
    X = []
    Y = []

    eps = 1e-8
    print('Start training')
    for _, sample in all_samples.iterrows():
        if sample['read_io'] + sample['write_io'] == 0:
            continue
        # if 'ratio' not in sample:
        #     sample['ratio'] = 1 - sample['cache_cap'] * 8 / M
        # print(
        #     sample['E'] / 8,
        #     sample['M'] / sample['ratio'],
        #     sample['N'],
        # )
        xc = get_cache_uniform(
            sample['T'],
            sample['h'],
            sample['ratio'],
            sample['z0'],
            sample['z1'],
            sample['q'],
            sample['w'],
            sample['E'] / 8,
            sample['M'] / sample['ratio'],
            sample['N'],
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
                sample['cache_hit_rate'] + eps,
                sample['E'] / 8,
                sample['M'] / sample['ratio'],
                sample['N'],
            )
        )
        y = (sample['total_latency']) / sample['queries']
        Y.append(y)

    Xc = np.array(Xc)
    Yc = np.array(Yc)
    Wc = np.linalg.lstsq(Xc, Yc, rcond=-1)[0]

    X = np.array(X)
    Y = np.array(Y)
    W = np.linalg.lstsq(X, Y, rcond=-1)[0]

    Wcs = []
    Ws = []
    kf = KFold(n_splits=fold)
    errors = []
    rerrors = []
    for train_index, test_index in kf.split(X):
        Xc_train = Xc[train_index]
        Yc_train = Yc[train_index]
        X_train = X[train_index]
        Y_train = Y[train_index]
        weights = 1 / Y_train
        Wc = np.linalg.lstsq(Xc_train, Yc_train, rcond=-1)[0]
        W = np.linalg.lstsq(
            X_train * weights[:, np.newaxis], Y_train * weights, rcond=-1
        )[0]
        # W = np.linalg.lstsq(X_train, Y_train, rcond=-1)[0]
        Wcs.append(Wc)
        Ws.append(W)
        Xc_test = Xc[test_index]
        yc = np.clip(np.exp(np.dot(Xc_test, Wc)), 0, 1)
        X_test = X[test_index]
        X_test[:, -1] = yc
        y_hat = np.dot(X_test, W)
        Y_test = Y[test_index]
        error = abs(y_hat - Y_test)
        rerror = abs(y_hat - Y_test) / Y_test
        for _y_hat, _y, _error, _rerror in zip(y_hat, Y_test, error, rerror):
            # print('=' * 50)
            # print(_y_hat, _y)
            # print(_error, _rerror)
            errors.append(_error)
            rerrors.append(_rerror)
    print('=' * 50)
    print(np.mean(errors), np.mean(rerrors))
    pkl.dump(Wcs, open(f'model/tier_lr_sim_cache_model.pkl', "wb"))
    pkl.dump(Ws, open(f'model/tier_lr_sim_cost_model.pkl', "wb"))
