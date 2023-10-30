import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import sys
import random
import time
import os
from datetime import datetime

sys.path.append('./lrkv')
from sklearn.model_selection import train_test_split
from utils.model_xgb import get_cost_uniform
from utils.lsm import *
from sklearn.model_selection import KFold

np.set_printoptions(suppress=True)

scaling = 10
E = 1000
Q = int(200000 / scaling)
B = 4
S = 2
M = 2147483648 / scaling  # 256MB

level_data = 'raw_data/samples_sim_xgb_level_uniform_final.csv'
tier_data = 'raw_data/samples_sim_xgb_tier_uniform_final.csv'
fold = 15

for num_sample in [100]:
    start_time = time.time()
    print('Start level training')
    all_samples = pd.read_csv(level_data)
    timestamp = os.path.getctime(level_data)
    creation_time = datetime.fromtimestamp(timestamp)
    print(creation_time)
    all_samples = all_samples.sample(frac=1)
    all_samples = all_samples[: num_sample * 15]
    print(len(all_samples))
    X = []
    Y = []
    for _, sample in all_samples.iterrows():
        if sample['read_io'] + sample['write_io'] == 0:
            continue
        # if 'ratio' not in sample:
        #     sample['ratio'] = sample['mbuf'] * 8 / (M - sample['h'] * sample['N'])
        # sample['ratio'] = 1 - sample['cache_cap'] * 8 / M
        # print(sample['ratio'])
        X.append(
            get_cost_uniform(
                sample['T'],
                sample['h'],
                sample['ratio'],
                sample['z0'],
                sample['z1'],
                sample['q'],
                sample['w'],
                M,
                sample['N'],
            )
        )
        y = sample['total_latency'] / sample['queries']
        Y.append(y)

    eps = 1e-8
    regrs = []
    X = np.array(X)
    Y = np.array(Y)

    kf = KFold(n_splits=fold)
    errors = []
    rerrors = []
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        Y_train = Y[train_index]
        weights = 1 / Y_train
        regr = xgb.XGBRegressor(learning_rate=1)
        # Train the XGBoost cache model
        regr.fit(X_train, Y_train)
        X_test = X[test_index]
        Y_test = Y[test_index]
        y_hat = regr.predict(X_test)
        error = abs(y_hat - Y_test)
        rerror = abs(y_hat - Y_test) / Y_test
        for _y_hat, _y, _error, _rerror in zip(y_hat, Y_test, error, rerror):
            errors.append(_error)
            rerrors.append(_rerror)
        regrs.append(regr)
    print(np.mean(errors), np.mean(rerrors))
    pickle.dump(regrs, open(f'model/level_cost_xgb_uni_sim.pkl', "wb"))
    print(time.time() - start_time)

    all_samples = pd.read_csv(tier_data)
    timestamp = os.path.getctime(tier_data)
    creation_time = datetime.fromtimestamp(timestamp)
    print(creation_time)
    all_samples = all_samples.sample(frac=1)
    all_samples = all_samples[: num_sample * 15]
    X = []
    Y = []
    for _, sample in all_samples.iterrows():
        if sample['read_io'] + sample['write_io'] == 0:
            continue
        # if 'ratio' not in sample:
        #     sample['ratio'] = 1 - sample['cache_cap'] * 8 / M
        # print(sample['ratio'])
        X.append(
            get_cost_uniform(
                sample['T'],
                sample['h'],
                sample['ratio'],
                sample['z0'],
                sample['z1'],
                sample['q'],
                sample['w'],
                M,
                sample['N'],
            )
        )
        y = sample['total_latency'] / sample['queries']
        Y.append(y)
    print(len(X))

    print('Start tier training')
    regrs = []
    X = np.array(X)
    Y = np.array(Y)

    kf = KFold(n_splits=fold)
    errors = []
    rerrors = []
    for train_index, test_index in kf.split(X):
        X_train = X[train_index]
        Y_train = Y[train_index]
        weights = 1 / Y_train
        regr = xgb.XGBRegressor(learning_rate=1)

        # Train the XGBoost cache model
        regr.fit(X_train, Y_train)
        X_test = X[test_index]
        Y_test = Y[test_index]
        y_hat = regr.predict(X_test)
        error = abs(y_hat - Y_test)
        rerror = abs(y_hat - Y_test) / Y_test
        for _y_hat, _y, _error, _rerror in zip(y_hat, Y_test, error, rerror):
            errors.append(_error)
            rerrors.append(_rerror)
        regrs.append(regr)
    print(np.mean(errors), np.mean(rerrors))
    pickle.dump(regrs, open(f'model/tier_cost_xgb_uni_sim.pkl', "wb"))
