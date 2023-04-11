import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import KFold
import sys

sys.path.append('./lrkv')
from utils.lsm import estimate_fpr, estimate_level
from utils.distribution import dist_regression
from utils.model_lr import get_cache, get_tier_cost, get_cache_uniform
from sklearn.linear_model import LinearRegression

np.set_printoptions(suppress=True)

E = 1024
Q = 200000
B = 4
S = 2
M = 2147483648  # 256MB

all_samples = pd.read_csv('raw_data/tier_cost_lr_uniform.csv')
all_samples = all_samples.sample(frac=1)
# all_samples = all_samples[:20]

Xc = []
Yc = []
X = []
Y = []

eps = 1e-8
fold = 10
print('Start training')
for _, sample in all_samples.iterrows():
    if sample['read_io'] + sample['write_io'] == 0:
        continue
    if 'ratio' not in sample:
        sample['ratio'] = sample['mbuf'] * 8 / (M - sample['h'] * sample['N'])
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
        get_tier_cost(
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
    W = np.linalg.lstsq(X_train * weights[:, np.newaxis], Y_train * weights, rcond=-1)[
        0
    ]
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
        print('=' * 50)
        print(_y_hat, _y)
        print(_error, _rerror)
        errors.append(_error)
        rerrors.append(_rerror)
print('=' * 50)
print(np.mean(errors), np.mean(rerrors))
pkl.dump(Wcs, open('model/tier_cache_lr_uniform.pkl', "wb"))
pkl.dump(Ws, open('model/tier_cost_lr_uniform.pkl', "wb"))


# Wcs = pkl.load(open("model/tier_cache_lr.pkl", "rb"))
# Ws = pkl.load(open("model/tier_cost_lr.pkl", "rb"))
# print('Start inference')
# error = []
# rerror = []
# for _, sample in all_samples.iterrows():
#     y_costs = []
#     alpha, c = dist_regression(sample)
#     y_hats = []
#     for Wc, W in zip(Wcs, Ws):
#         xc = get_cache(
#             sample['T'],
#             sample['h'],
#             sample['ratio'],
#             alpha,
#             c,
#             sample['z0'],
#             sample['z1'],
#             sample['q'],
#             sample['w'],
#         )
#         yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
#         x = get_cost(
#             sample['T'],
#             sample['h'],
#             sample['ratio'],
#             sample['z0'],
#             sample['z1'],
#             sample['q'],
#             sample['w'],
#             yc,
#         )
#         y_hat = np.dot(x, W)
#         y_hats.append(y_hat)
#     var = np.var(y_hats)
#     y_hat = np.mean(y_hats)
#     y = sample['total_latency'] / sample['queries']
#     error.append(abs(y_hat - y))
#     rerror.append(abs(y_hat - y) / y)
#     print('=' * 50)
#     print(y_hat, y)
#     print(yc, sample['cache_hit_rate'])
#     print(abs(y_hat - y), abs(y_hat - y) / y)
# print(np.mean(error), np.mean(rerror))
