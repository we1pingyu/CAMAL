import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import expn
from torch import nn
import torch

np.set_printoptions(suppress=True)


N = 1e6
E = 1024 * 8


def L(N, M, h, T, get_ceiling=True):
    """L(x) function from Eq. 38
    with h = x / N

    :param h:
    :param T:
    """
    mbuff = M - (h * N)
    l = np.log(((N * E) / mbuff) + 1) / np.log(T)
    if get_ceiling:
        l = np.ceil(l)

    return l


all_samples = pd.read_csv('data/linear_model_all_random_level_500.csv')
training_samples = all_samples[:200]
test_samples = all_samples[200:]
print(training_samples.head())
X = []
Y = []
Q = 100000
B = 4
S = 10

X = []
Y = []
z0_hit_ratio = []
z1_hit_ratio = []
q_hit_ratio = []
for _, sample in training_samples.iterrows():
    Q = sample['z0']
    if Q < 0.05:
        continue
    N = sample['N']
    data = np.zeros([int(N)])
    with open(sample['key_log'], 'r') as f:
        for l in f.readlines():
            last = ord(l.strip('\n')[-1])
            if last >= ord('A'):
                data[int(l.strip('\n')[:-1] + str(last - 65))] += 1
            else:
                data[int(l.strip('\n'))] += 1
    data = np.sort(np.squeeze(data[np.argwhere(data)]))[::-1]
    zipf_X = []
    zipf_Y = []
    for k, d in enumerate(data):
        x0 = np.log(k + 1)
        x1 = 1
        zipf_X.append([x0, x1])
        zipf_Y.append(np.log(d))
    zipf_X = np.array(zipf_X)
    zipf_Y = np.array(zipf_Y)
    alpha, c = np.linalg.lstsq(zipf_X, zipf_Y)[0]
    alpha = -alpha
    # print(sample['skew'], alpha, c)
    D = sample['cache_cap'] / 1024
    fpr = np.exp(-1 * sample['h'] * (np.log(2) ** 2))
    Nf = Q * N * sample['L'] * fpr
    delta = np.log(D) / np.log(Nf)
    x0 = alpha * delta * delta
    x1 = alpha * alpha * delta
    x2 = alpha * delta
    x3 = alpha * alpha
    x4 = delta * delta
    x5 = alpha
    x6 = delta
    x7 = alpha * alpha * delta * delta
    X.append([x0, x1, x2, x3, x4, x5, x6, x7, 1])
    Y.append(sample['z0_cache_hit_rate'])
X = np.array(X)
Y = np.array(Y)
beta_hat = np.linalg.lstsq(X, Y)[0]
print(beta_hat)
print([x0, x1, x2, x3, x4, x5, x6, x7, 1] * beta_hat)

error = 0
for _, sample in test_samples.iterrows():
    Q = sample['z0']
    if Q < 0.05:
        continue
    N = sample['N']
    data = np.zeros([int(N)])
    with open(sample['key_log'], 'r') as f:
        for l in f.readlines():
            last = ord(l.strip('\n')[-1])
            if last >= ord('A'):
                data[int(l.strip('\n')[:-1] + str(last - 65))] += 1
            else:
                data[int(l.strip('\n'))] += 1
    data = np.sort(np.squeeze(data[np.argwhere(data)]))[::-1]
    zipf_X = []
    zipf_Y = []
    for k, d in enumerate(data):
        x0 = np.log(k + 1)
        x1 = 1
        zipf_X.append([x0, x1])
        zipf_Y.append(np.log(d))
    zipf_X = np.array(zipf_X)
    zipf_Y = np.array(zipf_Y)
    alpha, c = np.linalg.lstsq(zipf_X, zipf_Y)[0]
    alpha = -alpha
    D = sample['cache_cap'] / 1024
    fpr = np.exp(-1 * sample['h'] * (np.log(2) ** 2))
    Nf = Q * N * sample['L'] * fpr
    delta = np.log(D) / np.log(Nf)
    x0 = alpha * delta * delta
    x1 = alpha * alpha * delta
    x2 = alpha * delta
    x3 = alpha * alpha
    x4 = delta * delta
    x5 = alpha
    x6 = delta
    x7 = alpha * alpha * delta * delta
    y_hat = np.dot([x0, x1, x2, x3, x4, x5, x6, x7, 1], beta_hat)
    print(sample['key_log'], y_hat, sample['z0_cache_hit_rate'])
    error += abs(y_hat - sample['z0_cache_hit_rate'])
print(error / len(test_samples))
