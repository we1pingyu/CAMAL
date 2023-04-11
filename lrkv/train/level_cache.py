import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

E = 1024
Qs = 100000
B = 4
S = 10


def L(N, mbuf, T, get_ceiling=True):
    l = np.log(((N * E) / mbuf) + 1) / np.log(T)
    if get_ceiling:
        l = np.ceil(l)
    return l


all_samples = pd.read_csv('data/level_cost_cache_ckpt.csv')
training_samples = all_samples[:400]
test_samples = all_samples[400:]
Xe = []
Ye = []
Xr = []
Yr = []
Xq = []
Yq = []

z0_hit_ratio = []
z1_hit_ratio = []
q_hit_ratio = []
for _, sample in training_samples.iterrows():
    N = sample['N']
    l = L(N, sample['mbuf'], sample['T'], False)
    D = sample['cache_cap'] / 1024
    fpr = np.exp(-1 * sample['h'] * (np.log(2) ** 2))
    data = np.zeros([int(N)])
    with open(sample['key_log'], 'r') as f:
        for line in f.readlines():
            last = ord(line.strip('\n')[-1])
            if last >= ord('A'):
                data[int(line.strip('\n')[:-1] + str(last - 65))] += 1
            else:
                data[int(line.strip('\n'))] += 1
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

    Q = sample['z0']
    if Q > 0.05:
        Nf = Q * Qs * l * fpr
        xe = [1]
        for power in (alpha, c, Nf, D, 1):
            for x in (alpha, c, Nf, D):
                xe.append(power * np.log(x))
        Xe.append(xe)
        Ye.append(np.log(sample['z0_cache_hit_rate']))

    Q = sample['z1']
    if Q > 0.05 and sample['z1_cache_hit_rate'] < 1:
        Nf1 = Q * Qs * (l * fpr)
        Nf2 = Q * Qs
        xr = [1]
        for power in (alpha, c, Nf1, Nf2, D, 1):
            for x in (alpha, c, Nf1, Nf2, D):
                xr.append(power * np.log(x))
        Xr.append(xr)
        Yr.append(np.log(sample['z1_cache_hit_rate']))

    Q = sample['q']
    if Q > 0.05:
        Nf1 = Q * Qs * l
        Nf2 = Q * Qs * (S / B)
        xq = [1]
        for power in (alpha, c, Nf1, Nf2, D, 1):
            for x in (alpha, c, Nf1, Nf2, D):
                xq.append(power * np.log(x))
        Xq.append(xq)
        Yq.append(np.log(sample['w_cache_hit_rate']))

    Q = sample['w']
    if Q > 0.05:
        Nf1 = Q * Qs * l
        Nf2 = Q * Qs * (S / B)
        xq = [1]
        for power in (alpha, c, Nf1, Nf2, D, 1):
            for x in (alpha, c, Nf1, Nf2, D):
                xq.append(power * np.log(x))
        Xw.append(xq)
        Yw.append(np.log(sample['q_cache_hit_rate']))

print('Start training cache exp')
Xe = np.array(Xe)
Ye = np.array(Ye)
We = np.linalg.lstsq(Xe, Ye)[0]

Xr = np.array(Xr)
Yr = np.array(Yr)
Wr = np.linalg.lstsq(Xr, Yr)[0]

Xq = np.array(Xq)
Yq = np.array(Yq)
Wq = np.linalg.lstsq(Xq, Yq)[0]

errore = []
errorr = []
errorq = []
for _, sample in test_samples.iterrows():
    N = sample['N']
    l = L(N, sample['mbuf'], sample['T'], False)
    fpr = np.exp(-1 * sample['h'] * (np.log(2) ** 2))
    D = sample['cache_cap'] / 1024
    data = np.zeros([int(N)])
    with open(sample['key_log'], 'r') as f:
        for line in f.readlines():
            last = ord(line.strip('\n')[-1])
            if last >= ord('A'):
                data[int(line.strip('\n')[:-1] + str(last - 65))] += 1
            else:
                data[int(line.strip('\n'))] += 1
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

    Q = sample['z0']
    if Q > 0.05:
        Nf = Q * Qs * l * fpr
        xe = [1]
        for power in (alpha, c, Nf, D, 1):
            for x in (alpha, c, Nf, D):
                xe.append(power * np.log(x))
        ye = np.dot(xe, We)
        # print(np.exp(ye), sample['z0_cache_hit_rate'])
        errore.append(abs(np.exp(ye) - sample['z0_cache_hit_rate']))

    Q = sample['z1']
    if Q > 0.05 and sample['z1_cache_hit_rate'] < 1:
        Nf1 = Q * Qs * (l * fpr)
        Nf2 = Q * Qs
        xr = [1]
        for power in (alpha, c, Nf1, Nf2, D, 1):
            for x in (alpha, c, Nf1, Nf2, D):
                xr.append(power * np.log(x))
        yr = np.dot(xr, Wr)
        # print(np.exp(yr), sample['z1_cache_hit_rate'])
        errorr.append(abs(np.exp(yr) - sample['z1_cache_hit_rate']))

    Q = sample['q']
    if Q > 0.05:
        Nf1 = Q * Qs * l
        Nf2 = Q * Qs * (S / B)
        xq = [1]
        for power in (alpha, c, Nf1, Nf2, D, 1):
            for x in (alpha, c, Nf1, Nf2, D):
                xq.append(power * np.log(x))
        yq = np.dot(xq, Wq)
        print(np.exp(yq), sample['q_cache_hit_rate'])
        errorq.append(abs(np.exp(yq) - sample['q_cache_hit_rate']))

print(np.mean(errore))
print(np.mean(errorr))
print(np.mean(errorq))
