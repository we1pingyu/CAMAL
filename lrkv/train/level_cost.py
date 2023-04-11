import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

E = 1024
Q = 100000
B = 4
S = 10


def L(N, mbuf, T, get_ceiling=True):
    l = np.log(((N * E) / mbuf) + 1) / np.log(T)
    if get_ceiling:
        l = np.ceil(l)
    return l


all_samples = pd.read_csv('data/tier_cost_buffer_ckpt.csv')
training_samples = all_samples[:10]
test_samples = all_samples[:10]

Xe = []
Ye = []
Xr = []
Yr = []
Xq = []
Yq = []
Xw = []
Yw = []
X = []
Y = []

z0_hit_ratio = []
z1_hit_ratio = []
q_hit_ratio = []
for _, sample in training_samples.iterrows():
    if sample['blocks_read'] + sample['write_io'] == 0:
        continue
    l = L(sample['N'], sample['mbuf'], sample['T'], False)
    D = sample['cache_cap'] / 1024
    fpr = np.exp(-1 * sample['h'] * (np.log(2) ** 2))
    data = np.zeros([int(sample['N'])])
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
    alpha, c = np.linalg.lstsq(zipf_X, zipf_Y, rcond=-1)[0]
    alpha = -alpha

    if sample['z0'] > 0.05:
        Nf = Q * sample['z0'] * l * fpr
        xe = [1]
        for power in (alpha, c, Nf, D, 1):
            for x in (alpha, c, Nf, D):
                xe.append(power * np.log(x))
        Xe.append(xe)
        Ye.append(np.log(sample['z0_cache_hit_rate']))

    if sample['z1'] > 0.05 and sample['z1_cache_hit_rate'] < 1:
        Nf1 = Q * sample['z1'] * (l * fpr)
        Nf2 = Q * sample['z1']
        xr = [1]
        for power in (alpha, c, Nf1, Nf2, D, 1):
            for x in (alpha, c, Nf1, Nf2, D):
                xr.append(power * np.log(x))
        Xr.append(xr)
        Yr.append(np.log(sample['z1_cache_hit_rate']))

    if sample['q'] > 0.05:
        Nf1 = Q * sample['q'] * l
        Nf2 = Q * sample['q'] * (S / B)
        xq = [1]
        for power in (alpha, c, Nf1, Nf2, D, 1):
            for x in (alpha, c, Nf1, Nf2, D):
                xq.append(power * np.log(x))
        Xq.append(xq)
        Yq.append(np.log(sample['q_cache_hit_rate']))

    if sample['w'] > 0.05:
        Nf1 = Q * sample['w'] * l * sample['T']
        Nf2 = Q * sample['w'] * l
        xw = [1]
        for power in (alpha, c, Nf1, Nf2, D, 1):
            for x in (alpha, c, Nf1, Nf2, D):
                xw.append(power * np.log(x))
        Xw.append(xw)
        Yw.append(np.log(sample['w_cache_hit_rate']))

    z0 = sample['z0'] * l * fpr * (1 - sample['z0_cache_hit_rate'])
    z11 = sample['z1'] * (1 - sample['z1_cache_hit_rate'])
    z12 = sample['z1'] * l * fpr * (1 - sample['z1_cache_hit_rate'])
    q1 = sample['q'] * l * (1 - sample['q_cache_hit_rate'])
    q2 = sample['q'] * (S / B) * (1 - sample['q_cache_hit_rate'])
    w0 = sample['w'] * l * sample['T'] / B * (1 - sample['w_cache_hit_rate'])
    w1 = sample['w'] * l * (1 - sample['w_cache_hit_rate'])
    w2 = sample['w'] * sample['mbuf'] * (1 - sample['w_cache_hit_rate'])
    w3 = sample['w'] * (1 - sample['w_cache_hit_rate'])
    w4 = sample['mbuf'] * (1 - sample['w_cache_hit_rate'])
    w5 = l * sample['T'] / B * (1 - sample['w_cache_hit_rate'])
    w6 = l * (1 - sample['w_cache_hit_rate'])
    w7 = sample['N'] * (1 - sample['w_cache_hit_rate'])
    w8 = sample['mbuf'] * (1 - sample['w_cache_hit_rate'])
    X.append([z0, z11, z12, q1, q2, w0, w1, w2, w3, w4, w5, w6, w7, w8, 1])
    y = (sample['write_io'] + sample['blocks_read']) / Q
    Y.append(y)

print('Start training cache exp')
Xe = np.array(Xe)
Ye = np.array(Ye)
We = np.linalg.lstsq(Xe, Ye, rcond=-1)[0]

Xr = np.array(Xr)
Yr = np.array(Yr)
Wr = np.linalg.lstsq(Xr, Yr, rcond=-1)[0]

Xq = np.array(Xq)
Yq = np.array(Yq)
Wq = np.linalg.lstsq(Xq, Yq, rcond=-1)[0]

Xw = np.array(Xw)
Yw = np.array(Yw)
Ww = np.linalg.lstsq(Xw, Yw, rcond=-1)[0]

X = np.array(X)
Y = np.array(Y)
W = np.linalg.lstsq(X, Y, rcond=-1)[0]

error = []
for _, sample in test_samples.iterrows():
    if sample['blocks_read'] + sample['write_io'] == 0:
        continue
    l = L(sample['N'], sample['mbuf'], sample['T'], False)
    fpr = np.exp(-1 * sample['h'] * (np.log(2) ** 2))
    D = sample['cache_cap'] / 1024
    data = np.zeros([sample['N']])
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

    Nf = sample['z0'] * Q * l * fpr
    xe = [1]
    for power in (alpha, c, Nf, D, 1):
        for x in (alpha, c, Nf, D):
            xe.append(power * np.log(x))
    ye = np.exp(np.dot(xe, We))

    Nf1 = sample['z1'] * Q * (l * fpr)
    Nf2 = sample['z1'] * Q
    xr = [1]
    for power in (alpha, c, Nf1, Nf2, D, 1):
        for x in (alpha, c, Nf1, Nf2, D):
            xr.append(power * np.log(x))
    yr = np.exp(np.dot(xr, Wr))

    Nf1 = sample['q'] * Q * l
    Nf2 = sample['q'] * Q * (S / B)
    xq = [1]
    for power in (alpha, c, Nf1, Nf2, D, 1):
        for x in (alpha, c, Nf1, Nf2, D):
            xq.append(power * np.log(x))
    yq = np.exp(np.dot(xq, Wq))

    Nf1 = sample['w'] * Q * l * sample['T']
    Nf2 = sample['w'] * Q * l
    xw = [1]
    for power in (alpha, c, Nf1, Nf2, D, 1):
        for x in (alpha, c, Nf1, Nf2, D):
            xw.append(power * np.log(x))
    yw = np.exp(np.dot(xw, Ww))

    z0 = sample['z0'] * l * fpr * (1 - ye)
    z11 = sample['z1'] * (1 - yr)
    z12 = sample['z1'] * l * fpr * (1 - yr)
    q1 = sample['q'] * l * (1 - yq)
    q2 = sample['q'] * (S / B) * (1 - yq)
    N = sample['N'] + Q * sample['w']
    l = L(sample['N'], sample['mbuf'], sample['T'], False)
    w0 = sample['w'] * l * sample['T'] / B * (1 - yw)
    w1 = sample['w'] * l * (1 - yw)
    w2 = sample['w'] * sample['mbuf'] * (1 - yw)
    w3 = sample['w'] * (1 - yw)
    w4 = sample['mbuf'] * (1 - yw)
    w5 = l * sample['T'] / B * (1 - yw)
    w6 = l * (1 - yw)
    w7 = sample['N'] * (1 - yw)
    w8 = sample['mbuf'] * (1 - yw)
    y_hat = np.dot([z0, z11, z12, q1, q2, w0, w1, w2, w3, w4, w5, w6, w7, w8, 1], W)
    y = (sample['write_io'] + sample['blocks_read']) / Q
    print(y_hat, y, yr, yq, ye, yw)
    error.append(abs(y_hat - y))
print(np.mean(error))
