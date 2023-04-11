import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

N = 1e7
E = 1024 * 8


def L(N, M, h, T, get_ceiling=True):
    """L(x) function from Eq. 38
    with h = x / N

    :param h:
    :param T:
    """
    mbuff = M - (h * N)
    est_l = np.log(((N * E) / mbuff) + 1) / np.log(T)
    if get_ceiling:
        est_l = np.ceil(est_l)

    return est_l


all_samples = pd.read_csv('/home/ubuntu/code/endure/data/linear_model_checkpoint.csv')
training_samples = all_samples[:-20]
test_samples = all_samples[-20:]
print(training_samples.head())
X = []
Y = []
Q = 100000
B = 4
S = 10
alpha = 4
for _, sample in training_samples.iterrows():
    est_l = L(sample['N'], sample['M'], sample['h'], sample['T'])
    _l = L(sample['N'], sample['M'], sample['h'], sample['T'], False)
    fpr = np.exp(-1 * sample['h'] * (np.log(2) ** 2))
    x0 = sample['w'] * est_l * sample['T'] / B
    x1 = sample['w'] * est_l
    x2 = sample['w'] * sample['mbuf']
    x3 = sample['w']
    x4 = sample['z0'] * est_l * sample['T'] * fpr
    x5 = sample['z1']
    x6 = sample['z1'] * est_l * fpr
    x7 = sample['z1'] * est_l * sample['T'] * fpr
    x8 = sample['q'] * est_l
    x9 = sample['q'] * (S / B) * sample['T']
    X.append([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, 1])
    Y.append((sample['write_io'] + sample['blocks_read']) / Q)
    # Y.append(sample['blocks_read']/Q)
    # Y.append(sample['write_io'] / Q)
    # X.append([x3,x4,1])
    # Y.append(sample['q_io'])
X = np.array(X)
Y = np.array(Y)
print(X.shape, Y.shape)
beta_hat = np.linalg.lstsq(X, Y)[0]
print(beta_hat)
X = []
X_0 = []
Y_hat = []
Y = []
for _, sample in test_samples.iterrows():
    est_l = L(sample['N'], sample['M'], sample['h'], sample['T'])
    _l = L(sample['N'], sample['M'], sample['h'], sample['T'], False)
    fpr = np.exp(-1 * sample['h'] * (np.log(2) ** 2))
    x0 = sample['w'] * est_l * sample['T'] / B
    x1 = sample['w'] * est_l
    x2 = sample['w'] * sample['mbuf']
    x3 = sample['w']
    x4 = sample['z0'] * est_l * sample['T'] * fpr
    x5 = sample['z1']
    x6 = sample['z1'] * est_l * fpr
    x7 = sample['z1'] * est_l * sample['T'] * fpr
    x8 = sample['q'] * est_l
    x9 = sample['q'] * (S / B) * sample['T']
    y_hat = np.dot([x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, 1], beta_hat)
    y = (sample['write_io'] + sample['blocks_read']) / Q
    # y = sample['blocks_read']/Q
    # y = sample['write_io'] / Q
    # X.append([x3,1])
    # X_0.append(x3)
    # y_hat = np.dot([x3,x4,1],beta_hat)
    # Y_hat.append(y_hat)
    # y = sample['q_io']
    # Y.append(y)
    print(y, y_hat, sample['model_io'] / Q)
    print(abs(y_hat - y) < abs(sample['model_io'] / Q - y))

for _, sample in all_samples.iterrows():
    pass
# plt.scatter(X_0,Y)
# plt.plot(X_0,Y_hat,'r',label="Fitted line")
# plt.legend()
# plt.savefig('q_io.png')
