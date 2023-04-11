import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset

np.set_printoptions(suppress=True)

E = 1024
Q = 200000
B = 4
S = 2


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class NNDataset(Dataset):
    def __init__(self,data):
        super(NNDataset, self).__init__()
        self.X,self.Y = data

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


def L(N, mbuf, T, get_ceiling=True):
    # l = np.log(((N * E) / (mbuf + 1)) + 1) / np.log(T)
    l = np.log((N * E / mbuf) * ((T - 1) / T)) / np.log(T)
    if get_ceiling:
        l = np.ceil(l)
    return l


all_samples = pd.read_csv('data/tier_cost_ckpt.csv')
all_samples = all_samples.sample(frac=1)
training_samples = all_samples[:360]
test_samples = all_samples[360:]


def process_df(samples):
    X = []
    Y = []
    for _, sample in samples.iterrows():
        if sample['read_io'] + sample['write_io'] == 0:
            continue
        l = L(sample['N'], sample['mbuf'], sample['T'], False)
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

        X.append([
                alpha,
                c,
                sample['z0'],
                sample['z1'],
                sample['q'],
                sample['w'],
                sample['T'],
                l,
                fpr,
                sample['cache_cap'],
                sample['mbuf']
            ])
        
        Y.append(sample['total_latency'] / Q)
    X=np.array(X)
    Y=np.array(Y)
    return X, Y


model = Model(11, 32, 1)
train_dataset = NNDataset(process_df(training_samples))
train_loader = DataLoader(train_dataset, batch_size=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
mse_loss = nn.SmoothL1Loss()

valid_time = 0
for epoch in range(16):
    train_loss = []
    model.train()
    for _, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.float()
        labels = labels.float()

        optimizer.zero_grad()
        preds = model(inputs).reshape(-1)

        loss = mse_loss(preds, labels)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())
    print(f'epoch:{epoch}, loss:{np.mean(train_loss)}')
    
test_dataset = NNDataset(process_df(test_samples))
test_loader = DataLoader(test_dataset)
error = []
rerror = []
for _, data in enumerate(test_loader):
    inputs, labels = data
    inputs = inputs.float()
    labels = labels.float()

    with torch.no_grad():
        preds = model(inputs).item()
        print(preds,labels.item())
        error.append(abs(preds - labels.item()))
        rerror.append(abs(preds - labels.item()) / labels.item())
print(np.mean(error), np.mean(rerror))

        