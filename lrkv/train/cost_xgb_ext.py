import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import sys
import random
import time
import os
import yaml
from datetime import datetime

sys.path.append("./lrkv")
from sklearn.model_selection import train_test_split
from utils.model_xgb_ext import get_cost_uniform
from utils.lsm import *
from sklearn.model_selection import KFold

np.set_printoptions(suppress=True)


config_yaml_path = os.path.join("lrkv/config/config.yaml")
with open(config_yaml_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
level_data = config["samples_path"]["xgb_ext_final"]
fold = 15

for num_sample in [100]:
    start_time = time.time()
    print("Start level training")
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
        if sample["read_io"] + sample["write_io"] == 0:
            continue
        X.append(
            get_cost_uniform(
                sample["T"],
                sample["h"],
                sample["ratio"],
                sample["z0"],
                sample["z1"],
                sample["q"],
                sample["w"],
                sample["E"] / 8,
                sample["M"] / sample["ratio"],
                sample["N"],
                sample["K"],
                sample["fs"],
            )
        )
        y = sample["total_latency"] / sample["queries"]
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
        regr = xgb.XGBRegressor(learning_rate=0.5, n_estimators=10)
        # Train the XGBoost cache model
        regr.fit(X_train, Y_train)
        X_test = X[test_index]
        Y_test = Y[test_index]
        # print(X_train.shape, X_test.shape)
        y_hat = regr.predict(X_test)
        error = abs(y_hat - Y_test)
        rerror = abs(y_hat - Y_test) / Y_test
        for _y_hat, _y, _error, _rerror in zip(y_hat, Y_test, error, rerror):
            errors.append(_error)
            rerrors.append(_rerror)
        regrs.append(regr)
    print(np.mean(errors), np.mean(rerrors))
    pickle.dump(regrs, open(config["xgb_model"]["ext_xgb_cost_model"], "wb"))
    print(time.time() - start_time)
