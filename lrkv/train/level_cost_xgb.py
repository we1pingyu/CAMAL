import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import sys
import random

sys.path.append('./lrkv')
from utils.model_xgb import prepare_df
from sklearn.model_selection import train_test_split


np.set_printoptions(suppress=True)

E = 1024
Q = 200000
B = 4
S = 2


# all_samples = pd.read_csv('raw_data/al_level_cost_ckpt.csv')
# all_samples = all_samples.sample(frac=1)
# prepare_df(all_samples, 'prepared_data/al_level_cost.csv')
all_samples = pd.read_csv('prepared_data/al_level_cost.csv')


print('Start training')
eps = 1e-8
n_estimators = 100

regrs = []
cache_regrs = []
for i in range(10):
    # Resample the data
    resampled_df = all_samples.sample(frac=1, replace=True)
    regr = xgb.XGBRegressor(n_estimators=n_estimators)
    # cache_regr = xgb.XGBRegressor(n_estimators=n_estimators)
    # Split the resampled data into training and testing sets
    # X_cache_train, _, y_cache_train, _ = train_test_split(
    #     resampled_df.drop(['total_latency', 'cache_hit_rate'], axis=1),
    #     resampled_df["cache_hit_rate"],
    #     test_size=0.2,
    # )
    # X_cache_train = np.array(X_cache_train)
    # y_cache_train = np.array(y_cache_train)
    # # Train the XGBoost cache model
    # cache_regr.fit(X_cache_train, y_cache_train)

    X_cost_train, X_cost_test, y_cost_train, y_cost_test = train_test_split(
        resampled_df.drop(['total_latency', 'cache_hit_rate'], axis=1),
        resampled_df["total_latency"] / Q,
        test_size=0.2,
    )
    X_cost_train = np.array(X_cost_train)
    y_cost_train = np.array(y_cost_train)

    # Train the XGBoost cache model
    regr.fit(X_cost_train, y_cost_train)
    # Evaluate the model on the test data
    accuracy = regr.score(X_cost_test, y_cost_test)
    print("Iteration {}: accuracy = {:.2f}%".format(i, accuracy * 100))
    # pickle.dump(cache_regr, open(f'model/al_level_cache_xgb_{i}.pkl', "wb"))
    pickle.dump(regr, open(f'model/al_level_cost_xgb_{i}.pkl', "wb"))
    # cache_regrs.append(cache_regr)
    regrs.append(regr)


error = []
rerror = []

print('Start inference')
for _, sample in all_samples.iterrows():
    y_costs = []
    for regr in regrs:
        # X_cache = np.array(
        #     [
        #         sample['alpha'],
        #         sample['c'],
        #         sample['z0'],
        #         sample['z1'],
        #         sample['q'],
        #         sample['w'],
        #         sample['T'],
        #         sample['l'],
        #         sample['fpr'],
        #         sample['cache_cap'],
        #     ]
        # )
        # y_cache = cache_regr.predict([X_cache])[0]
        X_cost = np.array(
            [
                sample['alpha'],
                sample['c'],
                sample['z0'],
                sample['z1'],
                sample['q'],
                sample['w'],
                sample['T'],
                sample['l'],
                sample['fpr'],
                sample['cache_cap'],
                # y_cache,
            ]
        )
        y_cost = regr.predict([X_cost])[0]
        y_costs.append(y_cost)
    var = np.var(y_costs)
    y = sample['total_latency'] / Q
    y_hat = np.mean(y_costs)
    print("=" * 50)
    # print(l, sample['T'])
    print(y_hat, y, var)
    error.append(abs(y_hat - y))
    rerror.append(abs(y_hat - y) / y)
print(np.mean(error), np.mean(rerror))
