import numpy as np
import sys
import random
import xgboost as xgb
import pickle
import pandas as pd
import time
import yaml
import os

sys.path.append("./lrkv")
from utils.lsm import estimate_level, estimate_fpr

eps = 1e-5


def iter_model(df, policy, E, M, N):
    X = []
    Y = []
    for sample in df:
        X.append(
            get_cost_uniform(
                sample["T"],
                sample["h"],
                sample["ratio"],
                sample["z0"],
                sample["z1"],
                sample["q"],
                sample["w"],
                E,
                M,
                N,
            )
        )
        Y.append(sample["total_latency"] / sample["queries"])
    _X = np.array(X)
    _Y = np.array(Y)
    regr = xgb.XGBRegressor()
    regr.fit(_X, _Y)
    return regr


def prepare_df(samples, save_path):
    df = []
    for _, sample in samples.iterrows():
        row = {}
        if sample["read_io"] + sample["write_io"] == 0:
            continue
        l = estimate_level(sample["N"], sample["mbuf"], sample["T"], get_ceiling=False)
        fpr = np.exp(-1 * sample["h"] * (np.log(2) ** 2))
        data = np.zeros([int(sample["N"])])
        with open(sample["key_log"], "r") as f:
            for line in f.readlines():
                last = ord(line.strip("\n")[-1])
                if last >= ord("A"):
                    data[int(line.strip("\n")[:-1] + str(last - 65))] += 1
                else:
                    data[int(line.strip("\n"))] += 1
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
        row["alpha"] = alpha
        row["c"] = c
        row["z0"] = sample["z0"]
        row["z1"] = sample["z1"]
        row["q"] = sample["q"]
        row["w"] = sample["w"]
        row["T"] = sample["T"]
        row["l"] = l
        row["fpr"] = fpr
        row["cache_cap"] = sample["cache_cap"]
        row["mbuf"] = sample["mbuf"]
        row["cache_hit_rate"] = sample["cache_hit_rate"]
        row["total_latency"] = sample["total_latency"]
        df.append(row)
    pd.DataFrame(df).to_csv(save_path, index=False)


def load_models(model_path, folds):
    models = []
    for fold in range(folds):
        model = pickle.load(open(model_path.replace("holder", str(fold)), "rb"))
        models.append(model)
    return models


def get_cache(current_T, current_h, current_ratio, alpha, c, z0, z1, q, w, M, N):
    fpr = estimate_fpr(current_h)
    buffer = current_ratio * (M - current_h * N)
    cache_cap = (1 - current_ratio) * (M - current_h * N) / 8
    l = estimate_level(N, buffer, current_T)
    return [alpha, c, z0, z1, q, w, current_T, l, fpr, cache_cap, buffer]


def get_cost_uniform(
    # is_leveling_policy,
    current_T,
    current_h,
    current_ratio,
    z0,
    z1,
    q,
    w,
    E,
    M,
    N,
):
    h = current_ratio * current_h
    fpr = estimate_fpr(h)
    # print(M, h, N)
    buffer = current_ratio * (M - current_h * N) / 8
    cache_cap = (1 - current_ratio) * M / 8
    l = estimate_level(N, buffer, current_T, E)
    # print(N, buffer, current_T, E, l)
    # is_leveling_policy = 1 if is_leveling_policy else 0
    return [z0, z1, q, w, current_T, l, fpr, cache_cap, buffer]


def get_cost(
    current_T,
    current_h,
    current_ratio,
    alpha,
    c,
    z0,
    z1,
    q,
    w,
    y_cache,
    M,
    N,
):
    fpr = estimate_fpr(current_h)
    buffer = current_ratio * (M - current_h * N)
    cache_cap = (1 - current_ratio) * M / 8
    l = estimate_level(N, buffer, current_T)
    return [alpha, c, z0, z1, q, w, current_T, l, fpr, cache_cap, buffer, y_cache]


def traverse_var_optimizer_uniform(cost_models, policy, z0, z1, q, w, E, M, N):
    start_time = time.time()
    costs = []
    xs = []
    settings = []
    for T in range(2, 100):
        for h in range(2, 11):
            for ratio in [0.9, 1.0]:
                x = get_cost_uniform(T, h, ratio, z0, z1, q, w, E, M, N)
                settings.append((T, h, ratio, None))
                xs.append(x)
    for cost_model in cost_models:
        X = np.array(xs)
        cost = cost_model.predict(X)
        costs.append(cost)
    costs = np.array(costs)
    vars = np.var(costs, axis=0)
    costs = np.mean(costs, axis=0)
    candidates = sorted(zip(costs, vars, settings), key=lambda x: x[0])
    candidate = candidates[0]
    print(time.time() - start_time)
    return (
        candidate[-1][0],
        candidate[-1][1],
        candidate[-1][2],
        candidate[1],
        candidate[0],
    )


def traverse_var_optimizer_uniform_T(cost_models, policy, z0, z1, q, w, M, N):
    start_time = time.time()
    costs = []
    xs = []
    settings = []
    for T in range(2, 78):
        h = 10
        ratio = 1
        x = get_cost_uniform(T, h, ratio, z0, z1, q, w, E, M, N)
        settings.append((T, h, ratio, None))
        xs.append(x)
    for cost_model in cost_models:
        X = np.array(xs)
        cost = cost_model.predict(X)
        costs.append(cost)
    costs = np.array(costs)
    costs = np.mean(costs, axis=0)
    candidates = sorted(zip(costs, settings), key=lambda x: x[0])
    candidate = candidates[0]
    print(time.time() - start_time)
    return candidate[1][0], candidate[1][1], candidate[1][2], None, candidate[0]


def traverse_var_optimizer_uniform_memory(cost_models, policy, z0, z1, q, w, E, M, N):
    start_time = time.time()
    costs = []
    xs = []
    settings = []
    for T in range(2, 78):
        for h in range(2, 15):
            ratio = 1
            x = get_cost_uniform(T, h, ratio, z0, z1, q, w, E, M, N)
            settings.append((T, h, ratio, None))
            xs.append(x)
    for cost_model in cost_models:
        X = np.array(xs)
        cost = cost_model.predict(X)
        costs.append(cost)
    costs = np.array(costs)
    costs = np.mean(costs, axis=0)
    candidates = sorted(zip(costs, settings), key=lambda x: x[0])
    candidate = candidates[0]
    print(time.time() - start_time)
    return candidate[1][0], candidate[1][1], candidate[1][2], None, candidate[0]


def traverse_for_T(cost_models, z0, z1, q, w, E, M, N, h0=10, ratio0=1.0, n=10):
    candidates = []
    for T in range(2, 100):
        h = h0
        ratio = ratio0
        costs = []
        for cost_model in cost_models:
            x = get_cost_uniform(T, h, ratio, z0, z1, q, w, E, M, N)
            costs.append(
                max(cost_model.predict(np.array([x]).reshape((1, -1)))[0], eps)
            )
        candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
    candidates.sort(key=lambda x: x[-1])
    # candidates = candidates[:10]
    # candidates.sort(key=lambda x: x[-2])
    return candidates[:n]


def traverse_for_h(cost_models, z0, z1, q, w, E, M, N, T0=10, ratio0=1.0, n=10):
    candidates = []
    for h in range(2, 11):
        T = T0
        ratio = ratio0
        costs = []
        for cost_model in cost_models:
            x = get_cost_uniform(T, h, ratio, z0, z1, q, w, E, M, N)
            costs.append(
                max(cost_model.predict(np.array([x]).reshape((1, -1)))[0], eps)
            )
        candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
    candidates.sort(key=lambda x: x[-1])
    # candidates = candidates[:10]
    # candidates.sort(key=lambda x: x[-2])
    return candidates[:n]


# def simulated_annealing(
#     cache_model: xgb.XGBRegressor,
#     cost_model: xgb.XGBRegressor,
#     init_T,
#     init_h,
#     init_ratio,
#     alpha,
#     c,
#     z0,
#     z1,
#     q,
#     w,
#     temperature=100,
#     cooling_rate=0.99,
# ):
#     current_T, current_h, current_ratio = [init_T, init_h, init_ratio]
#     x_cache = get_cache(current_T, current_h, current_ratio, alpha, c, z0, z1, q, w)
#     y_cache = cache_model.predict([x_cache])[0]
#     x_cost = get_cost(
#         current_T, current_h, current_ratio, alpha, c, z0, z1, q, w, y_cache
#     )
#     current_cost = cost_model.predict([x_cost])[0]
#     best_T, best_h, best_ratio = current_T, current_h, current_ratio
#     best_cost = current_cost
#     while temperature > 1e-8:
#         new_T = current_T + random.choice([-1, 0, 1])
#         new_h = current_h + random.choice([-1, 0, 1])
#         new_ratio = current_ratio + random.choice([-0.1, 0, 0.1])
#         if 2 <= new_T <= 16 and 8 <= new_h <= 13 and 0.25 <= new_ratio < 1.0:
#             x_cache = get_cache(new_T, new_h, new_ratio, alpha, c, z0, z1, q, w)
#             y_cache = cache_model.predict([x_cache])[0]
#             x_cost = get_cost(new_T, new_h, new_ratio, alpha, c, z0, z1, q, w, y_cache)
#             new_cost = cost_model.predict([x_cost])[0]
#             delta_cost = new_cost - current_cost
#             if delta_cost < 0:
#                 current_T, current_h, current_ratio = new_T, new_h, new_ratio
#                 current_cost = new_cost
#                 if current_cost < best_cost:
#                     best_T, best_h, best_ratio = current_T, current_h, current_ratio
#                     best_cost = current_cost
#             elif np.exp(-delta_cost / temperature) > random.uniform(0, 1):
#                 current_T, current_h, current_ratio = new_T, new_h, new_ratio
#                 current_cost = new_cost
#             # print(f'T: {current_T}, h: {current_h}, ratio: {current_ratio}, cost: {current_cost}')
#             temperature *= cooling_rate
#     return best_T, best_h, best_ratio, best_cost


# def get_candidate_simulated_annealing(
#     # cache_model: xgb.XGBRegressor,
#     cost_models,
#     init_T,
#     init_h,
#     init_ratio,
#     alpha,
#     c,
#     z0,
#     z1,
#     q,
#     w,
#     temperature=100,
#     cooling_rate=0.99,
# ):
#     results = []
#     current_T, current_h, current_ratio = [init_T, init_h, init_ratio]
#     x_cache = get_cache(current_T, current_h, current_ratio, alpha, c, z0, z1, q, w)
#     # y_cache = cache_model.predict([x_cache])[0]
#     # x_cost = get_cost(
#     #     current_T, current_h, current_ratio, alpha, c, z0, z1, q, w, y_cache
#     # )
#     current_costs = []
#     for cost_model in cost_models:
#         current_costs.append(
#             max(cost_model.predict(np.array([x_cache]).reshape((1, -1)))[0], eps)
#         )
#     current_cost = np.mean(current_costs)
#     best_T, best_h, best_ratio = current_T, current_h, current_ratio
#     best_cost = current_cost
#     results.append([best_T, best_h, best_ratio, best_cost])
#     while temperature > 1e-5:
#         new_T = current_T + random.choice([-1, 0, 1])
#         new_h = current_h + random.choice([-1, 0, 1])
#         new_ratio = current_ratio + random.choice([-0.1, 0, 0.1])
#         if 2 <= new_T <= 16 and 8 <= new_h <= 13 and 0.25 <= new_ratio < 1.0:
#             x_cache = get_cache(new_T, new_h, new_ratio, alpha, c, z0, z1, q, w)
#             # y_cache = cache_model.predict([x_cache])[0]
#             # x_cost = get_cost(new_T, new_h, new_ratio, alpha, c, z0, z1, q, w, y_cache)
#             new_costs = []
#             for cost_model in cost_models:
#                 new_costs.append(
#                     max(
#                         cost_model.predict(np.array([x_cache]).reshape((1, -1)))[0], eps
#                     )
#                 )
#             new_cost = np.mean(new_costs)
#             delta_cost = new_cost - current_cost
#             if delta_cost < 0:
#                 current_T, current_h, current_ratio = new_T, new_h, new_ratio
#                 current_cost = new_cost
#                 if current_cost < best_cost:
#                     best_T, best_h, best_ratio = current_T, current_h, current_ratio
#                     best_cost = current_cost
#                     results.append([best_T, best_h, best_ratio, best_cost])
#             elif np.exp(-delta_cost / temperature) > random.uniform(0, 1):
#                 results.append([current_T, current_h, current_ratio, current_cost])
#                 current_T, current_h, current_ratio = new_T, new_h, new_ratio
#                 current_cost = new_cost
#                 results.append([current_T, current_h, current_ratio, current_cost])
#             # print(f'T: {current_T}, h: {current_h}, ratio: {current_ratio}, cost: {current_cost}')
#             temperature *= cooling_rate
#     results.sort(key=lambda x: x[-1])
#     new_results = []
#     for result in results:
#         T, h, ratio, _ = result
#         if [T, h, ratio] not in new_results:
#             new_results.append([T, h, ratio])
#             if len(new_results) > 10:
#                 break
#     return new_results
