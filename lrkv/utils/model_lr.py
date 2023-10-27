import numpy as np
import sys
import random
import xgboost as xgb
import pickle
import pandas as pd
import time
import pickle as pkl

sys.path.append('./lrkv')
from utils.lsm import estimate_level, estimate_fpr

eps = 1e-5


def traverse_for_T(
    Ws, Wcs, z0, z1, q, w, h0=10, ratio0=1.0, N=1e6, n=10, policy='level'
):
    candidates = []
    for T in range(2, 78):
        h = h0
        ratio = ratio0
        costs = []
        for W, Wc in zip(Ws, Wcs):
            xc = get_cache_uniform(T, h0, ratio, z0, z1, q, w)
            yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
            if policy == 'level':
                x = get_level_cost(T, h0, ratio, z0, z1, q, w, yc)
            else:
                x = get_tier_cost(T, h0, ratio, z0, z1, q, w, yc)
            y = np.dot(x, W)
            costs.append(max(y, eps))
        candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
    candidates.sort(key=lambda x: x[-1])
    # candidates = candidates[:10]
    # candidates.sort(key=lambda x: x[-2])
    return candidates[:n]


def traverse_for_h(
    Ws, Wcs, z0, z1, q, w, T0=10, ratio0=1.0, N=1e6, n=10, policy='level'
):
    candidates = []
    for h in range(1, 16):
        T = T0
        ratio = ratio0
        costs = []
        for W, Wc in zip(Ws, Wcs):
            xc = get_cache_uniform(T, h, ratio, z0, z1, q, w)
            yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
            if policy == 'level':
                x = get_level_cost(T, h, ratio, z0, z1, q, w, yc)
            else:
                x = get_tier_cost(T, h, ratio, z0, z1, q, w, yc)
            y = np.dot(x, W)
            y = np.dot(x, W)
            costs.append(max(y, eps))
        candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
    candidates.sort(key=lambda x: x[-1])
    # candidates = candidates[:10]
    # candidates.sort(key=lambda x: x[-2])
    return candidates[:n]


def get_cache(
    current_T,
    current_h,
    current_ratio,
    alpha,
    c,
    z0,
    z1,
    q,
    w,
    M=214748364.8,
    N=1e6,
):
    fpr = estimate_fpr(current_h)
    buffer = current_ratio * (M - current_h * N)
    cache_cap = (1 - current_ratio) * (M - current_h * N) / 8
    l = estimate_level(N, buffer, current_T)
    xc = [1]
    for power in (alpha, c, z0, z1, q, w, 1):
        for x in (l, current_T, fpr, cache_cap, buffer):
            xc.append(power * np.log(x + eps))
    return xc


def get_cache_uniform(
    current_T,
    current_h,
    current_ratio,
    z0,
    z1,
    q,
    w,
    M=214748364.8,
    N=1e6,
):
    buffer = current_ratio * (M - current_h * N)
    h = current_ratio * current_h
    fpr = estimate_fpr(h)
    cache_cap = (1 - current_ratio) * M / 8
    l = estimate_level(N, buffer, current_T)
    xc = [1]
    for power in (z0, z1, q, w, 1):
        for x in (l, current_T, fpr, cache_cap, buffer):
            xc.append(power * np.log(x + eps))
    return xc


def get_level_cost(
    current_T,
    current_h,
    current_ratio,
    z0,
    z1,
    q,
    w,
    y_cache,
    M=214748364.8,
    N=1e6,
):
    buffer = current_ratio * (M - current_h * N)
    h = current_ratio * current_h
    fpr = estimate_fpr(h)
    l = estimate_level(N, buffer, current_T)
    yc = 1 - y_cache
    z00 = z0 * fpr * yc  # endure model
    z01 = z0 * yc  # blocks fall in L0 without bf
    z02 = z0 * buffer * yc  # blocks fall in memory
    z03 = z0 * l  # cpu time
    z10 = z1 * yc  # endure model + blocks fall in L0 without bf
    z11 = z1 * fpr * yc  # endure model
    z12 = z1 * buffer * yc  # blocks fall in memory
    z13 = z1 * l  # cpu time
    q0 = q * l * yc  # endure model
    q1 = q * yc  # endure model
    q2 = q * l  # cpu time
    w0 = w * l  # endure model
    w1 = w * l * current_T  # endure model
    w2 = w * buffer  # endure model
    return [z00, z01, z02, z03, z10, z11, z12, z13, q0, q1, q2, w0, w1, w2]


def get_tier_cost(
    current_T,
    current_h,
    current_ratio,
    z0,
    z1,
    q,
    w,
    y_cache,
    M=214748364.8,
    N=1e6,
):
    h = current_ratio * current_h
    fpr = estimate_fpr(h)
    buffer = current_ratio * (M - current_h * N)
    l = estimate_level(N, buffer, current_T)
    yc = 1 - y_cache  # cache miss rate
    z00 = z0 * fpr * yc  # endure model
    z01 = z0 * fpr * current_T * yc  # endure model
    z02 = z0 * yc  # blocks fall in L0 without bf
    z03 = z0 * buffer * yc  # blocks fall in memory
    z04 = z0 * l * current_T  # cpu time
    z10 = z1 * yc  # endure model + blocks fall in L0 without bf
    z11 = z1 * fpr * yc  # endure model
    z12 = z1 * fpr * current_T * yc  # endure model
    z13 = z1 * buffer * yc  # blocks fall in memory
    z14 = z1 * l * current_T  # cpu time
    q0 = q * l * yc  # endure model
    q1 = q * l * current_T * yc  # endure model
    q2 = q * yc  # blocks fall in memory
    q3 = q * l * current_T  # cpu time
    w0 = w * l  # endure model
    w1 = w * l / current_T
    w2 = w * l * current_T
    w3 = w * buffer
    return [
        z00,
        z01,
        z02,
        z03,
        z04,
        z10,
        z11,
        z12,
        z13,
        z14,
        q0,
        q1,
        q2,
        q3,
        w0,
        w1,
        w2,
        w3,
    ]


def traverse_candidate(Wcs, Ws, z0, z1, q, w, policy='level', number=3):
    candidates = []
    for T in range(2, 17):
        for h in range(8, 13):
            for ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                costs = []
                for Wc, W in zip(Wcs, Ws):
                    xc = get_cache_uniform(
                        T,
                        h,
                        ratio,
                        z0,
                        z1,
                        q,
                        w,
                    )
                    yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
                    if policy == 'level':
                        x = get_level_cost(
                            T,
                            h,
                            ratio,
                            z0,
                            z1,
                            q,
                            w,
                            yc,
                        )
                    else:
                        x = get_tier_cost(
                            T,
                            h,
                            ratio,
                            z0,
                            z1,
                            q,
                            w,
                            yc,
                        )
                    costs.append(max(np.dot(x, W), eps))
                candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
    candidates.sort(key=lambda x: x[-1])
    candidates = candidates[:10]
    candidates.sort(key=lambda x: x[-2])
    return candidates[-1 - number : -1]


def traverse_var_optimizer(
    Wcs,
    Ws,
    alpha,
    c,
    z0,
    z1,
    q,
    w,
    policy='level',
):
    candidates = []
    for T in range(2, 17):
        for h in range(8, 13):
            for ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                costs = []
                for Wc, W in zip(Wcs, Ws):
                    xc = get_cache(
                        T,
                        h,
                        ratio,
                        alpha,
                        c,
                        z0,
                        z1,
                        q,
                        w,
                    )
                    yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
                    if policy == 'level':
                        x = get_level_cost(
                            T,
                            h,
                            ratio,
                            z0,
                            z1,
                            q,
                            w,
                            yc,
                        )
                    else:
                        x = get_tier_cost(
                            T,
                            h,
                            ratio,
                            z0,
                            z1,
                            q,
                            w,
                            yc,
                        )
                    costs.append(max(np.dot(x, W), eps))
                candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
    candidates.sort(key=lambda x: x[-1])
    candidates = candidates[:10]
    candidates.sort(key=lambda x: x[-2])
    return candidates[0]


def traverse_var_optimizer_uniform(
    Wcs, Ws, z0, z1, q, w, policy='level', N=1e6, M=214748364.8
):
    start_time = time.time()
    candidates = []
    for T in range(2, 78):
        for h in range(1, 16):
            for ratio in [0.8, 0.85, 0.9, 0.95, 1.0]:
                costs = []
                for Wc, W in zip(Wcs, Ws):
                    xc = get_cache_uniform(T, h, ratio, z0, z1, q, w, N=N, M=M)
                    yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
                    if policy == 'level':
                        x = get_level_cost(T, h, ratio, z0, z1, q, w, yc, N=N, M=M)
                    else:
                        x = get_tier_cost(T, h, ratio, z0, z1, q, w, yc, N=N, M=M)
                    costs.append(max(np.dot(x, W), eps))
                candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
    candidates.sort(key=lambda x: x[-1])
    candidates = candidates[:10]
    candidates.sort(key=lambda x: x[-2])
    print(time.time() - start_time)
    return candidates[0]


def traverse_var_optimizer_uniform_T(
    Wcs, Ws, z0, z1, q, w, policy='level', N=1e6, M=214748364.8
):
    start_time = time.time()
    candidates = []
    for T in range(2, 78):
        h = 10
        ratio = 1
        costs = []
        for Wc, W in zip(Wcs, Ws):
            xc = get_cache_uniform(T, h, ratio, z0, z1, q, w, N=N, M=M)
            yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
            if policy == 'level':
                x = get_level_cost(T, h, ratio, z0, z1, q, w, yc, N=N, M=M)
            else:
                x = get_tier_cost(T, h, ratio, z0, z1, q, w, yc, N=N, M=M)
            costs.append(max(np.dot(x, W), eps))
        candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
    candidates.sort(key=lambda x: x[-1])
    candidates = candidates[:10]
    candidates.sort(key=lambda x: x[-2])
    print(time.time() - start_time)
    return candidates[0]


def traverse_var_optimizer_uniform_memory(
    Wcs, Ws, z0, z1, q, w, policy='level', N=1e6, M=214748364.8
):
    start_time = time.time()
    candidates = []
    for T in range(2, 78):
        for h in range(1, 16):
            ratio = 1
            costs = []
            for Wc, W in zip(Wcs, Ws):
                xc = get_cache_uniform(T, h, ratio, z0, z1, q, w, N=N, M=M)
                yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
                if policy == 'level':
                    x = get_level_cost(T, h, ratio, z0, z1, q, w, yc, N=N, M=M)
                else:
                    x = get_tier_cost(T, h, ratio, z0, z1, q, w, yc, N=N, M=M)
                costs.append(max(np.dot(x, W), eps))
            candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
    candidates.sort(key=lambda x: x[-1])
    candidates = candidates[:10]
    candidates.sort(key=lambda x: x[-2])
    print(time.time() - start_time)
    return candidates[0]


def traverse_var_optimizer_uniform_uncertainty(
    Wcs, Ws, workloads, policy='level', N=1e6, M=214748364.8
):
    candidates = {}

    for worklaod in workloads:
        z0, z1, q, w = worklaod
        for T in range(2, 17):
            for h in range(8, 13):
                for ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                    costs = []
                    for Wc, W in zip(Wcs, Ws):
                        xc = get_cache_uniform(T, h, ratio, z0, z1, q, w, N=N, M=M)
                        yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
                        if policy == 'level':
                            x = get_level_cost(T, h, ratio, z0, z1, q, w, yc, N=N, M=M)
                        else:
                            x = get_tier_cost(T, h, ratio, z0, z1, q, w, yc, N=N, M=M)
                        costs.append(max(np.dot(x, W), eps))
                    if (T, h, ratio) not in candidates.keys():
                        candidates[(T, h, ratio)] = [np.mean(costs)]
                    else:
                        candidates[(T, h, ratio)].append(np.mean(costs))
    for candidate in candidates.keys():
        candidates[candidate] = np.mean(candidates[candidate])
    candidates = dict(sorted(candidates.items(), key=lambda item: item[1]))
    key, value = next(iter(candidates.items()))
    return key[0], key[1], key[2], value


def traverse_optimizer(
    Wcs,
    Ws,
    alpha,
    c,
    z0,
    z1,
    q,
    w,
    policy='level',
):
    best_cost = 9999999
    for T in range(2, 17):
        for h in range(8, 13):
            for ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
                costs = []
                for Wc, W in zip(Wcs, Ws):
                    xc = get_cache(
                        T,
                        h,
                        ratio,
                        alpha,
                        c,
                        z0,
                        z1,
                        q,
                        w,
                    )
                    yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
                    if policy == 'level':
                        x = get_level_cost(
                            T,
                            h,
                            ratio,
                            z0,
                            z1,
                            q,
                            w,
                            yc,
                        )
                    else:
                        x = get_tier_cost(
                            T,
                            h,
                            ratio,
                            z0,
                            z1,
                            q,
                            w,
                            yc,
                        )
                    costs.append(max(np.dot(x, W), eps))
                if np.mean(costs) < best_cost:
                    best_T, best_h, best_ratio = T, h, ratio
                    best_cost = np.mean(costs)
    return best_T, best_h, best_ratio, best_cost


def get_candidate_simulated_annealing(
    Wcs,
    Ws,
    init_T,
    init_h,
    init_ratio,
    alpha,
    c,
    z0,
    z1,
    q,
    w,
    temperature=100,
    cooling_rate=0.99,
    policy='level',
):
    results = []
    current_T, current_h, current_ratio = [init_T, init_h, init_ratio]
    current_costs = []
    for Wc, W in zip(Wcs, Ws):
        xc = get_cache(
            current_T,
            current_h,
            current_ratio,
            alpha,
            c,
            z0,
            z1,
            q,
            w,
        )
        yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
        if policy == 'level':
            x = get_level_cost(
                current_T,
                current_h,
                current_ratio,
                z0,
                z1,
                q,
                w,
                yc,
            )
        else:
            x = get_tier_cost(
                current_T,
                current_h,
                current_ratio,
                z0,
                z1,
                q,
                w,
                yc,
            )
        y_hat = np.dot(x, W)
        current_costs.append(max(y_hat, eps))
    current_cost = np.mean(current_costs)
    best_T, best_h, best_ratio = current_T, current_h, current_ratio
    best_cost = current_cost
    results.append([best_T, best_h, best_ratio, best_cost])
    while temperature > 1e-8:
        new_T = current_T + random.choice([-1, 0, 1])
        new_h = current_h + random.choice([-1, 0, 1])
        new_ratio = current_ratio + random.choice([-0.1, 0, 0.1])
        if 2 <= new_T <= 16 and 8 <= new_h <= 13 and 0.25 <= new_ratio < 1.0:
            new_costs = []
            for Wc, W in zip(Wcs, Ws):
                xc = get_cache(
                    current_T,
                    current_h,
                    current_ratio,
                    alpha,
                    c,
                    z0,
                    z1,
                    q,
                    w,
                )
                yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
                if policy == 'level':
                    x = get_level_cost(
                        current_T,
                        current_h,
                        current_ratio,
                        z0,
                        z1,
                        q,
                        w,
                        yc,
                    )
                else:
                    x = get_tier_cost(
                        current_T,
                        current_h,
                        current_ratio,
                        z0,
                        z1,
                        q,
                        w,
                        yc,
                    )
                y_hat = np.dot(x, W)
                new_costs.append(max(y_hat, eps))
            new_cost = np.mean(new_costs)
            delta_cost = new_cost - current_cost
            if delta_cost < 0:
                current_T, current_h, current_ratio = new_T, new_h, new_ratio
                current_cost = new_cost
                if current_cost < best_cost:
                    best_T, best_h, best_ratio = current_T, current_h, current_ratio
                    best_cost = current_cost
                    results.append([best_T, best_h, best_ratio, best_cost])
            elif np.exp(-delta_cost / temperature) > random.uniform(0, 1):
                results.append([current_T, current_h, current_ratio, current_cost])
                current_T, current_h, current_ratio = new_T, new_h, new_ratio
                current_cost = new_cost
                results.append([current_T, current_h, current_ratio, current_cost])
            # print(f'T: {current_T}, h: {current_h}, ratio: {current_ratio}, cost: {current_cost}')
            temperature *= cooling_rate
    results.sort(key=lambda x: x[-1])
    new_results = []
    for result in results:
        T, h, ratio, _ = result
        if [T, h, ratio] not in new_results:
            new_results.append([T, h, ratio])
            if len(new_results) > 10:
                break
    return new_results


def simulated_annealing(
    Wcs,
    Ws,
    init_T,
    init_h,
    init_ratio,
    alpha,
    c,
    z0,
    z1,
    q,
    w,
    temperature=100,
    cooling_rate=0.99,
    policy='level',
):
    current_T, current_h, current_ratio = [init_T, init_h, init_ratio]
    current_costs = []
    for Wc, W in zip(Wcs, Ws):
        xc = get_cache(
            current_T,
            current_h,
            current_ratio,
            alpha,
            c,
            z0,
            z1,
            q,
            w,
        )
        yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
        if policy == 'level':
            x = get_level_cost(
                current_T,
                current_h,
                current_ratio,
                z0,
                z1,
                q,
                w,
                yc,
            )
        else:
            x = get_tier_cost(
                current_T,
                current_h,
                current_ratio,
                z0,
                z1,
                q,
                w,
                yc,
            )
        y_hat = np.dot(x, W)
        current_costs.append(max(y_hat, eps))
    current_cost = np.mean(current_costs)
    best_T, best_h, best_ratio = current_T, current_h, current_ratio
    best_cost = current_cost
    while temperature > 1e-8:
        new_T = current_T + random.choice([-1, 0, 1])
        new_h = current_h + random.choice([-1, 0, 1])
        new_ratio = current_ratio + random.choice([-0.1, 0, 0.1])
        if 2 <= new_T <= 16 and 8 <= new_h <= 13 and 0.25 <= new_ratio < 1.0:
            new_costs = []
            for Wc, W in zip(Wcs, Ws):
                xc = get_cache(
                    current_T,
                    current_h,
                    current_ratio,
                    alpha,
                    c,
                    z0,
                    z1,
                    q,
                    w,
                )
                yc = np.clip(np.exp(np.dot(xc, Wc)), 0, 1)
                if policy == 'level':
                    x = get_level_cost(
                        current_T,
                        current_h,
                        current_ratio,
                        z0,
                        z1,
                        q,
                        w,
                        yc,
                    )
                else:
                    x = get_tier_cost(
                        current_T,
                        current_h,
                        current_ratio,
                        z0,
                        z1,
                        q,
                        w,
                        yc,
                    )
                y_hat = np.dot(x, W)
                new_costs.append(max(y_hat, eps))
            new_cost = np.mean(new_costs)
            delta_cost = new_cost - current_cost
            if delta_cost < 0:
                current_T, current_h, current_ratio = new_T, new_h, new_ratio
                current_cost = new_cost
                if current_cost < best_cost:
                    best_T, best_h, best_ratio = current_T, current_h, current_ratio
                    best_cost = current_cost
            elif np.exp(-delta_cost / temperature) > random.uniform(0, 1):
                current_T, current_h, current_ratio = new_T, new_h, new_ratio
                current_cost = new_cost
            # print(f'T: {current_T}, h: {current_h}, ratio: {current_ratio}, cost: {current_cost}')
            temperature *= cooling_rate
    return best_T, best_h, best_ratio, best_cost


if __name__ == '__main__':
    level_cost_models = pkl.load(open(f"model/level_cost_lr_uniform_{1}.pkl", "rb"))
    level_cache_models = pkl.load(open(f"model/level_cache_lr_uniform_{1}.pkl", "rb"))
    tier_cost_models = pkl.load(open(f"model/tier_cost_lr_uniform_{1}.pkl", "rb"))
    tier_cache_models = pkl.load(open(f"model/tier_cache_lr_uniform_{1}.pkl", "rb"))
    (
        best_T,
        best_h,
        best_ratio,
        best_var,
        best_cost,
    ) = traverse_var_optimizer_uniform(
        level_cache_models, level_cost_models, 0.25, 0.25, 0.25, 0.25, N=1e6
    )
