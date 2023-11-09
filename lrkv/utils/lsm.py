import numpy as np
import pandas as pd
import pickle as pkl
import math
import random

from collections import deque
from scipy import optimize

# from sympy import *

# from lsm_tree.cost_function import NominalCostFunction
B = 4


def weight_sampling(l, o, n, r):
    a = []
    b = []
    for i in l:
        a.append(i[o])
        b.append(1.0 / i[-1])
    i = 0
    while len(r) < n:
        if i > 1e5:
            break
        i += 1
        s = random.choices(a, weights=b)[0]
        if s not in r:
            r.append(s)
    return r


def estimate_level(N, mbuf, T, E, get_ceiling=True):
    l = np.log(((N * E / 8) / (mbuf + 1)) + 1) / np.log(T)
    # print(N, E, mbuf, T)
    # l = np.abs(np.log((N * E / mbuf) * ((T - 1) / T)) / np.log(T))
    if get_ceiling:
        l = np.ceil(l)
    return l


def estimate_T(N, mbuf, L, E, get_ceiling=True):
    # l = np.log(((N * E) / (mbuf + 1)) + 1) / np.log(T)
    # print(N, E, mbuf, T)
    return int(np.exp(np.log(((N * E) / (mbuf + 1)) + 1)) / L)


def estimate_fpr(h):
    return np.exp(-1 * h * (np.log(2) ** 2))


# def f_level_T(q, w, T, h0=10, E=125, N=1e7 / 5, M=2147483648 / 5):
#     buffer_range = (M - 1e6 * h0) / 8 * 0.5
#     return (q / np.log(T) + w / B * T / np.log(T)) * np.log(N * E) * buffer_range


# def delat_level_T(q, w, T, s, B=4):
#     # print(s / ((w * T * np.log(T + 1) - q * B) / (T * np.log(T) * np.log(T))))
#     return abs(s / ((2 * w * T * np.log(T + 1) - q * B) / (T * np.log(T) * np.log(T))))


# def find_level_T(q, w, n, h0=10, N=1e6, M=2147483648 / 5):
#     buffer0 = (M - 1e6 * h0) / 8
#     Tlim = int(N * 125 / buffer0) + 1
#     splits = np.linspace(f_level_T(q, w, Tlim), f_level_T(q, w, 2), n)
#     # print(f'level T:{abs(f_level_T( q, w, Tlim)-f_level_T( q, w, 2))}')
#     results = [2, Tlim]
#     temp = -1
#     # print(splits)
#     for split in splits[1:-1]:
#         min_error = 1e9
#         for T in range(3, Tlim):
#             if abs(f_level_T(q, w, T) - split) < min_error:
#                 min_error = abs(f_level_T(q, w, T) - split)
#                 temp = T
#         if temp not in results:
#             results.append(temp)
#         else:
#             results.append(temp + 1)
#     # results = results.sort()
#     return set(results)


# def f_tier_T(z0, z1, q, w, T, h0=10, N=1e7 / 5, M=2147483648 / 5):
#     buffer_range = (M - 1e6 * h0) / 8 * 0.5
#     fpr = estimate_fpr(h0)
#     return z0 * fpr * T + z1 * (fpr * T + 1) + q * l * T + w / B * l


# def find_tier_T(z0, z1, q, w, n, h0=10, N=1e6):
#     buffer0 = (M - 1e6 * h0) / 8
#     Tlim = int(N * 125 / buffer0) + 1
#     splits = np.linspace(f_tier_T(z0, z1, q, w, Tlim), f_tier_T(z0, z1, q, w, 2), n)
#     # print(f'tier T:{abs(f_tier_T(z0, z1, q, w, Tlim) - f_tier_T(z0, z1, q, w, 2))}')
#     # print(splits)
#     results = [2, Tlim]
#     temp = -1
#     # print(splits)
#     for split in splits[1:-1]:
#         min_error = 1e9
#         for T in range(3, Tlim):
#             if abs(f_tier_T(z0, z1, q, w, T) - split) < min_error:
#                 min_error = abs(f_tier_T(z0, z1, q, w, T) - split)
#                 temp = T
#         if temp not in results:
#             results.append(temp)
#         else:
#             results.append(temp + 1)
#     # results = results.sort()
#     return set(results)


# def f_level_h(z0, z1, h):
#     return np.log(2) * np.log(2) * (z0 + z1) * np.exp(-h)


# def find_level_h(z0, z1, n, N=1e6):
#     splits = np.linspace(f_level_h(z0, z1, 1), f_level_h(z0, z1, 16), n)
#     # print(f'level h:{abs(f_level_h(z0, z1, 1)- f_level_h(z0, z1, 16))}')
#     results = [1, 16]
#     temp = -1
#     for split in splits[1:-1]:
#         min_error = 1e9
#         for h in range(2, 16):
#             if abs(f_level_h(z0, z1, h) - split) < min_error:
#                 min_error = abs(f_level_h(z0, z1, h) - split)
#                 temp = h
#         # if temp not in results:
#         results.append(temp)
#         # else:
#         # results.append(temp + 1)
#     # results = results.sort()
#     return set(results)


# def find_level_mbuf(q, w, n, h0=10, T0=10, N=1e6):
#     buffer0 = (M - N * h0) / 8
#     buffer1 = (M - N * h0) / 8 * 0.5
#     splits = np.linspace(
#         f_level_mbuf(q, w, buffer0, T0), f_level_mbuf(q, w, buffer1, T0), n
#     )
#     results = [0.6, 1.0]
#     temp = -1
#     for split in splits[1:-1]:
#         min_error = 1e9
#         for ratio in [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
#             buffer = (M - N * h0) / 8 * ratio
#             if abs(f_level_mbuf(q, w, buffer) - split) < min_error:
#                 min_error = abs(f_level_mbuf(q, w, buffer) - split)
#                 temp = ratio
#         # if temp not in results:
#         results.append(temp)
#         # else:
#         # results.append(temp + 0.1)
#     # results = results.sort()
#     return set(results)


# def find_tier_mbuf(q, w, n, h0=10, N=1e6):
#     buffer0 = (M - N * h0) / 8
#     buffer1 = (M - N * h0) / 8 * 0.5
#     splits = np.linspace(f_tier_mbuf(q, w, buffer0), f_tier_mbuf(q, w, buffer1), n)
#     # print(f'tier mbuf:{abs(f_tier_mbuf(q, w, buffer0)-f_tier_mbuf(q, w, buffer1))}')
#     results = [0.5, 1.0]
#     temp = -1
#     for split in splits[1:-1]:
#         min_error = 1e9
#         for ratio in np.arange(0.5, 1.0, 0.1):
#             buffer = (M - N * h0) / 8 * ratio
#             if abs(f_tier_mbuf(q, w, buffer) - split) < min_error:
#                 min_error = abs(f_tier_mbuf(q, w, buffer) - split)
#                 temp = ratio
#         if temp not in results:
#             results.append(temp)
#         else:
#             results.append(temp + 0.05)
#     # results = results.sort()
#     return set(results)


# def level_gradient(z0, z1, q, w, T, h, ratio, N=1e6, E=125, M=214748364.8):
#     mbuf = ratio * (M - N * h) / 8
#     l = estimate_level(N, mbuf, T)
#     fpr = estimate_fpr(h)
#     delta_T = w * l / B - (q + w * T / 4) * np.log(N * E / mbuf) / T / np.log(
#         T
#     ) / np.log(T)
#     delta_h = -(z0 + z1) * fpr
#     delta_ratio = -(q + w * T / 4) / np.log(T) / ratio
#     return (abs(delta_T), abs(delta_h), abs(delta_ratio))


# def active_learning_selection(workload, n=10, M=214748364.8):
#     pass


# def level_gradient_sampling(workload, n=10):
#     candidates = []
#     lengths = []
#     for T in range(2, 42):
#         for h in range(1, 17):
#             for ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#                 v = level_gradient(
#                     workload[0],
#                     workload[0],
#                     workload[0],
#                     workload[0],
#                     T,
#                     h,
#                     ratio,
#                 )
#                 candidates.append((T, h, ratio))
#                 length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
#                 lengths.append(length)
#     lengths = np.array(lengths)
#     prob = lengths / np.sum(lengths)
#     samples = random.choices(candidates, weights=prob, k=n)
#     return samples


# def tier_gradient_sampling(workload, n=10):
#     candidates = []
#     lengths = []
#     for T in range(2, 42):
#         for h in range(1, 17):
#             for ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#                 v = tier_gradient(
#                     workload[0],
#                     workload[0],
#                     workload[0],
#                     workload[0],
#                     T,
#                     h,
#                     ratio,
#                 )
#                 candidates.append((T, h, ratio))
#                 length = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
#                 lengths.append(length)
#     lengths = np.array(lengths)
#     prob = lengths / np.sum(lengths)
#     samples = random.choices(candidates, weights=prob, k=n)
#     return samples


# def cost_function():
#     pass


# def level_grid_sampling(workload, n=16, N=1e6, B=4, E=125):
#     z0, z1, q, w = workload
#     candadites = []
#     # T_list = [2]
#     min_diff = 1e9
#     for s in np.arange(0.01, 1.0, 0.01):
#         temp = [2]
#         T = 2
#         while True:
#             T += max(1, round(delat_level_T(q, w, T, s)))
#             if T > estimate_T(N, M / 2 / 8, 1):
#                 break
#             # print(T)
#             temp.append(T)
#         temp.append(estimate_T(N, M / 2 / 8, 1))
#         # print(temp)
#         if 0 <= len(set(temp)) - n < min_diff:
#             min_diff = abs(len(set(temp)) - n)
#             T_list = set(temp)
#     if len(T_list) > n:
#         T_list = random.sample(T_list, n)
#     h_list = []
#     while len(h_list) <= 4:
#         x = int(round(np.random.normal(12, scale=n / 2)))
#         if x >= 1 and x <= 16 and x not in h_list:
#             h_list.append(x)
#     for T in T_list:
#         h = random.choice(h_list)
#         ratio = random.uniform(0.5, 1.0)
#         candadites.append([T, h, ratio])
#     return random.sample(candadites, n)


def T_level_equation(x, q, w):
    return abs(2 * w * x * (np.log(x) - 1) - q * B)


# def filt_memory_level_equation(h, oh, l, M=214748364.8, N=1e6):
#     print((l * M - h * N) / np.exp(h), (M - oh * N) / np.exp(oh))
#     return abs((l * M - h * N) / np.exp(h) - (M - oh * N) / np.exp(oh))


def h_mbuf_level_equation(x, z0, z1, q, w, T, E, M, N):
    return abs(
        ((z0 + z1) * np.exp(-x / N)) / N
        - ((q + 2 * w * (T + 1) / B) / np.log(T) / (M - x))
    )


# def h_T_tier_equation(w, mfilt, T, N=1e6, B=4, E=125):
#     a = abs(
#         (1 - w) * np.exp(-mfilt / N * np.log(2) * np.log(2))
#         - (
#             w
#             * np.abs(np.log((N * E * 8 / (M - mfilt)) + 1) / np.log(T))
#             / (B * T * np.log(T))
#         )
#     )

#     b = abs(
#         w
#         / (B * np.log(T) * (M - mfilt))
#         / (
#             T
#             * np.log(2)
#             * np.log(2)
#             / N
#             * (1 - w)
#             * np.exp(-mfilt / N * np.log(2) * np.log(2))
#         )
#     )
#     # print(T, mfilt / N, a + b)
#     return a, b


def h_mbuf_tier_equation(x, z0, z1, w, q, T, E, M, N):
    return abs(
        ((z0 + z1) * T / N * np.exp(-x / N))
        - (((q + w) * T + w / B) / np.log(T) / (M - x))
    )


def T_tier_equation(x, z0, z1, q, w, sel, E, M, N, h=10):
    # p0 = estimate_fpr(1)
    # p1 = estimate_fpr(10)
    # m0 = M / 8 / 2
    # m1 = M / 8
    # p = estimate_fpr(16)
    mbuf = (M - N * h) / 8
    l = estimate_level(
        N,
        mbuf,
        x,
        E,
    )
    return abs(
        (z0 + z1) * estimate_fpr(h)
        + q * sel
        + l * (q * B * x * (np.log(x) - 1) - w) / (B * x * np.log(x))
    )


# def level_min_cost_sampling(workload, n=10, N=1e6, B=4, E=125):
#     z0, z1, q, w = workload
#     candadites = []
#     # print(C)
#     min_error = 1e9
#     temp = -1
#     for T in range(2, estimate_T(N, M / 2 / 8, 1) + 1):
#         if T_level_equation(T, q, w) < min_error:
#             min_error = T_level_equation(T, q, w)
#             temp = T
#     # print(temp)
#     # print(estimate_T(N, M / 2 / 8, 1) + 1)
#     T_list = [2, estimate_T(N, M / 2 / 8, 1) + 1]
#     while len(T_list) < n:
#         x = int(round(np.random.normal(temp, scale=n / 2)))
#         # print(x)
#         if x >= 2 and x < (estimate_T(N, M / 2 / 8, 1) + 1) and x not in T_list:
#             T_list.append(x)
#     print(temp)

#     for T in T_list:
#         temp = -1
#         min_error = 1e9
#         for ratio in np.arange(0.01, 1, 0.01):
#             m = ratio * M
#             c = h_mbuf_level_equation(m, z0, z1, w, q, T)
#             if c < min_error:
#                 min_error = c
#                 temp = int(ratio * M / N)
#         temp = min(16, max(1, temp))
#         # print(temp)
#         h_list = []
#         while len(h_list) <= 4:
#             x = int(round(np.random.normal(temp, scale=4)))
#             if x >= 1 and x <= 16 and x not in h_list:
#                 h_list.append(x)
#         # print(T, h_list)
#         for h in h_list:
#             ratio = random.uniform(0.5, 1.0)
#             candadites.append([T, h, ratio])
#     return random.sample(candadites, n)


# def tier_min_cost_sampling(workload, n=10, N=1e6, B=4, E=125):
#     z0, z1, q, w = workload
#     candadites = []
#     C = q * B / w
#     # print(C)
#     min_error = 1e9
#     temp = 2
#     h = 0
#     for T in range(2, estimate_T(N, M / 2 / 8, 1) + 1):
#         c = T_tier_equation(
#             T,
#             z0,
#             z1,
#             q,
#             w,
#             h,
#         )
#         if c < min_error:
#             min_error = c
#             temp = T
#     print(temp)
#     T_list = [2, estimate_T(N, M / 2 / 8, 1) + 1]
#     while len(T_list) < n:
#         x = int(round(np.random.normal(temp, scale=n / 2)))
#         if x >= 2 and x < (estimate_T(N, M / 2 / 8, 1) + 1) and x not in T_list:
#             T_list.append(x)
#     for T in T_list:
#         min_error = 1e9
#         for ratio in np.arange(0.01, 1, 0.01):
#             m = ratio * M
#             if h_mbuf_tier_equation(m, z0, z1, w, q, T) < min_error:
#                 min_error = h_mbuf_tier_equation(m, z0, z1, w, q, T)
#                 temp = int(ratio * M / N)
#         temp = min(16, max(1, temp))
#         # print(temp)

#         h_list = []
#         while len(h_list) <= 4:
#             x = int(round(np.random.normal(temp, scale=4)))
#             if x >= 1 and x <= 16 and x not in h_list:
#                 h_list.append(x)
#         # print(T, h_list)
#         for h in h_list:
#             ratio = random.uniform(0.5, 1.0)
#             candadites.append([T, h, ratio])
#     return random.sample(candadites, n)


# def grid_sampling(n=10):
#     candadites = []
#     T_list = []
#     for L in range(1, 32):
#         # print(estimate_T(N=1e6, mbuf=M / 2 / 8, L=L))
#         T_list.append(estimate_T(N=1e6, mbuf=M / 2 / 8, L=L))
#     T_list = set(T_list)
#     print(T_list)
#     h_list = []
#     while len(h_list) <= 4:
#         x = int(round(np.random.normal(12, scale=n / 2)))
#         if x >= 1 and x <= 16 and x not in T_list:
#             h_list.append(x)
#     print(T_list)
#     for T in T_list:
#         h = random.choice(h_list)
#         ratio = random.uniform(0.5, 1.0)
#         candadites.append([T, h, ratio])
#     return random.sample(candadites, n)


# def traverse_var_optimizer_uniform(cost_models, z0, z1, q, w, N=1e6):
#     candidates = []
#     for T in range(2, 42):
#         for h in range(1, 16):
#             for ratio in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
#                 costs = []
#                 for cost_model in cost_models:
#                     x = get_cost_uniform(T, h, ratio, z0, z1, q, w, N=N)
#                     costs.append(cost_model.predict([x])[0])
#                 candidates.append([T, h, ratio, np.var(costs), np.mean(costs)])
#     candidates.sort(key=lambda x: x[-1])
#     candidates = candidates[:10]
#     candidates.sort(key=lambda x: x[-2])
#     return candidates[0]


if __name__ == "__main__":
    pass
