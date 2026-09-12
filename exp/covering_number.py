# -*- coding: utf-8 -*-
"""
独立求解：用 n 个等半径圆盘覆盖单位圆盘所需的最小半径 r_n（即单位圆盘上的 k-中心问题）。
用于判定"6 个点是否足以覆盖 R=1800、Rmin=1000 的区域"，即 r_6 是否 > 1000/1800 = 0.555556。

方法：极小极大优化 max_{g in disk} min_i |g - x_i|，多起点 + 自适应步长坐标下降 + 随机扰动。
"""
from __future__ import annotations

import math
import sys

import numpy as np

RNG = np.random.default_rng(20260913)


def sample_disk(n=60000):
    r = np.sqrt(RNG.random(n))
    a = RNG.random(n) * 2 * math.pi
    return np.stack([r * np.cos(a), r * np.sin(a)], 1)


def covering_radius(X, G):
    d = np.hypot(G[:, None, 0] - X[None, :, 0], G[:, None, 1] - X[None, :, 1])
    return float(d.min(axis=1).max())


def refine(X, G, iters=2500):
    """坐标下降 + 随机扰动，逐步缩小步长"""
    best = covering_radius(X, G)
    step = 0.08
    for it in range(iters):
        if it % 400 == 0 and it > 0:
            step *= 0.6
        i = RNG.integers(len(X))
        cand = X.copy()
        cand[i] += RNG.normal(0, step, 2)
        v = covering_radius(cand, G)
        if v < best - 1e-12:
            X, best = cand, v
        # 周期性做一次全量随机重启式扰动
        if it % 800 == 799:
            j = RNG.integers(len(X))
            cand = X.copy()
            cand[j] += RNG.normal(0, 0.15, 2)
            v = covering_radius(cand, G)
            if v < best:
                X, best = cand, v
    return X, best


def solve(n, G, restarts=12):
    best, bestX = 1e9, None
    # 起点 1：n 点均匀环
    for k in range(restarts):
        if k == 0:
            rho = 1.0 / math.sqrt(3.0)
            ang = np.arange(n) * 2 * math.pi / n
            X = np.stack([rho * np.cos(ang), rho * np.sin(ang)], 1)
        elif k == 1:
            rho = 0.8
            ang = np.arange(n - 1) * 2 * math.pi / (n - 1)
            X = np.vstack([[[0.0, 0.0]],
                           np.stack([rho * np.cos(ang), rho * np.sin(ang)], 1)])
        else:
            X = RNG.uniform(-0.7, 0.7, (n, 2))
        X, v = refine(X, G)
        if v < best:
            best, bestX = v, X
    return best, bestX


if __name__ == "__main__":
    G = sample_disk(20000)
    G2 = sample_disk(20000)          # 独立样本，用于验证不依赖采样
    print("=" * 70)
    print("单位圆盘覆盖：最小半径 r_n（n 个等圆）")
    print("=" * 70)
    res = {}
    for n in (5, 6, 7):
        r, X = solve(n, G, restarts=12)
        r2 = covering_radius(X, G2)
        res[n] = r
        print("  n=%d：r = %.6f（独立网格复核 %.6f）" % (n, r, r2))
    print()
    thr = 1000.0 / 1800.0
    print("  本文阈值 Rmin/R = %.6f" % thr)
    print("  r_6 = %.6f > 阈值 ⟹ 6 点**不足**（差 %.4f%%）"
          % (res[6], 100 * (res[6] / thr - 1)) if res[6] > thr
          else "  r_6 = %.6f <= 阈值 ⟹ 6 点足够（差 %.4f%%）"
               % (res[6], 100 * (1 - res[6] / thr)))
    print("  r_7 = %.6f，换算到 R=1800 为 %.1f m" % (res[7], res[7] * 1800))

