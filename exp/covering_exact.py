# -*- coding: utf-8 -*-
"""
精确计算给定圆心配置的覆盖半径（不用采样网格，避免过拟合）：
单位圆盘内 max_x min_i |x - x_i| 的最大值只可能出现在
  (a) Voronoi 顶点（三个圆心的外心）且落在单位圆盘内；
  (b) 单位圆周上 g(θ)=min_i |u(θ)-x_i| 的局部极大。
后者用细角度网格 + 局部二分精修。
"""
from __future__ import annotations

import itertools
import math
import sys

import numpy as np

RNG = np.random.default_rng(11)


def coverage_exact(X: np.ndarray, n_ang: int = 720000) -> float:
    best = 0.0
    m = len(X)
    # (a) Voronoi 顶点
    for i, j, k in itertools.combinations(range(m), 3):
        A, B, C = X[i], X[j], X[k]
        d = 2 * (A[0] * (B[1] - C[1]) + B[0] * (C[1] - A[1]) + C[0] * (A[1] - B[1]))
        if abs(d) < 1e-14:
            continue
        ux = ((A @ A) * (B[1] - C[1]) + (B @ B) * (C[1] - A[1])
              + (C @ C) * (A[1] - B[1])) / d
        uy = ((A @ A) * (C[0] - B[0]) + (B @ B) * (A[0] - C[0])
              + (C @ C) * (B[0] - A[0])) / d
        P = np.array([ux, uy])
        if P @ P > 1.0:
            continue
        dd = np.hypot(X[:, 0] - ux, X[:, 1] - uy)
        r = float(dd.min())
        # 只在该点确实是最近三个圆心的外心时才是局部极大
        if abs(r - math.hypot(ux - A[0], uy - A[1])) < 1e-9:
            best = max(best, r)
    # (b) 单位圆周
    th = np.arange(n_ang) * (2 * math.pi / n_ang)
    U = np.stack([np.cos(th), np.sin(th)], 1)
    g = np.min(np.hypot(U[:, None, 0] - X[None, :, 0],
                        U[:, None, 1] - X[None, :, 1]), axis=1)
    best = max(best, float(g.max()))
    return best


def ring(n, rho, phase=0.0):
    a = phase + np.arange(n) * 2 * math.pi / n
    return np.stack([rho * np.cos(a), rho * np.sin(a)], 1)


def center_ring(n, rho, phase=0.0):
    return np.vstack([[[0.0, 0.0]], ring(n, rho, phase)])


def optimize(X0, steps=(0.05, 0.02, 0.008, 0.003, 0.001), iters=600):
    X = X0.copy()
    best = coverage_exact(X)
    for st in steps:
        improved = True
        rounds = 0
        while improved and rounds < iters:
            improved = False
            rounds += 1
            for i in range(len(X)):
                for _ in range(6):
                    cand = X.copy()
                    cand[i] += RNG.normal(0, st, 2)
                    if cand[i] @ cand[i] > 1.2:      # 圆心可略在圆外
                        continue
                    v = coverage_exact(cand)
                    if v < best - 1e-10:
                        X, best, improved = cand, v, True
    return X, best


if __name__ == "__main__":
    thr = 1000.0 / 1800.0
    print("阈值 Rmin/R = %.7f" % thr)
    print("=" * 72)
    print("A. 已知构造（用于校验评估器）")
    print("=" * 72)
    r7 = center_ring(6, 1.0 / math.sqrt(3.0))
    print("  中心+6 环 ρ=1/√3 ⟹ 覆盖半径 %.6f（理论值 0.5）" % coverage_exact(r7))
    print("  6 环 ρ=1/√3        ⟹ 覆盖半径 %.6f（理论值 %.6f）"
          % (coverage_exact(ring(6, 1 / math.sqrt(3.0))), 1 / math.sqrt(3.0)))

    print()
    print("=" * 72)
    print("B. 6 个圆心的最优努力（多起点 + 精确评估）")
    print("=" * 72)
    cands = [ring(6, 1 / math.sqrt(3.0))]
    for rho in (0.70, 0.75, 0.80, 0.85):
        cands.append(center_ring(5, rho))
    cands.append(ring(3, 0.55).tolist() and np.vstack([ring(3, 0.5),
                                                       ring(3, 0.5, math.pi / 3)]))
    best, bX = 9.0, None
    for c in cands:
        X, v = optimize(np.asarray(c, float))
        print("   起点覆盖半径 %.6f ⟹ 优化后 %.6f" % (coverage_exact(np.asarray(c, float)), v))
        if v < best:
            best, bX = v, X
    print()
    print("  最优找到：r_6 = %.6f" % best)
    print("  与阈值比较：%s（差 %.3f%%）"
          % ("**大于** ⟹ 6 点不足" if best > thr else "小于等于 ⟹ 6 点可能足够",
             100 * (best / thr - 1)))
    print("  最优配置：")
    print(np.round(bX, 4).tolist())
