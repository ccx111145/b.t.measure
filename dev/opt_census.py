# -*- coding: utf-8 -*-
"""
巡测网布局优化：在"覆盖半径 1800 圆域"的约束下最小化巡测路线长度

约束：目标圆内任一点到最近巡测点的距离 ≤ R_lo（保守取有效接收半径下界 1000 m），
     以保证**检测完备**与**证否完备**。
目标：min 路线长度（从原点出发的 TSP 近似解）。

搜索布局族：
  A. 中心 + k 环（半径 ρ）
  B. 纯 k 环
  C. 双环（等比半径，相对旋转 180/m 度）
  D. 六角格点
"""
from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARENA_R = 1800.0


def sample_grid(step: float = 10.0):
    xs = np.arange(-ARENA_R, ARENA_R + 1e-9, step)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    m = X * X + Y * Y <= ARENA_R * ARENA_R
    return np.stack([X[m], Y[m]], 1)


def coverage(pts, S) -> float:
    if len(pts) == 0:
        return float("inf")
    d = np.full(len(S), np.inf)
    for p in pts:
        d = np.minimum(d, np.hypot(S[:, 0] - p[0], S[:, 1] - p[1]))
    return float(d.max())


def route_len(pts, start=(0.0, 0.0)) -> float:
    """最近邻 + 2-opt 近似 TSP（从 start 出发，不必回到起点）"""
    n = len(pts)
    if n == 0:
        return 0.0
    P = [np.asarray(p, float) for p in pts]
    s = np.asarray(start, float)
    k = int(np.argmin([np.linalg.norm(p - s) for p in P]))
    order = [k]
    left = set(range(n)) - {k}
    cur = P[k]
    while left:
        j = min(left, key=lambda i: np.linalg.norm(P[i] - cur))
        order.append(j)
        left.discard(j)
        cur = P[j]
    seq = [s] + [P[i] for i in order]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(seq) - 1):
            for j in range(i + 1, len(seq)):
                a, b, c, d = seq[i - 1], seq[i], seq[j], seq[j - 1] if False else seq[j]
                # 2-opt：反转 seq[i..j]
                old = (np.linalg.norm(seq[i - 1] - seq[i]) + np.linalg.norm(seq[j] - seq[j + 1])
                       if j + 1 < len(seq) else
                       np.linalg.norm(seq[i - 1] - seq[i]) + 0.0)
                new = (np.linalg.norm(seq[i - 1] - seq[j]) + np.linalg.norm(seq[i] - seq[j + 1])
                       if j + 1 < len(seq) else
                       np.linalg.norm(seq[i - 1] - seq[j]) + 0.0)
                if new < old - 1e-9:
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    improved = True
    return float(sum(np.linalg.norm(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)))


def ring(k, rho, phase=0.0):
    return [(rho * math.cos(phase + 2 * math.pi * i / k),
             rho * math.sin(phase + 2 * math.pi * i / k)) for i in range(k)]


def main():
    S = sample_grid(10.0)
    print("覆盖采样点数 =", len(S))
    results = []

    # A/B 单环
    for k in range(3, 11):
        for rho in np.arange(700, 2100, 25.0):
            for center in (True, False) if k > 3 else (True,):
                pts = ([(0.0, 0.0)] if center else []) + ring(k, rho)
                c = coverage(pts, S)
                if c <= 1000.0:
                    results.append((route_len(pts), c, len(pts),
                                    "中心+%d环 ρ=%.0f" % (k, rho) if center else "%d环 ρ=%.0f" % (k, rho), pts))

    # C 双环
    for k1 in range(3, 9):
        for k2 in range(3, 9):
            for r1 in np.arange(700, 1600, 50.0):
                for r2 in np.arange(700, 1600, 50.0):
                    pts = ring(k1, r1) + ring(k2, r2, math.pi / max(1, k2))
                    c = coverage(pts, S)
                    if c <= 1000.0:
                        results.append((route_len(pts), c, len(pts),
                                        "双环 %d@%.0f + %d@%.0f" % (k1, r1, k2, r2), pts))

    # D 六角格
    for a in np.arange(700, 1900, 50.0):
        for b in np.arange(700, 1900, 50.0):
            pts = [(0.0, 0.0)]
            for i in range(-3, 4):
                for j in range(-3, 4):
                    if i == 0 and j == 0:
                        continue
                    x = a * (i + 0.5 * j)
                    y = b * j * math.sqrt(3) / 2 * (b / (b or 1))
                    y = b * j * math.sqrt(3) / 2
                    pts.append((x, y))
            pts = [p for p in pts if math.hypot(*p) <= 2600]
            c = coverage(pts, S)
            if c <= 1000.0 and len(pts) <= 12:
                results.append((route_len(pts), c, len(pts),
                                "六角格 a=%.0f b=%.0f (%d点)" % (a, b, len(pts)), pts))

    results.sort(key=lambda t: t[0])
    print()
    print("%-34s %6s %10s %10s" % ("布局", "点数", "最大未覆盖", "路线长度(m)"))
    for L, c, n, name, pts in results[:20]:
        print("%-34s %6d %10.1f %10.0f" % (name, n, c, L))
    print()
    best = results[0]
    print("最优:", best[3], " 点数=%d 最大未覆盖=%.2f m 路线=%.0f m (移动 %.0f s)"
          % (best[2], best[1], best[0], best[0] / 5))
    ref = [r for r in results if r[3].startswith("中心+6环 ρ=1200")]
    if ref:
        print("参考 中心+6环 ρ=1200: 路线=%.0f m (移动 %.0f s)，可省 %.0f s"
              % (ref[0][0], ref[0][0] / 5, (ref[0][0] - best[0]) / 5))
    print()
    print("最优布局坐标:", [(round(p[0], 1), round(p[1], 1)) for p in best[4]])


if __name__ == "__main__":
    main()
