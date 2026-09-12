# -*- coding: utf-8 -*-
"""
计算检测网络的多重覆盖冗余度 r：
    r = min over G∈A, over 朝向 d  of  #{ P ∈ P : ‖P-G‖ ≤ R_min 且 P ∈ H(G,d) }

含义：一个真实源不论朝哪，至少会被 r 个巡测点听到。
这是**抗离群值证否**的关键量：若读数中至多有 q 个伪装测向角，则
  * 真有源 ⟹ 检出点数 ≥ r（其中至少 r-q 个是真检出）
  * 空频道 ⟹ 检出点数 ≤ q（全部是伪装）
故当 r > q 时，"检出点数 ≤ q" 可判定为无源。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R_LO, ARENA_R = 1000.0, 1800.0


def hex_pts(step, extent):
    pts, dy = [], step * math.sqrt(3) / 2
    for j in range(-int(2 * extent / dy) - 2, int(2 * extent / dy) + 3):
        y = j * dy
        off = (step / 2) if (j % 2) else 0.0
        for i in range(-int(2 * extent / step) - 2, int(2 * extent / step) + 3):
            x = i * step + off
            if math.hypot(x, y) <= extent + 1e-9:
                pts.append((x, y))
    return np.array(pts)


def ring_net():
    return np.array([(0.0, 0.0)] + [(1200 * math.cos(2 * math.pi * i / 6),
                                     1200 * math.sin(2 * math.pi * i / 6))
                                    for i in range(6)])


def two_layer():
    inner = hex_pts(1000.0, 1800.0)
    outer = np.array([(1900 * math.cos(math.pi / 12 + 2 * math.pi * i / 12),
                       1900 * math.sin(math.pi / 12 + 2 * math.pi * i / 12))
                      for i in range(12)])
    return np.vstack([inner, outer])


def redundancy(P, S, n_ang=72):
    """r = min over G, d of 检出点数"""
    worst, worst_g = 10 ** 9, None
    for g in S:
        d = np.hypot(P[:, 0] - g[0], P[:, 1] - g[1])
        near = P[d <= R_LO + 1e-9]
        if len(near) == 0:
            return 0, tuple(g), None
        v = near - g
        # 每个朝向 d：H = 半平面 {p: (p-g)·u(d) >= 0}
        th = np.radians(np.arange(n_ang) * 360.0 / n_ang)
        cnt = (v[:, None, 0] * np.cos(th)[None, :]
               + v[:, None, 1] * np.sin(th)[None, :]) >= -1e-9
        c = int(cnt.sum(axis=0).min())
        if c < worst:
            worst, worst_g = c, tuple(np.round(g, 1))
    return worst, worst_g, None


def grid(step):
    xs = np.arange(-ARENA_R, ARENA_R + 1e-9, step)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    m = X * X + Y * Y <= ARENA_R ** 2
    return np.stack([X[m], Y[m]], 1)


if __name__ == "__main__":
    for name, P in (("七点环状网（问题3）", ring_net()),
                    ("25 点两层网（问题4）", two_layer())):
        for step in (60.0, 30.0):
            S = grid(step)
            r, g, bad = redundancy(P, S)
            print("%-22s 网格 %4.0f m（%6d 点）：r = %d，最坏位置 %s"
                  % (name, step, len(S), r, g))
        print()
