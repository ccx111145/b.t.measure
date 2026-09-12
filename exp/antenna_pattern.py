# -*- coding: utf-8 -*-
"""
1a 真实方向图下的检测完备性：25 点网还能覆盖多少？需要多密？

判据 (3)：对每个朝向 d，至少有一个网络点落在可检区域 A(d) 内。
理想半平面下它退化为论文定理 3 的 G ∈ conv(S_G)。
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp"))

from sim.antenna import PATTERNS, detectable_for_all_orientations   # noqa: E402
from nonconvex import hex_lattice                                   # noqa: E402

ARENA_R, R_REF = 1800.0, 1500.0


def two_layer():
    inner = hex_lattice(1000.0, 1800.0)
    outer = np.array([(1900 * math.cos(math.pi / 12 + 2 * math.pi * i / 12),
                       1900 * math.sin(math.pi / 12 + 2 * math.pi * i / 12))
                      for i in range(12)])
    return np.vstack([inner, outer])


def grid(step):
    xs = np.arange(-ARENA_R, ARENA_R + 1e-9, step)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    m = X * X + Y * Y <= ARENA_R ** 2
    return np.stack([X[m], Y[m]], 1)


def frac_ok(P, G, pattern, n_orient=120):
    ok = 0
    for g in G:
        if detectable_for_all_orientations(P, tuple(g), pattern, R_REF, n_orient):
            ok += 1
    return ok / len(G)


def min_hex_step(pattern, G, steps=(1000, 800, 600, 450, 350, 250, 180, 120),
                 n_orient=120, need=0.999):
    """找到使判据 (3) 全域成立的**最小**六角点阵间距（返回 step 与点数）"""
    for st in steps:
        P = hex_lattice(float(st), ARENA_R + 3.0 * st)
        f = frac_ok(P, G, pattern, n_orient)
        if f >= need:
            return st, len(P), f
    return None, None, 0.0


if __name__ == "__main__":
    G = grid(120.0)
    print("评测网格 %d 点\n" % len(G))
    P25 = two_layer()
    print("=" * 96)
    print("A. 论文的 25 点两层网在各方向图下的可检出比例")
    print("=" * 96)
    print("%-44s %10s %12s %12s" % ("方向图", "可检出比例", "最坏朝向半径", "前向半径"))
    for name, g in PATTERNS.items():
        t0 = time.monotonic()
        f = frac_ok(P25, G, g)
        psi = np.linspace(0, math.pi, 721)
        vals = np.asarray(g(psi))
        print("%-44s %10.3f %12.0f m %10.0f m   (%.0fs)"
              % (name, f, R_REF * vals.min(), R_REF * vals.max(), time.monotonic() - t0))

    print()
    print("=" * 96)
    print("B. 使判据 (3) 全域成立所需的六角点阵间距（越小越密）")
    print("=" * 96)
    print("%-44s %10s %10s %12s" % ("方向图", "最小间距", "点数", "可检出比例"))
    for name, g in PATTERNS.items():
        t0 = time.monotonic()
        st, n, f = min_hex_step(g, G)
        print("%-44s %10s %10s %12.3f   (%.0fs)"
              % (name, ("%.0f m" % st) if st else ">1000 m",
                 n if n else "-", f, time.monotonic() - t0))
