# -*- coding: utf-8 -*-
"""
从构造侧收窄 [9, 52]：先用较密点阵保证可行，再贪心剪枝
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp"))

from nonconvex import (ARENA_R, Rect, detectable, hex_lattice, los)   # noqa: E402
from nonconvex_run import OBSTACLES, two_layer                        # noqa: E402


def grid_pts(h):
    out = []
    for x in np.arange(-ARENA_R, ARENA_R + 1e-9, h):
        for y in np.arange(-ARENA_R, ARENA_R + 1e-9, h):
            if math.hypot(x, y) > ARENA_R:
                continue
            if any(o.contains((x, y)) for o in OBSTACLES):
                continue
            out.append((x, y))
    return out


def n_bad(P, G):
    return sum(0 if detectable(P, g, OBSTACLES) else 1 for g in G)


def prune(P, G, max_pass=40):
    P = P.copy()
    for _ in range(max_pass):
        improved = False
        base = n_bad(P, G)
        order = np.argsort(-np.hypot(P[:, 0], P[:, 1]))     # 先试删外侧点
        for i in order:
            Q = np.delete(P, i, axis=0)
            if n_bad(Q, G) <= base:
                P = Q
                improved = True
                break
        if not improved:
            break
    return P


if __name__ == "__main__":
    G = grid_pts(80.0)
    print("评测网格 %d 点" % len(G))
    print("=" * 76)
    print("A. 起点：密点阵（保证可行）")
    print("=" * 76)
    for step in (900.0, 800.0, 700.0):
        P = hex_lattice(step, ARENA_R + 2 * step, OBSTACLES)
        bad = n_bad(P, G)
        print("  hex %.0f m：%3d 点，未通过位置 %d" % (step, len(P), bad))
        if bad == 0:
            break

    print()
    print("=" * 76)
    print("B. 贪心剪枝")
    print("=" * 76)
    t0 = time.monotonic()
    P1 = prune(P, G)
    print("  剪枝后：%d 点（从 %d 点），未通过 %d，用时 %.1f s"
          % (len(P1), len(P), n_bad(P1, G), time.monotonic() - t0))

    print()
    print("=" * 76)
    print("C. 对照")
    print("=" * 76)
    P25 = two_layer()
    print("  圆盘最优 25 点网，在该障碍工作区上未通过位置 %d / %d" % (n_bad(P25, G), len(G)))
    print("  贪心补点解 52 点，未通过 %d" % n_bad(hex_lattice(1000.0, ARENA_R, OBSTACLES), G))

    print()
    print("=" * 76)
    print("D. 结论区间")
    print("=" * 76)
    print("  严格下界 k >= 9（多重覆盖）")
    print("  构造上界 k <= %d（剪枝解）" % len(P1))
    np.save("tab/nonconvex_pruned.npy", P1)
