# -*- coding: utf-8 -*-
"""
严格的连续域认证（保守 cell certificate，无需细分）

对边长 h 的方格 C（中心 c、半边长 a、外接半径 eta）：
    S_C^- = { P : max_{G in C} |P - G| <= R_min }
因为 |P-G| 关于 G 是凸的，最大值必在某个角点取到，故
    max_{G in C} |P-G| = max over 4 corners |P - corner|
于是 S_C^- 可用四个角点精确判定。

        S_C^-  ⊆  S_G      对**所有** G ∈ C 成立
    C ⊆ conv(S_C^-)  ⟹  G ∈ conv(S_C^-) ⊆ conv(S_G)  ∀G ∈ C
而 C 是四个角点的凸包、conv(S_C^-) 是凸集，
故只需检查**四个角点是否都在 conv(S_C^-) 中**。

这个判据**不依赖** R_min 圆周是否穿过 C，因此不需要递归细分。
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))

from verify_continuous import ARENA_R, R_LO, ring_net, two_layer   # noqa: E402


def in_conv(P, G) -> bool:
    """G 是否在有限点集 P 的凸包内（极角间隙判据）"""
    v = P - np.asarray(G, float)
    d = np.hypot(v[:, 0], v[:, 1])
    near = v[d <= R_LO + 1e-9]
    if len(near) == 0:
        return False
    dd = np.hypot(near[:, 0], near[:, 1])
    if (dd <= 1e-9).any():
        return True                      # G 与网络点重合
    if len(near) == 1:
        return False
    if len(near) == 2:
        return False                     # 两点：只有恰在连线（测度零）才成立
    a = np.sort(np.arctan2(near[:, 1], near[:, 0]))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * math.pi]]))
    return bool(gaps.max() <= math.pi + 1e-9)


def cell_certified_conservative(P, c, h) -> bool:
    """严格保守判据（不需要细分）"""
    a = h / 2.0
    corners = np.array([[c[0] + sx * a, c[1] + sy * a]
                        for sx in (-1, 1) for sy in (-1, 1)])
    # max_{G in C} |P - G| = 四个角点上的最大距离
    dmax = np.max(np.hypot(P[None, :, 0] - corners[:, None, 0],
                           P[None, :, 1] - corners[:, None, 1]), axis=0)
    S = P[dmax <= R_LO + 1e-9]           # = S_C^-
    if len(S) < 3:
        return False
    return all(in_conv(S, tuple(g)) for g in corners)


def certify(P, h):
    half = h / 2.0
    ok = bad = 0
    first = None
    for x in np.arange(-ARENA_R + half, ARENA_R, h):
        for y in np.arange(-ARENA_R + half, ARENA_R, h):
            c = (x, y)
            if math.hypot(x, y) + half * math.sqrt(2) > ARENA_R + 1e-9:
                continue                 # 只统计完全落在区域内的单元
            if cell_certified_conservative(P, c, h):
                ok += 1
            else:
                bad += 1
                if first is None:
                    first = (round(x, 1), round(y, 1))
    return ok, bad, first


if __name__ == "__main__":
    print("=" * 88)
    print("严格保守 cell certificate（无需递归细分）")
    print("=" * 88)
    for name, P in (("25 点两层网", two_layer()), ("七点环状网", ring_net())):
        for h in (40.0, 20.0, 10.0):
            t0 = time.monotonic()
            ok, bad, first = certify(P, h)
            print("  %-12s h=%4.0f m：单元 %7d，通过 %7d，失败 %6d（%.4f%%），首个失败 %s  [%.0fs]"
                  % (name, h, ok + bad, ok, bad, 100.0 * ok / max(1, ok + bad),
                     first, time.monotonic() - t0), flush=True)
        print()
