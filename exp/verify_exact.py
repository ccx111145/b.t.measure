# -*- coding: utf-8 -*-
"""
连续区域的**精确严格**认证（回应审稿意见 1）

原理
----
对单元 C（边长 h 的方格）：
  1. 若没有任何网络圆周边界 ∂B(P_i, R_min) 穿过 C，则 S_G = {P : |P-G| <= R_min}
     在整个 C 上恒定，取 C 的中心算一次即可；
  2. 此时条件 "G ∈ conv(S)" 关于 G 是**凸**的（conv(S) 是凸多边形）。
     由于 C 是凸的（正方形），**四个角点都在 conv(S) 内 ⟹ 整个 C ⊆ conv(S)**。
     —— 这是精确判据，没有余量损失。
  3. 若确有圆周穿过 C，则把 C 细分后递归处理。

因此"每个单元被认证" ⟹ 检测完备性对区域内**每一点**成立（除细分到极限的边界情形）。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp"))

from verify_continuous import ARENA_R, R_LO, two_layer, ring_net   # noqa: E402


def in_conv(P, G) -> bool:
    """G 是否在 P 的凸包内（极角间隙判据，单点版本）"""
    v = P - np.asarray(G, float)
    d = np.hypot(v[:, 0], v[:, 1])
    near = v[d <= R_LO + 1e-9]
    if len(near) == 0:
        return False
    dd = np.hypot(near[:, 0], near[:, 1])
    if (dd <= 1e-9).any():
        return True
    if len(near) < 3:
        # 两点：只有 G 恰在连线上才算（此时极点角差为 pi）
        if len(near) == 2:
            return abs(near[0][0] * near[1][1] - near[0][1] * near[1][0]) < 1e-9 \
                and (near[0] @ near[1]) < 0
        return False
    a = np.sort(np.arctan2(near[:, 1], near[:, 0]))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * math.pi]]))
    return bool(gaps.max() <= math.pi + 1e-9)


def circle_crosses_cell(cx, cy, h, P):
    """是否有某个 |x-P|=R_LO 的圆周穿过以 (cx,cy) 为中心、边长 h 的方格"""
    half = h / 2.0
    # 方格到 P 的最近/最远距离
    dx = max(abs(cx - P[0]) - half, 0.0)
    dy = max(abs(cy - P[1]) - half, 0.0)
    dmin = math.hypot(dx, dy)
    # 最远点必在某个角
    dmax = max(math.hypot(cx + sx * half - P[0], cy + sy * half - P[1])
               for sx in (-1, 1) for sy in (-1, 1))
    return dmin <= R_LO <= dmax


def certify_exact(P, h, max_depth=6, verbose=False):
    half = h / 2.0
    xs = np.arange(-ARENA_R + half, ARENA_R, h)
    stats = {"tot": 0, "bad": 0, "split": 0}
    worst = None

    def rec(cx, cy, hh, depth):
        # 只处理完全落在区域内的单元
        if math.hypot(cx, cy) + hh * math.sqrt(2) / 2 > ARENA_R + 1e-9:
            return True
        cross = any(circle_crosses_cell(cx, cy, hh, Q) for Q in P)
        if cross and depth < max_depth:
            stats["split"] += 1
            q = hh / 2.0                     # 子单元边长 = hh/2，半边长 = hh/4
            return all(rec(cx + sx * q / 2.0, cy + sy * q / 2.0, q, depth + 1)
                       for sx in (-1, 1) for sy in (-1, 1))
        stats["tot"] += 1
        S = P[np.hypot(P[:, 0] - cx, P[:, 1] - cy) <= R_LO + 1e-9]
        if len(S) == 0:
            stats["bad"] += 1
            return False
        hh2 = hh / 2.0
        corners = [(cx + sx * hh2, cy + sy * hh2)
                   for sx in (-1, 1) for sy in (-1, 1)]
        ok = all(in_conv(S, c) for c in corners)
        if not ok:
            stats["bad"] += 1
        return ok

    for x0 in xs:
        for y0 in xs:
            rec(x0, y0, h, 0)
    return stats


if __name__ == "__main__":
    print("== 精确严格认证（凸性 + 角点判据）==")
    for name, P in (("七点环状网", ring_net()), ("25 点两层网", two_layer())):
        for h in (20.0, 10.0, 5.0):
            st = certify_exact(P, h, max_depth=4)
            print("  %-12s h=%4.0f m：单元 %6d，未通过 %d，细分 %d"
                  % (name, h, st["tot"], st["bad"], st["split"]))
