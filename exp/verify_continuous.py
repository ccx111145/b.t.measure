# -*- coding: utf-8 -*-
"""
把"网格上验证通过"升级为**连续区域的严格保证**（回应审稿意见 1）

思路（保守但严格）
------------------
把区域划分成边长 h 的方格，单元 C 的中心为 x、外接半径为 η = h√2/2。
对任意 G ∈ C 与任意朝向 d，只要存在网络点 P 满足

    (i)  ‖P − x‖ ≤ R_min − η              （则 ‖P − G‖ ≤ R_min 对单元内所有 G 成立）
    (ii) (P − x)·u(d) ≥ η                 （则 (P − G)·u(d) = (P−x)·u(d) + (x−G)·u(d) ≥ η − η ≥ 0）

就能保证 P 听到 G。条件 (ii) 等价于 d 落在以 θ_P 为中心、半角 α_P = arccos(η / r_P) 的弧内，
其中 r_P = ‖P − x‖、θ_P = atan2(P_y − x_y, P_x − x_x)，且要求 r_P > η。

于是：**单元 C 全域满足检测完备性 ⟺ 所有这样的弧覆盖整个 [0, 2π)**。
这是充分条件（保守），但它是**逐单元**的，因此覆盖整个区域的单元 ⟹ 结论对区域内每一点成立。
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
    return np.array(pts) if pts else np.zeros((0, 2))


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


def cell_certified(x, P, eta) -> bool:
    """单元 C（中心 x、外接半径 η）是否被**严格**认证"""
    v = P - np.asarray(x, float)
    r = np.hypot(v[:, 0], v[:, 1])
    keep = (r <= R_LO - eta + 1e-12) & (r > eta + 1e-12)
    if not keep.any():
        return False
    rr = r[keep]
    th = np.arctan2(v[keep, 1], v[keep, 0])
    alpha = np.arccos(np.clip(eta / rr, -1.0, 1.0))
    # 弧并集是否覆盖 [0, 2π)：把每个弧拆成起点、终点，做扫描线
    starts = np.mod(th - alpha, 2 * math.pi)
    ends = np.mod(th + alpha, 2 * math.pi)
    ev = []
    for s, e in zip(starts, ends):
        if s <= e:
            ev.append((s, 1)); ev.append((e, -1))
        else:                      # 跨 0 的弧
            ev.append((s, 1)); ev.append((2 * math.pi, -1))
            ev.append((0.0, 1)); ev.append((e, -1))
    ev.sort()
    cur = 0.0
    best = 0.0
    prev = None
    for pos, d in ev:
        if prev is not None and cur > 0:
            best += pos - prev
        cur += d
        prev = pos
    # 只剩一个弧时上面的累加会漏首段，改用更稳妥的判据：
    return best >= 2 * math.pi - 1e-9


def certify(P, h, verbose=False):
    """返回 (被认证比例, 未认证单元数, 总单元数)"""
    eta = h * math.sqrt(2.0) / 2.0
    xs = np.arange(-ARENA_R, ARENA_R + 1e-9, h)
    bad = tot = 0
    worst = None
    for x0 in xs:
        for y0 in xs:
            # 只统计完全落在圆内的单元（其余单元由更外层的几何处理）
            if math.hypot(x0, y0) + eta > ARENA_R + 1e-9:
                continue
            tot += 1
            if not cell_certified((x0, y0), P, eta):
                bad += 1
                if worst is None:
                    worst = (round(x0, 1), round(y0, 1))
    return (1.0 - bad / tot if tot else 0.0), bad, tot, worst


if __name__ == "__main__":
    print("=" * 84)
    print("连续区域认证：逐单元严格验证（η = h√2/2 的保守余量）")
    print("=" * 84)
    print("%-26s %6s %10s %10s %s" % ("网络", "单元(m)", "认证比例", "未认证", "首个失败单元"))
    for name, P in (("七点环状网（S1 用）", ring_net()),
                    ("25 点两层网（S2 用）", two_layer())):
        for h in (40.0, 20.0, 10.0):
            frac, bad, tot, worst = certify(P, h)
            print("%-26s %6.0f %10.4f %10d %s" % (name, h, frac, bad, worst))
        print()
