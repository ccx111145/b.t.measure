# -*- coding: utf-8 -*-
"""
问题4 巡测网：第三方独立校验 + 重新搜索"密集内层 + 外环"

校验：G∈conv(S_G) 用三种互相独立的方法实现
  (1) 逐点极角间隙（Python）
  (2) 向量化极角间隙
  (3) **三角形剖分判定**：枚举所有三点组合，用重心坐标判断 G 是否落在某个三角形内
      （与极角法完全不同的算法，可交叉验证）
搜索：内层必须是 d≤1000 的密集点阵（已被证明 d=1200 不满足），
      再看外围环能否用更少的点替代 extent=2800 的整片点阵。
"""
from __future__ import annotations

import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARENA_R, R_LO = 1800.0, 1000.0


def hex_pts(step, extent):
    pts, dy = [], step * math.sqrt(3) / 2
    ny, nx = int(2 * extent / dy) + 2, int(2 * extent / step) + 2
    for j in range(-ny, ny + 1):
        y = j * dy
        off = (step / 2) if (j % 2) else 0.0
        for i in range(-nx, nx + 1):
            x = i * step + off
            if math.hypot(x, y) <= extent + 1e-9:
                pts.append((x, y))
    return np.array(pts) if pts else np.zeros((0, 2))


def ring(n, rho, phase=0.0):
    return np.array([(rho * math.cos(phase + 2 * math.pi * i / n),
                      rho * math.sin(phase + 2 * math.pi * i / n)) for i in range(n)])


def in_hull_angle(g, near):
    if len(near) == 0:
        return False
    d = near - np.asarray(g, float)
    if (np.hypot(d[:, 0], d[:, 1]) <= 1e-6).any():
        return True
    if len(near) < 3:
        return False
    a = np.sort(np.arctan2(d[:, 1], d[:, 0]))
    return bool(np.diff(np.concatenate([a, [a[0] + 2 * math.pi]])).max() <= math.pi + 1e-9)


def in_hull_tri(g, near):
    """独立实现：枚举三角形 + 重心坐标"""
    m = len(near)
    if m == 0:
        return False
    d = near - np.asarray(g, float)
    if (np.hypot(d[:, 0], d[:, 1]) <= 1e-6).any():
        return True
    if m < 3:
        return False
    for i, j, k in itertools.combinations(range(m), 3):
        A, B, C = near[i], near[j], near[k]
        v0, v1, v2 = B - A, C - A, np.asarray(g, float) - A
        den = v0[0] * v1[1] - v1[0] * v0[1]
        if abs(den) < 1e-9:
            continue
        u = (v2[0] * v1[1] - v1[0] * v2[1]) / den
        v = (v0[0] * v2[1] - v2[0] * v0[1]) / den
        if u >= -1e-9 and v >= -1e-9 and u + v <= 1 + 1e-9:
            return True
    return False


def ratio_angle(P, S):
    n_ok = 0
    for g in S:
        d = np.hypot(P[:, 0] - g[0], P[:, 1] - g[1])
        if in_hull_angle(g, P[d <= R_LO + 1e-9]):
            n_ok += 1
    return n_ok / len(S)


def ratio_tri(P, S):
    n_ok = 0
    for g in S:
        d = np.hypot(P[:, 0] - g[0], P[:, 1] - g[1])
        if in_hull_tri(g, P[d <= R_LO + 1e-9]):
            n_ok += 1
    return n_ok / len(S)


def route_len(P, start=(0.0, 0.0)):
    n = len(P)
    if n == 0:
        return 0.0
    s = np.asarray(start, float)
    k = int(np.argmin(np.hypot(P[:, 0] - s[0], P[:, 1] - s[1])))
    order, left, cur = [k], set(range(n)) - {k}, P[k]
    while left:
        j = min(left, key=lambda i: float(np.hypot(*(P[i] - cur))))
        order.append(j)
        left.discard(j)
        cur = P[j]
    seq = [s] + [P[i] for i in order]
    improved, guard = True, 0
    while improved and guard < 60:
        improved, guard = False, guard + 1
        for i in range(1, len(seq) - 1):
            for j in range(i + 1, len(seq)):
                old = float(np.hypot(*(seq[i - 1] - seq[i])))
                new = float(np.hypot(*(seq[i - 1] - seq[j])))
                if j + 1 < len(seq):
                    old += float(np.hypot(*(seq[j] - seq[j + 1])))
                    new += float(np.hypot(*(seq[i] - seq[j + 1])))
                if new < old - 1e-9:
                    seq[i:j + 1] = seq[i:j + 1][::-1]
                    improved = True
    return float(sum(np.hypot(*(seq[i + 1] - seq[i])) for i in range(len(seq) - 1)))


def main():
    rng = np.random.default_rng(7)
    n = 1200
    r = ARENA_R * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * math.pi
    S = np.stack([r * np.cos(a), r * np.sin(a)], 1)

    print("=" * 80)
    print("第三方校验：极角法 vs 三角形重心坐标法（1200 个随机采样点）")
    print("=" * 80)
    for step, ext in ((1000.0, 2800.0), (1000.0, 1800.0), (1200.0, 2800.0)):
        P = hex_pts(step, ext)
        r1, r2 = ratio_angle(P, S), ratio_tri(P, S)
        print("  hex step=%.0f extent=%.0f（%2d 点）：极角法 %.4f，三角剖分法 %.4f，差 %.4f"
              % (step, ext, len(P), r1, r2, abs(r1 - r2)))

    print()
    print("=" * 80)
    print("重新搜索：密集内层（d ≤ 1000）+ 外围环")
    print("=" * 80)
    cands = []
    for d in (800.0, 900.0, 1000.0):
        for R_in in (1200.0, 1500.0, 1800.0, 2100.0):
            inner = hex_pts(d, R_in)
            for n_out in (8, 9, 10, 11, 12, 14, 16, 18, 20):
                for rho_out in (2000.0, 2200.0, 2400.0, 2600.0, 2800.0):
                    P = np.vstack([inner, ring(n_out, rho_out, math.pi / n_out)])
                    if len(P) > 40:
                        continue
                    if ratio_angle(P, S) >= 0.9999:
                        cands.append((route_len(P), len(P), d, R_in, n_out, rho_out, P))
    cands.sort(key=lambda t: t[0])
    print("可行组合 %d 个，按路线排序前 12：" % len(cands))
    print("%-52s %5s %10s" % ("布局", "点数", "路线(m)"))
    for L, n_, d, R_in, n_out, rho_out, P in cands[:12]:
        print("内层 hex d=%4.0f R=%4.0f（%2d 点）+ 外环 n=%2d rho=%4.0f   %5d %10.0f"
              % (d, R_in, len(hex_pts(d, R_in)), n_out, rho_out, n_, L))
    cur = hex_pts(1000.0, 2800.0)
    L_cur = route_len(cur)
    print()
    print("当前方案：hex step=1000 extent=2800  %d 点  路线 %.0f m（移动 %.0f s）"
          % (len(cur), L_cur, L_cur / 5))
    if cands:
        b = cands[0]
        print("最优发现：%d 点  路线 %.0f m（移动 %.0f s），省 %.0f m = %.0f s"
              % (b[1], b[0], b[0] / 5, L_cur - b[0], (L_cur - b[0]) / 5))
        print("坐标：", np.round(b[6], 0).astype(int).tolist())


if __name__ == "__main__":
    main()
