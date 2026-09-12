# -*- coding: utf-8 -*-
"""
问题4 巡测网设计：让"任意位置、任意朝向"的定向源都可被检出

理论条件
--------
定向源 G、朝向 d，检测点 P 能收到信号 ⟺ ``P ∈ B(G,R) ∩ H(G,d)``（R ≥ 1000）。
要让**任意 d** 都能被检出，必须存在网络点落在该半平面内；等价地

    G ∈ conv( S_G ),   S_G = { 网络点中到 G 距离 ≤ 1000 的点 }

（若所有近邻网络点都落在某个过 G 的闭半平面之外，则该朝向的源不可检出。）

本脚本数值验证若干点阵是否满足该条件，并给出路线长度与点数。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R_LO = 1000.0
ARENA_R = 1800.0


def in_convex_hull(p: np.ndarray, pts: np.ndarray, tol: float = 1e-6) -> bool:
    """点 p 是否落在点集 pts 的凸包内（二维，O(n log n) 走极角 + 逐边判）"""
    if len(pts) == 0:
        return False
    d = pts - p
    r = np.hypot(d[:, 0], d[:, 1])
    if (r <= tol).any():
        return True
    ang = np.arctan2(d[:, 1], d[:, 0])
    order = np.argsort(ang)
    a = ang[order]
    # 凸包包含 p ⟺ 相邻极角间隙不超过 π
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * math.pi]]))
    return bool(gaps.max() <= math.pi + 1e-9)


def check_lattice(pts: np.ndarray, n_sample: int = 40000, seed: int = 7,
                  extent: float = ARENA_R) -> float:
    """返回被"不可检出"的采样点比例（0 = 完美保证）"""
    rng = np.random.default_rng(seed)
    r = extent * np.sqrt(rng.random(n_sample))
    a = rng.random(n_sample) * 2 * math.pi
    G = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    bad = 0
    for g in G:
        d = np.hypot(pts[:, 0] - g[0], pts[:, 1] - g[1])
        near = pts[d <= R_LO + 1e-9]
        if not in_convex_hull(g, near):
            bad += 1
    return bad / len(G)


def square_lattice(step: float, extent: float = ARENA_R):
    xs = np.arange(-extent, extent + 1e-9, step)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel()], 1)
    keep = np.hypot(P[:, 0], P[:, 1]) <= extent + 1e-9
    return P[keep]


def square_lattice_full(step: float, extent: float = 2800.0):
    """不裁剪到目标圆，允许检测点落在区域外（题目允许机器狗走出区域）"""
    xs = np.arange(-extent, extent + 1e-9, step)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], 1)


def hex_lattice(step: float, extent: float = ARENA_R):
    pts = []
    ny = int(2 * extent / (step * math.sqrt(3) / 2)) + 2
    nx = int(2 * extent / step) + 2
    for j in range(-ny, ny + 1):
        y = j * step * math.sqrt(3) / 2
        off = (step / 2) if (j % 2) else 0.0
        for i in range(-nx, nx + 1):
            x = i * step + off
            if math.hypot(x, y) <= extent + 1e-9:
                pts.append((x, y))
    return np.array(pts) if pts else np.zeros((0, 2))


def hex_lattice_full(step: float, extent: float = 2800.0):
    pts = []
    ny = int(2 * extent / (step * math.sqrt(3) / 2)) + 2
    nx = int(2 * extent / step) + 2
    for j in range(-ny, ny + 1):
        y = j * step * math.sqrt(3) / 2
        off = (step / 2) if (j % 2) else 0.0
        for i in range(-nx, nx + 1):
            x = i * step + off
            if math.hypot(x, y) <= extent + 1e-9:
                pts.append((x, y))
    return np.array(pts) if pts else np.zeros((0, 2))


def two_ring(step: float, extent: float = ARENA_R):
    """内层方格外加一圈外圈（环半径 = extent + 600）"""
    P = square_lattice_full(step, extent)
    return P


def route_len(pts: np.ndarray, start=(0.0, 0.0)) -> float:
    n = len(pts)
    if n == 0:
        return 0.0
    s = np.asarray(start, float)
    k = int(np.argmin(np.hypot(pts[:, 0] - s[0], pts[:, 1] - s[1])))
    order = [k]
    left = set(range(n)) - {k}
    cur = pts[k]
    while left:
        j = min(left, key=lambda i: float(np.hypot(*(pts[i] - cur))))
        order.append(j)
        left.discard(j)
        cur = pts[j]
    seq = [s] + [pts[i] for i in order]
    improved = True
    while improved:
        improved = False
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
    print("=" * 92)
    print("条件：G ∈ conv(S_G)，S_G = 网络点中到 G 距离 ≤ 1000 m 的点（采样 20000 点）")
    print("检测点允许落在目标区域之外（题目允许机器狗走出区域，|坐标| ≤ 2e6）")
    print("=" * 92)
    print("%-30s %6s %12s %12s" % ("布局", "点数", "不可检出比例", "路线(m)"))
    rows = []
    for name, gen in (("方格(含区域外)", square_lattice_full),
                      ("六角(含区域外)", hex_lattice_full),
                      ("方格(裁剪)", square_lattice),
                      ("六角(裁剪)", hex_lattice)):
        for step in (700, 800, 900, 1000, 1100, 1200, 1300, 1400):
            P = gen(step)
            if len(P) == 0:
                continue
            bad = check_lattice(P, n_sample=20000)
            L = route_len(P)
            rows.append((bad, L, len(P), "%s step=%.0f" % (name, step)))
            print("%-30s %6d %12.4f %12.0f" % ("%s step=%.0f" % (name, step),
                                               len(P), bad, L))
    good = [r for r in rows if r[0] == 0.0]
    good.sort(key=lambda t: t[1])
    print()
    if good:
        print("满足『任意位置任意朝向可检出』且路线最短的布局：")
        for bad, L, n, name in good[:8]:
            print("   %-26s 点数=%2d 路线=%6.0f m 移动=%5.0f s" % (name, n, L, L / 5))
        best = good[0]
        print()
        print("推荐：%s  点数=%d  路线=%.0f m (移动 %.0f s)"
              % (best[3], best[2], best[1], best[1] / 5))
    else:
        print("没有任何布局满足保证！")
    ring = np.array([[0.0, 0.0]] + [(1200 * math.cos(2 * math.pi * k / 6),
                                     1200 * math.sin(2 * math.pi * k / 6)) for k in range(6)])
    print()
    print("对照 中心+6环 ρ=1200（问题3 用）：点数 7，路线 %.0f m，不可检出比例 %.4f"
          % (route_len(ring), check_lattice(ring, n_sample=20000)))


if __name__ == "__main__":
    main()
