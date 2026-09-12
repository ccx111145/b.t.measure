# -*- coding: utf-8 -*-
"""
问题4 巡测网再优化（向量化版）：六角点阵 step=1000（31 点）是否必要？

条件：G ∈ conv(S_G)，S_G = 网中到 G 距离 ≤ R_min = 1000 m 的点。

直觉：`最近邻间距 ≤ R_min` 只是**充分**条件。
  * 内部：G 被最近 3 点围住即可，六角格三角形外接半径 d/√3 ⟹ 内部只需 d ≤ 1732 m；
  * 边界：朝外辐射的源需要**外侧**有点，这才是逼高密度的真正原因。
故把"内层点阵"与"外围环"分开搜索。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

R_LO = 1000.0
ARENA_R = 1800.0


def sample_arena(step: float):
    xs = np.arange(-ARENA_R, ARENA_R + 1e-9, step)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    m = X * X + Y * Y <= ARENA_R * ARENA_R
    return np.stack([X[m], Y[m]], 1).astype(np.float32)


def guarantee_mask(P: np.ndarray, S: np.ndarray) -> np.ndarray:
    """向量化：返回每个采样点是否满足 G ∈ conv(S_G)（纯 numpy 暴力查询，点数少）"""
    if len(P) == 0:
        return np.zeros(len(S), bool)
    P = P.astype(np.float32)
    S = S.astype(np.float32)
    dx = P[None, :, 0] - S[:, None, 0]          # (n, m)
    dy = P[None, :, 1] - S[:, None, 1]
    d2 = dx * dx + dy * dy
    valid = d2 <= np.float32(R_LO * R_LO + 1e-3)
    ang = np.where(valid, np.arctan2(dy, dx), np.float32(np.inf))
    ang.sort(axis=1)
    cnt = valid.sum(axis=1)
    gaps = np.diff(ang, axis=1)
    gaps = np.where(np.isfinite(gaps), gaps, 0.0)
    m = ang.shape[1]
    maxgap = gaps.max(axis=1) if m > 1 else np.zeros(len(S), dtype=np.float32)
    first = ang[:, 0]
    safe = np.clip(cnt - 1, 0, m - 1)
    last = ang[np.arange(len(S)), safe]
    wrap = np.where(cnt > 0, 2 * math.pi - (last - first), 2 * math.pi)
    maxgap = np.maximum(maxgap, wrap)
    ok = (cnt >= 2) & (maxgap <= math.pi + 1e-6)
    ok |= np.sqrt(d2[:, 0]) <= 1e-4
    return ok


def ratio(P, S) -> float:
    return float(guarantee_mask(P, S).mean())


def hex_pts(step: float, extent: float):
    pts = []
    dy = step * math.sqrt(3) / 2
    ny = int(2 * extent / dy) + 2
    nx = int(2 * extent / step) + 2
    for j in range(-ny, ny + 1):
        y = j * dy
        off = (step / 2) if (j % 2) else 0.0
        for i in range(-nx, nx + 1):
            x = i * step + off
            if math.hypot(x, y) <= extent + 1e-9:
                pts.append((x, y))
    return np.array(pts) if pts else np.zeros((0, 2))


def ring(n: int, rho: float, phase: float = 0.0):
    return np.array([(rho * math.cos(phase + 2 * math.pi * i / n),
                      rho * math.sin(phase + 2 * math.pi * i / n)) for i in range(n)])


def route_len(P: np.ndarray, start=(0.0, 0.0)) -> float:
    n = len(P)
    if n == 0:
        return 0.0
    s = np.asarray(start, float)
    k = int(np.argmin(np.hypot(P[:, 0] - s[0], P[:, 1] - s[1])))
    order = [k]
    left = set(range(n)) - {k}
    cur = P[k]
    while left:
        j = min(left, key=lambda i: float(np.hypot(*(P[i] - cur))))
        order.append(j)
        left.discard(j)
        cur = P[j]
    seq = [s] + [P[i] for i in order]
    improved, guard = True, 0
    while improved and guard < 100:
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
    S_coarse = sample_arena(30.0)
    S_fine = sample_arena(10.0)
    print("粗网格 %d 点，细网格 %d 点" % (len(S_coarse), len(S_fine)))

    print("\n" + "=" * 84)
    print("A. 失败分布：六角点阵（extent=2800）的满足率随步长变化")
    print("=" * 84)
    r_all = np.hypot(S_coarse[:, 0], S_coarse[:, 1])
    bins = [(0, 600), (600, 1200), (1200, 1800)]
    for step in (1000.0, 1200.0, 1400.0, 1600.0, 1732.0):
        P = hex_pts(step, 2800.0)
        ok = guarantee_mask(P, S_coarse)
        parts = []
        for a, b in bins:
            m = (r_all >= a) & (r_all < b)
            parts.append("%d-%d:%.3f" % (a, b, ok[m].mean() if m.sum() else float("nan")))
        print("  step=%4.0f（%2d 点）总=%.4f | %s" % (step, len(P), ok.mean(), "  ".join(parts)))

    print("\n" + "=" * 84)
    print("B. 搜索：内层点阵 + 外围环（粗筛 → 细验）")
    print("=" * 84)
    cands = []
    for d in (1200.0, 1400.0, 1600.0, 1732.0):
        for R_in in (1400.0, 1800.0, 2200.0):
            inner = hex_pts(d, R_in)
            for n_out in (0, 8, 9, 10, 11, 12, 14, 16, 20):
                for rho_out in (2000.0, 2200.0, 2400.0, 2600.0):
                    if n_out == 0:
                        if R_in < 2800.0:
                            continue
                        P = inner
                    else:
                        P = np.vstack([inner, ring(n_out, rho_out,
                                                   math.pi / max(1, n_out))])
                    if len(P) > 45 or len(P) == 0:
                        continue
                    if ratio(P, S_coarse) >= 0.9999:
                        cands.append((route_len(P), len(P), d, R_in, n_out, rho_out, P))
    cands.sort(key=lambda t: t[0])
    keep = []
    for L, n, d, R_in, n_out, rho_out, P in cands:
        if ratio(P, S_fine) >= 0.99999:
            keep.append((L, n, d, R_in, n_out, rho_out, P))
    print("粗筛通过 %d 个，细验通过 %d 个" % (len(cands), len(keep)))
    print("%-46s %5s %10s" % ("布局", "点数", "路线(m)"))
    for L, n, d, R_in, n_out, rho_out, P in keep[:15]:
        print("内层 hex d=%4.0f R=%4.0f + 外环 n=%2d rho=%4.0f     %5d %10.0f"
              % (d, R_in, n_out, rho_out, n, L))

    cur = hex_pts(1000.0, 2800.0)
    L_cur = route_len(cur)
    print()
    print("当前方案：hex step=1000 extent=2800  %2d 点  路线 %6.0f m（移动 %.0f s）"
          % (len(cur), L_cur, L_cur / 5))
    if keep:
        b = keep[0]
        print("最优发现：%2d 点  路线 %6.0f m（移动 %.0f s）  省 %.0f m = %.0f s"
              % (b[1], b[0], b[0] / 5, L_cur - b[0], (L_cur - b[0]) / 5))
        print("参数：内层 hex d=%.0f extent=%.0f，外环 n=%d rho=%.0f"
              % (b[2], b[3], b[4], b[5]))
        print("坐标：", np.round(b[6], 0).astype(int).tolist())


if __name__ == "__main__":
    main()
