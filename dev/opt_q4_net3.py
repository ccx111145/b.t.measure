# -*- coding: utf-8 -*-
"""
在"密集内层点阵 + 紧贴区域的外环"这一族里精细搜索，并用**独立方法**复核最优解
  * 极角间隙法（向量化）
  * 三角形重心坐标法（独立算法）
  * 10 m 细网格（101765 点）
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


def sample_arena(step):
    xs = np.arange(-ARENA_R, ARENA_R + 1e-9, step)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    m = X * X + Y * Y <= ARENA_R * ARENA_R
    return np.stack([X[m], Y[m]], 1)


def ok_angle_vec(P, S, chunk: int = 400):
    """分块计算，避免单次分配过大（本机提交内存紧张）"""
    P32 = P.astype(np.float32)
    out = np.empty(len(S), bool)
    for a in range(0, len(S), chunk):
        b = min(a + chunk, len(S))
        S32 = S[a:b].astype(np.float32)
        dx = P32[None, :, 0] - S32[:, None, 0]
        dy = P32[None, :, 1] - S32[:, None, 1]
        d2 = dx * dx + dy * dy
        valid = d2 <= np.float32(R_LO * R_LO + 1e-3)
        ang = np.where(valid, np.arctan2(dy, dx), np.float32(np.inf))
        ang.sort(axis=1)
        cnt = valid.sum(axis=1)
        gaps = np.diff(ang, axis=1)
        gaps = np.where(np.isfinite(gaps), gaps, 0.0)
        m = ang.shape[1]
        maxgap = gaps.max(axis=1) if m > 1 else np.zeros(b - a, dtype=np.float32)
        first = ang[:, 0]
        last = ang[np.arange(b - a), np.clip(cnt - 1, 0, m - 1)]
        wrap = np.where(cnt > 0, 2 * math.pi - (last - first), 2 * math.pi)
        maxgap = np.maximum(maxgap, wrap)
        ok = (cnt >= 2) & (maxgap <= math.pi + 1e-6)
        ok |= (d2.min(axis=1) <= np.float32(1e-6))
        out[a:b] = ok
        del dx, dy, d2, valid, ang, gaps
    return out


def in_hull_tri(g, near):
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


def ratio_tri(P, S):
    ok = 0
    for g in S:
        d = np.hypot(P[:, 0] - g[0], P[:, 1] - g[1])
        if in_hull_tri(g, P[d <= R_LO + 1e-9]):
            ok += 1
    return ok / len(S)


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
    while improved and guard < 80:
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
    S_c = sample_arena(40.0)
    print("粗网格 %d 点" % len(S_c))
    cands = []
    for d in (900.0, 950.0, 1000.0):
        for R_in in np.arange(1400, 2201, 100.0):
            inner = hex_pts(d, float(R_in))
            for n_out in range(8, 19):
                for rho in np.arange(1850, 2501, 50.0):
                    P = np.vstack([inner, ring(n_out, float(rho), math.pi / n_out)])
                    if len(P) > 36:
                        continue
                    if ok_angle_vec(P, S_c).mean() >= 0.9999:
                        cands.append((route_len(P), len(P), d, float(R_in), n_out,
                                      float(rho), P))
    cands.sort(key=lambda t: t[0])
    print("粗筛可行 %d 个" % len(cands))

    S_f = sample_arena(10.0)
    keep = []
    for L, n_, d, R_in, n_out, rho, P in cands[:60]:
        if ok_angle_vec(P, S_f).mean() >= 0.99999:
            keep.append((L, n_, d, R_in, n_out, rho, P))
    print("细网格(10 m)复核通过 %d 个" % len(keep))
    print()
    print("%-46s %5s %9s %10s" % ("布局", "点数", "移动(s)", "路线(m)"))
    for L, n_, d, R_in, n_out, rho, P in keep[:8]:
        print("内层 hex d=%4.0f R=%4.0f + 外环 n=%2d rho=%4.0f      %3d %9.0f %10.0f"
              % (d, R_in, n_out, rho, n_, L / 5, L))

    if keep:
        b = keep[0]
        print()
        print("=" * 70)
        print("推荐布局（独立方法复核）")
        print("=" * 70)
        print("内层：六角点阵 step=%.0f m，extent=%.0f m（%d 点）"
              % (b[2], b[3], len(hex_pts(b[2], b[3]))))
        print("外环：半径 %.0f m，%d 点（相位 %.1f°）" % (b[5], b[4], 180.0 / b[4]))
        print("合计 %d 点，路线 %.0f m（移动 %.0f s）" % (b[1], b[0], b[0] / 5))
        r1 = ok_angle_vec(b[6], S_f).mean()
        r2 = ratio_tri(b[6], sample_arena(25.0))
        print("复核：极角法(10 m 网格) = %.5f，三角剖分法(25 m 网格) = %.5f" % (r1, r2))
        print("相对原方案（31 点 / 30732 m）：省 %.0f m = %.0f s"
              % (30732 - b[0], (30732 - b[0]) / 5))
        print()
        print("坐标：")
        print(np.round(b[6], 0).astype(int).tolist())
        # 与旧方案的每点对比
        print()
        print("新方案巡测点数 =", b[1], "（旧 31）")


if __name__ == "__main__":
    main()

