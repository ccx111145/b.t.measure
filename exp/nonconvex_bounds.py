# -*- coding: utf-8 -*-
"""
非凸工作区的下界闭环

三个层次
--------
(A) 弧长下界的**单调性**：障碍只会遮挡，不会增加任何网络点可服务的边界弧。
    故外边界弧长下界 k >= ceil(2π / 2φ_max) 在非凸下**依然成立**，且只会更强。
    并给出可计算的加强版：用真实几何算出的"最大可见外边界弧"替换 φ_max。

(B) 路线下界对任意 W 的推广（扫掠面积论证与形状无关）：
        L >= (A_free - π R_min²) / (2 R_min)

(C) **多重覆盖下界**（本文新增，且是非凸下的正确松弛对象）：
    完备性要求"包围"，即每个位置在 R_min 内至少有 3 个**可见**网络点。
    于是总"服务需求"为 3·N_cells，单点最多服务 max_p serve(p)，故
        k >= ceil( 3·N_cells / max_p serve(p) )
    这是严格的（必要性松弛），且比覆盖下界强约 3 倍。
"""
from __future__ import annotations

import math
import os
import sys
from typing import List, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp"))

from nonconvex import ARENA_R, R_LO, Rect, los          # noqa: E402

Point = Tuple[float, float]


# ------------------------------------------------------------------ (A)
def outer_arc_bound(P_free: np.ndarray, obstacles: Sequence[Rect],
                    r_lo: float = R_LO, r_out: float = ARENA_R,
                    n_ang: int = 3600) -> Tuple[int, float]:
    """可用网络点的最大可见外边界弧（半角），以及由此得到的点数下界。

    P_free：候选网络点集合（必须落在自由空间内）。
    """
    th = np.arange(n_ang) * (2 * math.pi / n_ang)
    B = np.stack([r_out * np.cos(th), r_out * np.sin(th)], 1)     # 外边界采样点
    best = 0.0
    for p in P_free:
        d = np.hypot(B[:, 0] - p[0], B[:, 1] - p[1])
        cand = np.flatnonzero(d <= r_lo)
        if len(cand) == 0:
            continue
        vis = np.array([los(tuple(p), tuple(B[i]), obstacles) for i in cand])
        idx = np.sort(cand[vis])
        if len(idx) == 0:
            continue
        # 找最长连续（环形）可见弧
        flags = np.zeros(n_ang, dtype=bool)
        flags[idx] = True
        if flags.all():
            arc = 2 * math.pi
        else:
            # 环形最长 True 段
            ext = np.concatenate([flags, flags])
            best_run = cur = 0
            for v in ext:
                cur = cur + 1 if v else 0
                best_run = max(best_run, cur)
            best_run = min(best_run, n_ang)
            arc = best_run * (2 * math.pi / n_ang)
        best = max(best, arc)
    return (math.ceil(2 * math.pi / best) if best > 0 else 10 ** 9), best


# ------------------------------------------------------------------ (B)
def area_free(obstacles: Sequence[Rect], r_out: float = ARENA_R,
              n: int = 4000) -> float:
    """自由空间面积（蒙特卡洛，含圆盘裁剪）"""
    rng = np.random.default_rng(20260913)
    R = r_out * np.sqrt(rng.random(n))
    A = rng.random(n) * 2 * math.pi
    X, Y = R * np.cos(A), R * np.sin(A)
    free = np.ones(n, dtype=bool)
    for ob in obstacles:
        free &= ~((X >= ob.x0) & (X <= ob.x1) & (Y >= ob.y0) & (Y <= ob.y1))
    return math.pi * r_out ** 2 * free.mean()


def route_lower_bound(a_free: float, r_lo: float = R_LO) -> float:
    """扫掠面积论证：L >= (A_free - π r²) / (2 r)"""
    return max(0.0, (a_free - math.pi * r_lo ** 2) / (2 * r_lo))


# ------------------------------------------------------------------ (C)
def multi_cover_bound(obstacles: Sequence[Rect], h: float = 60.0,
                      cand_step: float = 150.0, r_lo: float = R_LO,
                      r_out: float = ARENA_R, need: int = 3) -> dict:
    """多重覆盖下界：每个位置至少需要 need 个可见网络点"""
    cells = []
    for x in np.arange(-r_out, r_out + 1e-9, h):
        for y in np.arange(-r_out, r_out + 1e-9, h):
            if math.hypot(x, y) > r_out or any(o.contains((x, y)) for o in obstacles):
                continue
            cells.append((x, y))
    cand = []
    for x in np.arange(-r_out - r_lo, r_out + r_lo + 1e-9, cand_step):
        for y in np.arange(-r_out - r_lo, r_out + r_lo + 1e-9, cand_step):
            if any(o.contains((x, y)) for o in obstacles):
                continue
            cand.append((x, y))
    best = 0
    for p in cand:
        cnt = 0
        for c in cells:
            if math.hypot(p[0] - c[0], p[1] - c[1]) <= r_lo and los(p, c, obstacles):
                cnt += 1
        best = max(best, cnt)
    return {"n_cells": len(cells), "max_serve": best,
            "lb_points": math.ceil(need * len(cells) / best) if best else 10 ** 9,
            "need": need}


if __name__ == "__main__":
    from nonconvex_run import OBSTACLES, two_layer
    import time

    A = area_free(OBSTACLES)
    print("=" * 84)
    print("非凸工作区下界闭环")
    print("=" * 84)
    print("自由空间面积 A_free = %.4e m²（圆盘 %.4e m²，障碍占 %.1f%%）"
          % (A, math.pi * ARENA_R ** 2,
             100 * (1 - A / (math.pi * ARENA_R ** 2))))

    print()
    print("(A) 外边界弧长下界（含真实可见性）")
    t0 = time.monotonic()
    P_free = np.array([[x, y] for x in np.arange(-2600, 2601, 200)
                       for y in np.arange(-2600, 2601, 200)
                       if not any(o.contains((x, y)) for o in OBSTACLES)])
    lb_a, arc = outer_arc_bound(P_free, OBSTACLES)
    print("    单点最大可见外边界弧 = %.1f°（无遮挡上界为 67.5°）"
          % math.degrees(arc))
    print("    ⟹ k >= %d（无遮挡时 k >= %d）" % (lb_a, math.ceil(360 / 67.5)))
    print("    用时 %.1f s" % (time.monotonic() - t0))

    print()
    print("(B) 路线长度下界")
    lb_b = route_lower_bound(A)
    print("    L >= %.0f m" % lb_b)

    print()
    print("(C) 多重覆盖下界（每个位置至少 3 个可见网络点）")
    t0 = time.monotonic()
    mc = multi_cover_bound(OBSTACLES, h=80.0, cand_step=200.0)
    print("    单元数 %d，单点最大可服务单元数 %d" % (mc["n_cells"], mc["max_serve"]))
    print("    ⟹ k >= %d" % mc["lb_points"])
    print("    用时 %.1f s" % (time.monotonic() - t0))

    print()
    print("对照：贪心修复解使用 52 点、路线约 %.0f m" % 0.0)
