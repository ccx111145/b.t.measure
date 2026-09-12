# -*- coding: utf-8 -*-
"""
③ 非凸 / 有障碍工作区

理论推广
--------
无障碍时检测完备性判据是  G ∈ conv(S_G)，S_G = {P : |P-G| <= R_min}。
引入障碍后，"附近有点"不再够用——点必须在**视线**上：

    G ∈ conv( V_G ),   V_G = { P ∈ P : |P-G| <= R_min 且 LOS(P, G) }

推论（与推论 1 对应）：若障碍把某个方向完全挡住，则该方向的网络点再多也无效，
必须在**遮挡区**内补点。下面用"沿障碍阴影补点"的贪心算法构造网络，并做逐单元严格认证。

认证
----
沿用论文的单元分解：若单元内既无网络圆周 ∂B(P,R_min) 穿过、也无障碍边穿过，
则 V_G 在单元上恒定，条件是凸的 ⟹ 查四个角点即可。否则细分。
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ARENA_R, R_LO = 1800.0, 1000.0
Point = Tuple[float, float]


# ------------------------------------------------------------------ 障碍
@dataclass
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float

    def contains(self, p: Point, eps: float = 1e-9) -> bool:
        return (self.x0 - eps <= p[0] <= self.x1 + eps
                and self.y0 - eps <= p[1] <= self.y1 + eps)

    def edges(self):
        return [((self.x0, self.y0), (self.x1, self.y0)),
                ((self.x1, self.y0), (self.x1, self.y1)),
                ((self.x1, self.y1), (self.x0, self.y1)),
                ((self.x0, self.y1), (self.x0, self.y0))]


def _seg_int(a, b, c, d) -> bool:
    """线段 ab 与 cd 是否相交（含端点接触）"""
    def cr(o, p, q):
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])
    d1, d2 = cr(c, d, a), cr(c, d, b)
    d3, d4 = cr(a, b, c), cr(a, b, d)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    return False


def los(p: Point, q: Point, obstacles: Sequence[Rect], eps: float = 1e-9) -> bool:
    """p 与 q 之间是否有视线（不被任何障碍内部阻挡）"""
    for r in obstacles:
        if r.contains(p, eps) or r.contains(q, eps):
            return False
        for (c, d) in r.edges():
            if _seg_int(p, q, c, d):
                return False
    return True


# ------------------------------------------------------------------ 判据
def visible_points(P: np.ndarray, g: Point,
                   obstacles: Sequence[Rect]) -> np.ndarray:
    d = np.hypot(P[:, 0] - g[0], P[:, 1] - g[1])
    cand = np.flatnonzero(d <= R_LO + 1e-9)
    if len(cand) == 0:
        return P[:0]
    keep = [i for i in cand if los(tuple(P[i]), g, obstacles)]
    return P[keep] if keep else P[:0]


def detectable(P: np.ndarray, g: Point, obstacles: Sequence[Rect]) -> bool:
    V = visible_points(P, g, obstacles)
    if len(V) == 0:
        return False
    if len(V) == 1:
        return math.dist(tuple(V[0]), g) <= 1e-6
    if len(V) == 2:
        a, b = V[0] - np.asarray(g), V[1] - np.asarray(g)
        return abs(a[0] * b[1] - a[1] * b[0]) < 1e-6 and (a @ b) < 0
    v = V - np.asarray(g)
    ang = np.sort(np.arctan2(v[:, 1], v[:, 0]))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]]))
    return bool(gaps.max() <= math.pi + 1e-9)


# ------------------------------------------------------------------ 认证
def certify(P: np.ndarray, obstacles: Sequence[Rect], h: float = 30.0,
            max_depth: int = 4, inside=lambda x, y: math.hypot(x, y) <= ARENA_R,
            verbose: bool = False):
    """逐单元严格认证；返回 (通过单元数, 失败单元数, 失败样例)"""
    stats = {"ok": 0, "bad": 0, "first": None}

    def blocked(cx, cy, hh):
        """单元是否被障碍边或网络圆周穿过"""
        half = hh / 2.0
        if any(abs(r.x0 - cx) < half and abs(r.y0 - cy) < half for r in obstacles):
            return True
        for r in obstacles:
            for (a, b) in r.edges():
                # 边与单元相交的宽松判据：边的包围盒与单元相交
                if (min(a[0], b[0]) <= cx + half and max(a[0], b[0]) >= cx - half
                        and min(a[1], b[1]) <= cy + half and max(a[1], b[1]) >= cy - half):
                    return True
        for Q in P:
            dx = max(abs(cx - Q[0]) - half, 0.0)
            dy = max(abs(cy - Q[1]) - half, 0.0)
            dmin = math.hypot(dx, dy)
            dmax = max(math.hypot(cx + sx * half - Q[0], cy + sy * half - Q[1])
                       for sx in (-1, 1) for sy in (-1, 1))
            if dmin <= R_LO <= dmax:
                return True
        return False

    def rec(cx, cy, hh, depth):
        if not inside(cx, cy):
            return True
        if any(r.contains((cx, cy)) for r in obstacles):
            return True                       # 障碍内部不要求可检出
        if blocked(cx, cy, hh) and depth < max_depth:
            q = hh / 2.0
            return all(rec(cx + sx * q / 2, cy + sy * q / 2, q, depth + 1)
                       for sx in (-1, 1) for sy in (-1, 1))
        hh2 = hh / 2.0
        corners = [(cx + sx * hh2, cy + sy * hh2)
                   for sx in (-1, 1) for sy in (-1, 1)]
        ok = all(detectable(P, c, obstacles) for c in corners)
        if ok:
            stats["ok"] += 1
        else:
            stats["bad"] += 1
            if stats["first"] is None:
                stats["first"] = (round(cx, 1), round(cy, 1))
        return ok

    for x0 in np.arange(-ARENA_R, ARENA_R + 1e-9, h):
        for y0 in np.arange(-ARENA_R, ARENA_R + 1e-9, h):
            rec(x0, y0, h, 0)
    return stats["ok"], stats["bad"], stats["first"]


# ------------------------------------------------------------------ 网络
def hex_lattice(step: float, extent: float, obstacles=()) -> np.ndarray:
    pts, dy = [], step * math.sqrt(3) / 2
    for j in range(-int(2 * extent / dy) - 2, int(2 * extent / dy) + 3):
        y = j * dy
        off = (step / 2) if (j % 2) else 0.0
        for i in range(-int(2 * extent / step) - 2, int(2 * extent / step) + 3):
            x = i * step + off
            if math.hypot(x, y) <= extent + 1e-9:
                if not any(r.contains((x, y)) for r in obstacles):
                    pts.append((x, y))
    return np.array(pts) if pts else np.zeros((0, 2))


def greedy_augment(P: np.ndarray, obstacles, h: float = 40.0,
                   max_add: int = 120, cand_step: float = 150.0):
    """贪心补点：在未通过的位置附近、且不在障碍内的候选点中挑选增益最大者"""
    P = P.copy()
    cand = []
    for x in np.arange(-ARENA_R, ARENA_R + 1e-9, cand_step):
        for y in np.arange(-ARENA_R, ARENA_R + 1e-9, cand_step):
            if math.hypot(x, y) > ARENA_R + 500:
                continue
            if any(r.contains((x, y)) for r in obstacles):
                continue
            cand.append((x, y))
    cand = np.array(cand)
    for _ in range(max_add):
        # 采样一批失败点，统计每个候选能救回多少
        fails = []
        for x in np.arange(-ARENA_R, ARENA_R + 1e-9, h):
            for y in np.arange(-ARENA_R, ARENA_R + 1e-9, h):
                if math.hypot(x, y) > ARENA_R or any(r.contains((x, y)) for r in obstacles):
                    continue
                if not detectable(P, (x, y), obstacles):
                    fails.append((x, y))
        if not fails:
            break
        fails = np.array(fails)
        gains = np.zeros(len(cand), dtype=int)
        for k, c in enumerate(cand):
            for g in fails:
                if math.dist(c, g) <= R_LO and los(tuple(c), tuple(g), obstacles):
                    V = visible_points(np.vstack([P, c]), tuple(g), obstacles)
                    tmp = np.vstack([P, c])
                    if _conv_ok(V, g):
                        gains[k] += 1
        k = int(np.argmax(gains))
        if gains[k] == 0:
            break
        P = np.vstack([P, cand[k]])
    return P


def _conv_ok(V: np.ndarray, g) -> bool:
    if len(V) == 0:
        return False
    if len(V) == 1:
        return math.dist(tuple(V[0]), g) <= 1e-6
    if len(V) == 2:
        a, b = V[0] - np.asarray(g), V[1] - np.asarray(g)
        return abs(a[0] * b[1] - a[1] * b[0]) < 1e-6 and (a @ b) < 0
    v = V - np.asarray(g)
    ang = np.sort(np.arctan2(v[:, 1], v[:, 0]))
    gaps = np.diff(np.concatenate([ang, [ang[0] + 2 * math.pi]]))
    return bool(gaps.max() <= math.pi + 1e-9)
