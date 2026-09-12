# -*- coding: utf-8 -*-
"""
B题 · 问题1/2 几何内核

统一约定
--------
* 坐标系：x 轴正东，y 轴正北，逆时针为正，方位角单位「度」，范围 [0,360)。
* 半平面统一写成 ``A·x >= b``，其中 ``A`` 为长度 2 的 numpy 数组、``b`` 为 float。
* 凸多边形统一用 **逆时针** 顶点列表 ``list[tuple[float,float]]`` 表示。

核心对象
--------
``wedge_halfplanes``  示向度楔形（顶点 S、中心方位 θ、半张角 δ）的凸锥半平面表示
``locate_region``     半平面裁剪法求交会定位区域（凸多边形）
``diameter_*``        凸多边形直径（顶点枚举 O(m^2) / 旋转卡壳 O(m)）
``mec``               最小包围圆（Welzl 迭代式，期望 O(n)）
``region_is_bounded`` 两楔形交是否有界（回收锥判据）
``covers_with_diameter_circle``  以直径为直径的圆能否覆盖多边形（Thales 判据）
``ArenaRaster``       目标区域栅格（问题3/4 的非凸可行域表示）
"""

from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------- 常量
D2R = math.pi / 180.0
R2D = 180.0 / math.pi

ARENA_R = 1800.0        # 目标区域半径 (m)
R_RECV_MIN = 1000.0     # 有效接收半径下界 (m)
R_RECV_MAX = 1500.0     # 有效接收半径上界 (m)
SV_DELTA = 1.0          # 示向度误差半幅 (度)
NEAR_R = 5.0            # 近距离阈值 (m)
CLEAR_R = 20.0          # 清除半径 (m)
DOG_SPEED = 5.0         # 机器狗速度 (m/s)
T_MEASURE = 5.0         # 检测动作耗时 (s)
T_SWITCH = 1.0          # 切频道耗时 (s)
T_CLEAR_OK = 5.0        # 精确定位 + 清除 (s)
T_CLEAR_FAIL = 3.0      # 仅精确定位 (s)

EPS = 1e-9

Point = Tuple[float, float]
HalfPlane = Tuple[np.ndarray, float]
Polygon = List[Point]


# ================================================================ 基础向量
def unit(alpha_deg: float) -> np.ndarray:
    """方位角(度) -> 单位方向向量 (x 正东, y 正北)"""
    a = alpha_deg * D2R
    return np.array([math.cos(a), math.sin(a)], dtype=float)


def bearing(p_from: Sequence[float], p_to: Sequence[float]) -> float:
    """从 p_from 指向 p_to 的方位角，返回 [0,360)"""
    return math.degrees(math.atan2(p_to[1] - p_from[1], p_to[0] - p_from[0])) % 360.0


def ang_diff(a: float, b: float) -> float:
    """两方位角的带符号差，归一化到 (-180, 180]"""
    return (a - b + 180.0) % 360.0 - 180.0


def ang_dist(a: float, b: float) -> float:
    """两方位角的圆周距离 [0,180]"""
    return abs(ang_diff(a, b))


def dist(p, q) -> float:
    return math.hypot(p[0] - q[0], p[1] - q[1])


def cross2(u, v) -> float:
    """二维叉积（numpy 2.0 起 np.cross 对二维向量已弃用）"""
    return float(u[0]) * float(v[1]) - float(u[1]) * float(v[0])


# ================================================================ 楔形
def wedge_halfplanes(S: Sequence[float], theta_deg: float,
                     delta: float = SV_DELTA) -> List[HalfPlane]:
    """示向度楔形的半平面表示：``W(S,θ,δ) = {S + t·u(α) : t ≥ 0, α ∈ [θ-δ, θ+δ]}``

    记 ``f(X;α) = (X-S) × u(α) = (x-Sx)·sinα - (y-Sy)·cosα``。
    对 ``X = S + t·u(β)``（t>0）有 ``f = t·sin(α-β)``，于是

        β ∈ [θ-δ, θ+δ]  ⟺  f(X; θ-δ) ≤ 0  且  f(X; θ+δ) ≥ 0

    返回两个 ``A·X >= b`` 形式的半平面。
    """
    Sx, Sy = float(S[0]), float(S[1])
    out: List[HalfPlane] = []
    for a, keep_le in ((theta_deg - delta, True), (theta_deg + delta, False)):
        ar = a * D2R
        s = 1.0 if keep_le else -1.0
        A = np.array([-s * math.sin(ar), s * math.cos(ar)], dtype=float)
        out.append((A, float(A[0] * Sx + A[1] * Sy)))
    return out


def in_wedge(X: Sequence[float], S: Sequence[float], theta_deg: float,
             delta: float = SV_DELTA, tol: float = 1e-9) -> bool:
    """点 X 是否落在楔形内（直接按半平面判据，供测试用）"""
    for A, b in wedge_halfplanes(S, theta_deg, delta):
        if float(A[0] * X[0] + A[1] * X[1]) - b < -tol:
            return False
    return True


# ================================================================ 多边形工具
def polygon_area(poly: Sequence[Point]) -> float:
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def polygon_centroid(poly: Sequence[Point]) -> np.ndarray:
    """面积质心；退化时退化为顶点算术平均"""
    n = len(poly)
    if n == 0:
        return np.zeros(2)
    a = polygon_area(poly)
    if abs(a) < 1e-12 or n < 3:
        return np.mean(np.asarray(poly, dtype=float), axis=0)
    cx = cy = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        cr = x1 * y2 - x2 * y1
        cx += (x1 + x2) * cr
        cy += (y1 + y2) * cr
    return np.array([cx / (6 * a), cy / (6 * a)])


def polygon_perimeter(poly: Sequence[Point]) -> float:
    n = len(poly)
    if n < 2:
        return 0.0
    return sum(dist(poly[i], poly[(i + 1) % n]) for i in range(n))


def ensure_ccw(poly: Sequence[Point]) -> Polygon:
    """保证凸多边形顶点逆时针排列（并去掉紧邻重复点）"""
    pts = dedup(poly)
    if len(pts) >= 3 and polygon_area(pts) < 0:
        pts = pts[::-1]
    return pts


def sort_ccw(points: Sequence[Point]) -> Polygon:
    """把**凸位置**上的散点按绕质心的极角排成逆时针环（解析求交后的必要步骤）"""
    pts = dedup(points)
    if len(pts) < 3:
        return pts
    c = np.mean(np.asarray(pts, float), axis=0)
    pts.sort(key=lambda p: math.atan2(p[1] - c[1], p[0] - c[0]))
    if polygon_area(pts) < 0:
        pts = pts[::-1]
    return pts


def dedup(poly: Iterable[Point], eps: float = 1e-7) -> Polygon:
    out: Polygon = []
    for p in poly:
        p = (float(p[0]), float(p[1]))
        if not out or dist(p, out[-1]) > eps:
            out.append(p)
    while len(out) > 1 and dist(out[0], out[-1]) <= eps:
        out.pop()
    return out


def is_convex_ccw(poly: Sequence[Point], tol: float = 1e-7) -> bool:
    n = len(poly)
    if n < 3:
        return True
    for i in range(n):
        a, b, c = (np.asarray(poly[i], float),
                   np.asarray(poly[(i + 1) % n], float),
                   np.asarray(poly[(i + 2) % n], float))
        if cross2(b - a, c - b) < -tol:
            return False
    return True


# ================================================================ 半平面裁剪
def clip_halfplane(poly: Sequence[Point], A: np.ndarray, b: float,
                   tol: float = 1e-9) -> Polygon:
    """Sutherland–Hodgman：保留 ``A·X >= b`` 的部分（输入输出均为凸多边形）"""
    n = len(poly)
    if n < 3:
        return []
    out: Polygon = []
    for i in range(n):
        P = np.asarray(poly[i], dtype=float)
        Q = np.asarray(poly[(i + 1) % n], dtype=float)
        fp = float(A @ P) - b
        fq = float(A @ Q) - b
        if fp >= -tol:
            out.append((P[0], P[1]))
        if (fp > tol and fq < -tol) or (fp < -tol and fq > tol):
            t = fp / (fp - fq)
            R = P + t * (Q - P)
            out.append((float(R[0]), float(R[1])))
    return dedup(out)


def clip_halfplanes(poly: Sequence[Point],
                    hs: Sequence[HalfPlane]) -> Polygon:
    for A, b in hs:
        poly = clip_halfplane(poly, A, b)
        if len(poly) < 3:
            return []
    return poly


def circle_polygon(radius: float, n: int = 180, circumscribe: bool = True) -> Polygon:
    """正 n 边形近似圆（逆时针）

    ``circumscribe=True`` 时取外切多边形（顶点在半径 ``radius/cos(π/n)`` 上），
    即**包含**整个圆盘；作为先验凸约束使用时可保证真值不被裁掉。
    """
    r = radius / math.cos(math.pi / n) if circumscribe else radius
    return [(r * math.cos(2 * math.pi * k / n),
             r * math.sin(2 * math.pi * k / n)) for k in range(n)]


def box_polygon(half: float = 20000.0) -> Polygon:
    return [(-half, -half), (half, -half), (half, half), (-half, half)]


# ================================================================ 定位区域
def locate_region(measures: Sequence[Tuple[float, float, float]],
                  arena: bool = True,
                  arena_radius: float = ARENA_R,
                  arena_sides: int = 180,
                  box_half: float = 20000.0,
                  initial_poly: Optional[Sequence[Point]] = None,
                  delta: float = SV_DELTA) -> Polygon:
    """交会定位区域（凸多边形，逆时针）

    Parameters
    ----------
    measures : [(Sx, Sy, theta_deg), ...]
        各检测点坐标与其测得的示向度（度）。
    arena : bool
        True 时用目标圆域（外切正 ``arena_sides`` 边形）做先验约束，
        这样结果**恒有界**；False 时只交楔形（可能无界，由大框截断）。
    initial_poly : 可选
        直接给定初始凸多边形（优先于 ``arena``/``box_half``），
        例如传入某个矩形以便把定位区域限制在已知范围内。
    """
    if initial_poly is not None:
        poly: Polygon = list(initial_poly)
    elif arena:
        poly = circle_polygon(arena_radius, arena_sides, circumscribe=True)
    else:
        poly = box_polygon(box_half)
    for (Sx, Sy, th) in measures:
        for A, b in wedge_halfplanes((Sx, Sy), th, delta):
            poly = clip_halfplane(poly, A, b)
            if len(poly) < 3:
                return []
    return ensure_ccw(poly)


def rect_polygon(xmin: float, xmax: float, ymin: float, ymax: float) -> Polygon:
    return [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]


def fast_region_vertices(measures: Sequence[Tuple[float, float, float]],
                         tol: float = 1e-7) -> Polygon:
    """仅由楔形半平面解析求交（O(k^2)，k=2·n_measures），不含圆域先验。

    用于选点扫描等需要百万次调用的场合。返回凸多边形顶点（可能 < 3 表示空）。
    """
    hs: List[HalfPlane] = []
    for (Sx, Sy, th) in measures:
        hs += wedge_halfplanes((Sx, Sy), th)
    n = len(hs)
    pts: Polygon = []
    for i in range(n):
        A1, b1 = hs[i]
        for j in range(i + 1, n):
            A2, b2 = hs[j]
            M = np.array([A1, A2])
            det = float(np.linalg.det(M))
            if abs(det) < 1e-12:
                continue
            p = np.linalg.solve(M, np.array([b1, b2]))
            if all(float(A @ p) - b >= -1e-6 for A, b in hs):
                if not any(dist(p, q) < tol for q in pts):
                    pts.append((float(p[0]), float(p[1])))
    return sort_ccw(pts)


def halfplane_set_bounded(hs: Sequence[HalfPlane], tol: float = 1e-12) -> bool:
    """半平面交 ``{x : A_i·x >= b_i}`` 是否有界。

    依据：$C=\\bigcap_i H_i$ 有界 $\\iff$ 其回收锥 $K=\\{d: A_i\\cdot d\\ge0\\ \\forall i\\}=\\{0\\}$；
    而 $K$ 的极射线必落在某条约束的边界上，即 $d\\parallel(-A_{i,2},A_{i,1})$。
    枚举这 $2k$ 个方向逐个检验即可（精确、$O(k^2)$）。
    """
    for A, _ in hs:
        for s in (1.0, -1.0):
            d = np.array([-A[1], A[0]], float) * s
            nrm = float(np.linalg.norm(d))
            if nrm < 1e-12:
                continue
            d = d / nrm
            if all(float(Aj @ d) >= -tol for Aj, _ in hs):
                return False
    return True


def region_is_bounded(measures: Sequence[Tuple[float, float, float]],
                      delta: float = SV_DELTA, method: str = "interval") -> bool:
    """楔形交是否有界。

    回收锥 $K=\\bigcap_i W(0,\\theta_i)=\\{t\\,u(\\alpha):t\\ge0,\\ \\alpha\\in\\bigcap_i I_i\\}$，
    故 **有界 $\\iff$ 各楔形方向区间的公共交集 $\\bigcap_i I_i=\\varnothing$**。
    注意：对 $n\\ge3$，**两两不交 $\\ne$ 公共交集为空**，必须求公共交集。

    method="interval"：按上述弧交集判据（解析）。
    method="rays"    ：对半平面集合枚举回收射线（与区间法互为独立校验）。
    """
    hs: List[HalfPlane] = []
    for (Sx, Sy, th) in measures:
        hs += wedge_halfplanes((Sx, Sy), th, delta)
    if method == "rays":
        return halfplane_set_bounded(hs)
    return not wedge_intervals_overlap(measures, delta)


def wedge_intervals_overlap(measures: Sequence[Tuple[float, float, float]],
                            delta: float = SV_DELTA) -> bool:
    """各楔形方向区间 $I_i=[\\theta_i-\\delta,\\ \\theta_i+\\delta]$ 的公共交集是否非空。

    圆上弧交集判据：交集非空 $\\iff$ 存在某个 $i$ 使 $I_i$ 的起点 $\\theta_i-\\delta$
    落在所有 $I_j$ 之内。（若交集非空，从交集中任一点沿顺时针走到某条弧的起点，该点仍在交集中。）

    对 $n=2$ 退化为 $d_{\\mathrm{circ}}(\\theta_1,\\theta_2)\\le2\\delta$。
    """
    ths = [m[2] for m in measures]
    n = len(ths)
    if n == 0:
        return False
    for i in range(n):
        p = (ths[i] - delta) % 360.0
        if all(ang_dist(p, ths[j]) <= delta + EPS for j in range(n)):
            return True
    return False


# ================================================================ 直径
def diameter_bruteforce(poly: Sequence[Point]) -> Tuple[float, Optional[Tuple[Point, Point]]]:
    """凸多边形直径：枚举全部顶点对 O(m^2)。凸性保证极值在顶点取得。"""
    m = len(poly)
    if m < 2:
        return 0.0, None
    best, pair = -1.0, None
    for i in range(m):
        xi, yi = poly[i]
        for j in range(i + 1, m):
            d = math.hypot(xi - poly[j][0], yi - poly[j][1])
            if d > best:
                best, pair = d, (poly[i], poly[j])
    return best, pair


def diameter_calipers(poly: Sequence[Point]) -> Tuple[float, Optional[Tuple[Point, Point]]]:
    """凸多边形直径：旋转卡壳 O(m)。``poly`` 必须为逆时针凸多边形。"""
    m = len(poly)
    if m < 2:
        return 0.0, None
    if m == 2:
        return dist(poly[0], poly[1]), (poly[0], poly[1])
    if not is_convex_ccw(poly):
        poly = convex_hull(poly)
        m = len(poly)
        if m < 3:
            return diameter_bruteforce(poly)
    P = [np.asarray(p, float) for p in poly]

    def area2(i: int, j: int, k: int) -> float:
        return abs(cross2(P[j] - P[i], P[k] - P[i]))

    best, pair = -1.0, None
    j = 1
    budget = 4 * m
    for i in range(m):
        ni = (i + 1) % m
        while budget > 0:
            nj = (j + 1) % m
            if area2(i, ni, nj) > area2(i, ni, j) + 1e-12:
                j = nj
                budget -= 1
            else:
                break
        for k in (i, ni):
            d = float(np.linalg.norm(P[k] - P[j]))
            if d > best:
                best, pair = d, (poly[k], poly[j])
    return best, pair


def convex_hull(points: Sequence[Point]) -> Polygon:
    """Andrew 单调链，返回逆时针凸包"""
    pts = sorted(set((float(p[0]), float(p[1])) for p in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: Polygon = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: Polygon = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def polygon_diameter(poly: Sequence[Point], method: str = "auto") -> float:
    """统一入口：返回直径长度"""
    d, _ = polygon_diameter_pair(poly, method)
    return d


def polygon_diameter_pair(poly: Sequence[Point], method: str = "auto"):
    if len(poly) < 3:
        return diameter_bruteforce(poly)
    if method == "brute" or (method == "auto" and len(poly) <= 16):
        return diameter_bruteforce(poly)
    return diameter_calipers(poly)


# ================================================================ 最小包围圆
def _circle_from2(p, q):
    c = (np.asarray(p, float) + np.asarray(q, float)) / 2.0
    return c, float(np.linalg.norm(np.asarray(p, float) - c))


def _circle_from3(a, b, c):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    c = np.asarray(c, float)
    d = 2.0 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-12:
        # 退化：取最远两点
        ds = [(np.linalg.norm(a - b), a, b), (np.linalg.norm(a - c), a, c),
              (np.linalg.norm(b - c), b, c)]
        ds.sort(key=lambda t: -t[0])
        return _circle_from2(ds[0][1], ds[0][2])
    ux = ((a @ a) * (b[1] - c[1]) + (b @ b) * (c[1] - a[1]) + (c @ c) * (a[1] - b[1])) / d
    uy = ((a @ a) * (c[0] - b[0]) + (b @ b) * (a[0] - c[0]) + (c @ c) * (b[0] - a[0])) / d
    ctr = np.array([ux, uy])
    return ctr, float(np.linalg.norm(a - ctr))


def mec(points: Sequence[Sequence[float]], seed: int = 20260913):
    """最小包围圆 (Minimum Enclosing Circle)，Welzl 迭代式，期望 O(n)。

    返回 ``(center, radius)``。``points`` 为空时返回 ``(None, 0.0)``。
    """
    P = [np.asarray(p, float) for p in points]
    if not P:
        return None, 0.0
    if len(P) == 1:
        return P[0], 0.0
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(P))
    P = [P[i] for i in idx]

    c = P[0].copy()
    r = 0.0
    for i in range(1, len(P)):
        if np.linalg.norm(P[i] - c) <= r + 1e-9:
            continue
        c, r = P[i].copy(), 0.0
        for j in range(i):
            if np.linalg.norm(P[j] - c) <= r + 1e-9:
                continue
            c, r = _circle_from2(P[i], P[j])
            for k in range(j):
                if np.linalg.norm(P[k] - c) <= r + 1e-9:
                    continue
                c, r = _circle_from3(P[i], P[j], P[k])
    return c, float(r)


def polygon_mec(poly: Sequence[Point], seed: int = 20260913):
    """凸多边形的 MEC 只由顶点决定"""
    return mec(poly, seed=seed)


# ================================================================ 覆盖判据
def covers_with_diameter_circle(poly: Sequence[Point], tol: float = 1e-9):
    """以「定位区域直径为直径的圆」能否覆盖该区域

    Thales 判据：设直径端点 A、B，点 P 在以 AB 为直径的闭圆内 ⟺ ``(P-A)·(P-B) ≤ 0``。

    Returns
    -------
    dict(covered, diameter, A, B, R_star, R_half, ratio, violators, max_dot)
    """
    res = {"covered": False, "diameter": 0.0, "A": None, "B": None,
           "R_star": 0.0, "R_half": 0.0, "ratio": float("inf"),
           "violators": [], "max_dot": 0.0}
    if len(poly) < 2:
        return res
    D, pair = polygon_diameter_pair(poly)
    A = np.asarray(pair[0], float)
    B = np.asarray(pair[1], float)
    viol, maxdot = [], -1e18
    for P in poly:
        X = np.asarray(P, float)
        dot = float((X - A) @ (X - B))
        maxdot = max(maxdot, dot)
        if dot > tol:
            viol.append((P[0], P[1], dot))
    R_half = D / 2.0
    ctr = (A + B) / 2.0
    R_star = max(float(np.linalg.norm(np.asarray(P, float) - ctr)) for P in poly)
    res.update(covered=(len(viol) == 0), diameter=D, A=tuple(A), B=tuple(B),
               R_star=R_star, R_half=R_half,
               ratio=(R_star / R_half if R_half > 0 else float("inf")),
               violators=viol, max_dot=maxdot)
    return res


# ================================================================ 栅格
class ArenaRaster:
    """目标区域内的方形栅格（外近似），用于问题3/4 的非凸可行域。

    栅格中心落在半径 ``radius`` 圆内的单元被保留。分辨率 ``cell`` 取 10 m 时
    若可行域塌缩到单个单元，则单元中心到真值 ≤ ``cell·√2/2`` = 7.07 m ≤ 20 m，
    此时 ``/clear`` 必然成功（提供可证明的停止判据）。
    """

    def __init__(self, cell: float = 10.0, radius: float = ARENA_R):
        self.cell = float(cell)
        self.radius = float(radius)
        n = int(math.floor(radius / cell))
        idx = np.arange(-n, n + 1, dtype=np.int32)
        IX, IY = np.meshgrid(idx, idx, indexing="ij")
        X = (IX.ravel() * np.float32(cell)).astype(np.float32)
        Y = (IY.ravel() * np.float32(cell)).astype(np.float32)
        keep = X * X + Y * Y <= np.float32(radius * radius + 1e-9)
        self.x = X[keep]
        self.y = Y[keep]
        self.n = int(keep.sum())
        self.mask = np.ones(self.n, dtype=bool)
        # 热点运算的免分配工作缓冲（float32，避免大数组反复分配触发内存上限）
        self._w1 = np.empty(self.n, dtype=np.float32)
        self._w2 = np.empty(self.n, dtype=np.float32)

    # ---- 内部：免分配距离平方 ----
    def _d2(self, center) -> np.ndarray:
        np.subtract(self.x, np.float32(center[0]), out=self._w1)
        np.multiply(self._w1, self._w1, out=self._w1)
        np.subtract(self.y, np.float32(center[1]), out=self._w2)
        np.multiply(self._w2, self._w2, out=self._w2)
        np.add(self._w1, self._w2, out=self._w1)
        return self._w1

    # ---- 基本算子 ----
    def reset(self):
        self.mask[:] = True

    def copy_mask(self) -> np.ndarray:
        return self.mask.copy()

    def set_mask(self, m: np.ndarray):
        self.mask = m.astype(bool).copy()

    def count(self) -> int:
        return int(self.mask.sum())

    @property
    def pad(self) -> float:
        """单元外接圆半径：外近似所需容差（保证真值所在单元不被误删）"""
        return self.cell * 0.7072 + 1e-9

    def extent(self):
        """当前可行点的包围盒 (xmin,xmax,ymin,ymax)，空时返回 None"""
        if self.count() == 0:
            return None
        x, y = self.x[self.mask], self.y[self.mask]
        return float(x.min()), float(x.max()), float(y.min()), float(y.max())

    def points(self) -> np.ndarray:
        return np.stack([self.x[self.mask], self.y[self.mask]], axis=1)

    def hull_points(self) -> np.ndarray:
        """可行点集的凸包顶点（MEC 只由凸包顶点决定，可把 1e5 级点集压到 1e3 级）"""
        pts = self.points()
        if len(pts) <= 64:
            return pts
        try:
            from scipy.spatial import ConvexHull
            h = ConvexHull(pts)
            return pts[np.unique(h.vertices)]
        except Exception:
            return pts

    def mec(self, seed: int = 20260913):
        """精确最小包围圆 (center, radius)。空集返回 (None, 0.0)。"""
        pts = self.hull_points()
        if len(pts) == 0:
            return None, 0.0
        return mec(pts, seed=seed)

    def max_dist_to(self, p) -> float:
        """可行域内所有格点到 p 的最大距离（严谨的"能否一次 clear 成功"判据）"""
        if self.count() == 0:
            return float("inf")
        d2 = (self.x[self.mask] - p[0]) ** 2 + (self.y[self.mask] - p[1]) ** 2
        return float(np.sqrt(d2.max()))

    def best_clear_point(self, radius: float = CLEAR_R, mask: Optional[np.ndarray] = None):
        """找一个点，使可行域**任意点**（含真值）都落在其 ``radius`` 邻域内。

        Returns ``(ok, center, maxdist)``。``ok=True`` 时到 ``center`` 执行
        ``/clear`` **必然成功**：栅格为外近似 ⟹ 真值必落在某个可行单元内，
        而任何可行点到 ``center`` 的距离都 ``≤ radius``。
        """
        m = self.mask if mask is None else mask
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            return False, None, float("inf")
        x, y = self.x[idx], self.y[idx]
        pts = np.stack([x, y], axis=1)
        c, r = mec(pts)
        if c is None:
            return False, None, float("inf")
        md = float(np.sqrt(((x - c[0]) ** 2 + (y - c[1]) ** 2).max()))
        # 真值落在某单元内、与该单元中心相距不超过 pad，故阈值要扣掉 pad
        if md <= radius - self.pad + 1e-9:
            return True, (float(c[0]), float(c[1])), md
        return False, (float(c[0]), float(c[1])), md

    # ---- 外近似算子（全部带 pad，保证"真值不丢"）----
    def keep_ball(self, center, radius: float, mask: Optional[np.ndarray] = None):
        """保留 ``|X-center| <= radius`` 的**外近似**（阈值放宽 ``pad``）"""
        lim = np.float32((radius + self.pad) ** 2)
        keep = self._d2(center) <= lim
        return keep if mask is None else (mask & keep)

    def keep_outside_ball(self, center, radius: float,
                          mask: Optional[np.ndarray] = None):
        """保留"可能有部分落在球外"的单元：仅当单元整体落入球内才剔除。

        用于 ``no_signal``（源必在接收半径外）与 ``/clear`` 未发现（源必在 20 m 外）。
        """
        lim = np.float32(max(0.0, radius - self.pad) ** 2)
        keep = self._d2(center) > lim
        return keep if mask is None else (mask & keep)

    def keep_wedge(self, S, theta_deg: float, delta: float = SV_DELTA,
                   tol: float = 1e-9, mask: Optional[np.ndarray] = None):
        """落在示向度楔形内的外近似掩码（外扩一个单元外接圆半径）"""
        pad = self.pad + tol
        m = np.ones(self.n, dtype=bool)
        for A, b in wedge_halfplanes(S, theta_deg, delta):
            a0, a1 = np.float32(A[0]), np.float32(A[1])
            np.multiply(self.x, a0, out=self._w1)
            np.multiply(self.y, a1, out=self._w2)
            np.add(self._w1, self._w2, out=self._w1)
            m &= self._w1 >= np.float32(b - pad * float(np.hypot(A[0], A[1])))
        return m if mask is None else (mask & m)

    def restrict(self, mask: np.ndarray, keep: np.ndarray) -> np.ndarray:
        return mask & keep

    def keep_halfplane(self, A: np.ndarray, b: float, pad: float = 0.0,
                       mask: Optional[np.ndarray] = None):
        a0, a1 = np.float32(A[0]), np.float32(A[1])
        np.multiply(self.x, a0, out=self._w1)
        np.multiply(self.y, a1, out=self._w2)
        np.add(self._w1, self._w2, out=self._w1)
        keep = self._w1 >= np.float32(b - pad * float(np.hypot(A[0], A[1])))
        return keep if mask is None else (mask & keep)

    def interior_point(self) -> Optional[Point]:
        """当前可行域的一个代表点（质心最近的格心）"""
        if self.count() == 0:
            return None
        x, y = self.x[self.mask], self.y[self.mask]
        cx, cy = float(x.mean()), float(y.mean())
        k = int(np.argmin((x - cx) ** 2 + (y - cy) ** 2))
        return float(x[k]), float(y[k])


# ---- 共享栅格实例（坐标数组只读，多局共用可显著降低内存与构造开销）----
_RASTER_CACHE: dict = {}


def get_raster(cell: float = 10.0, radius: float = ARENA_R) -> "ArenaRaster":
    key = (round(float(cell), 6), round(float(radius), 6))
    r = _RASTER_CACHE.get(key)
    if r is None:
        r = ArenaRaster(cell=cell, radius=radius)
        _RASTER_CACHE[key] = r
    return r


# ================================================================ 时间模型
def move_time(p, q, speed: float = DOG_SPEED) -> float:
    return dist(p, q) / speed

def measure_cost(p_prev, p_new, ch_prev, ch_new,
                 speed: float = DOG_SPEED) -> float:
    """一次合法 /measure 的虚拟耗时 = 移动 + 切频道 + 检测"""
    t = move_time(p_prev, p_new, speed) + T_MEASURE
    if ch_prev is not None and ch_prev != ch_new:
        t += T_SWITCH
    return t


def clear_cost(p_prev, p_new, success: bool, speed: float = DOG_SPEED) -> float:
    """一次合法 /clear 的虚拟耗时 = 移动 + (精确定位 [+清除])；不切频道"""
    return move_time(p_prev, p_new, speed) + (T_CLEAR_OK if success else T_CLEAR_FAIL)


def scan_cost(n_channels: int, n_switch: Optional[int] = None) -> float:
    """单点扫描若干频道的虚拟耗时（升序扫描时 switch = n_channels - 1 + 跨点额外 1）"""
    if n_switch is None:
        n_switch = max(0, n_channels - 1)
    return n_channels * T_MEASURE + n_switch * T_SWITCH
