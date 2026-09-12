# -*- coding: utf-8 -*-
"""
B题 · 问题2：第二检测点选择策略

模型
----
第一次测量给出 $S_1$ 与示向度 $\\theta_1$ 后，干扰源的可行域为

    F1 = 目标圆域 A  ∩  楔形 W(S1, θ1, δ)  ∩  圆盘 B(S1, R_recv_max)

**目标**（极小极大）：选 $S_2$，使 $F_1$ 内所有可能位置、所有 $\\pm\\delta$ 误差组合下
定位区域直径的最大值最小。

**硬约束**（最易被忽略）：第二点必须仍能测到信号，保守取

    max_{G in F1} |S2 - G| <= R_lo        (R_lo = 有效接收半径下界 = 1000 m)

令 $S_2=S_1+d\\,u(\\theta_1+\\psi)$，$F_1$ 最远点约 $S_1+R_{\\max}u(\\theta_1)$，则

    d^2 + R_max^2 - 2 d R_max cos ψ <= R_lo^2

给出闭式的**候选区域**（环形扇区）与数值最优解。

对外接口
--------
``precision_threshold``   40 m 精度阈值表（交会角 vs 最大作用距离）
``worst_diam_analytic``   定位区域直径解析上界 2δ(r1+r2)/sinγ
``F1Set``                 源可行域采样集
``evaluate``              单点评价
``scan``                  (ψ, d) 网格扫描
``candidate_region``      候选区域的解析刻画与多边形表示
``analytic_psi_max``      可检测性约束给出的 |ψ| 上界
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry as G

INF = float("inf")


# ================================================================ 解析结论
#
# 记 S2 = S1 + d·u(θ1+ψ)，G = S1 + r·u(θ1+β)，r∈[0,R_max]，β∈[−δ,δ]（暂略去圆域先验）：
#     |S2−G|² = d² + r² − 2 d r cos(ψ−β)
# 对固定的 β，关于 r 的凸二次函数在 r=R_max 取最大；再对 β 取使 cos(ψ−β) 最小的端点，
# 即 |ψ−β| 最大 = |ψ|+δ。于是
#     max_{G∈F1} |S2−G|² = d² + R_max² − 2 d R_max cos(|ψ|+δ)
# 加上 r→0 的退化情形 |S2−S1| = d，可检测性硬约束为
#     (C1) d ≤ R_lo
#     (C2) d² − 2 R_max cos(|ψ|+δ)·d + (R_max² − R_lo²) ≤ 0
# (C2) 有解要求 cos(|ψ|+δ) ≥ √(1−(R_lo/R_max)²)，即
#     |ψ| ≤ arcsin(R_lo/R_max) − δ
# ─────────────────────────────────────────────────────────────────

def _cos_theta(psi_deg: float, delta_deg: float = G.SV_DELTA) -> float:
    """θ = |ψ| + δ，取余弦"""
    th = min(abs(psi_deg) + delta_deg, 180.0)
    return math.cos(th * G.D2R)


def psi_max_upper(R_lo: float = G.R_RECV_MIN,
                  R_max: float = G.R_RECV_MAX) -> float:
    """$|\\psi|$ 的**闭式上界**（必要条件）

    $$|\\psi|\\ \\le\\ \\arcsin\\!\\Big(\\frac{R_{lo}}{R_{max}}\\Big)-\\delta$$

    来自远支二次不等式判别式非负：$\\cos^2\\theta\\ge1-(R_{lo}/R_{max})^2$。
    由于近支（$r=r_{\\min}$）还会再收紧一点，这不是精确边界。
    """
    if R_lo >= R_max:
        return 90.0 - G.SV_DELTA
    return max(0.0, math.degrees(math.asin(R_lo / R_max)) - G.SV_DELTA)


def analytic_psi_max(R_lo: float = G.R_RECV_MIN,
                     R_max: float = G.R_RECV_MAX,
                     r_min: float = G.NEAR_R,
                     tol: float = 1e-10) -> float:
    """可检测性硬约束允许的最大 $|\\psi|$（**精确值**）

    在闭式上界 $\\psi_{\\sup}$ 内二分：可行 $\\iff$ 远支与近支两个二次不等式有公共解，
    即 $a_1(\\theta)\\le b_2(\\theta)$，其中
    $a_1=R_{max}\\cos\\theta-\\sqrt{R_{max}^2\\cos^2\\theta-(R_{max}^2-R_{lo}^2)}$,
    $b_2=r_{\\min}\\cos\\theta+\\sqrt{R_{lo}^2-r_{\\min}^2\\sin^2\\theta}$，$\\theta=|\\psi|+\\delta$。
    """
    hi = psi_max_upper(R_lo, R_max)
    if not math.isnan(analytic_d_feasible_range(hi, R_lo, R_max, r_min)[0]):
        return hi
    lo = 0.0
    if math.isnan(analytic_d_feasible_range(0.0, R_lo, R_max, r_min)[0]):
        return 0.0
    while hi - lo > tol:
        mid = (lo + hi) / 2.0
        if math.isnan(analytic_d_feasible_range(mid, R_lo, R_max, r_min)[0]):
            hi = mid
        else:
            lo = mid
    return lo


def analytic_d_feasible_range(psi_deg: float,
                              R_lo: float = G.R_RECV_MIN,
                              R_max: float = G.R_RECV_MAX,
                              r_min: float = G.NEAR_R) -> Tuple[float, float]:
    """给定 ψ，使可检测性硬约束成立的 d 区间 `[d1, d2]`

    $\\max_{G}|S_2-G|^2$ 关于 $r$ 是凸函数，故极值只在 $r\\in\\{r_{\\min},R_{max}\\}$ 取得；
    两支各给一个二次不等式，求交即得：

    * 远支 $d^2-2R_{max}\\cos\\theta\\,d+(R_{max}^2-R_{lo}^2)\\le0$
    * 近支 $d^2-2r_{\\min}\\cos\\theta\\,d+(r_{\\min}^2-R_{lo}^2)\\le0$

    其中 $\\theta=|\\psi|+\\delta$。无解返回 ``(nan, nan)``。
    """
    c = _cos_theta(psi_deg)
    lo, hi = 0.0, INF

    def _tighten(rad: float):
        nonlocal lo, hi
        disc = rad * rad * c * c - (rad * rad - R_lo * R_lo)
        if disc < 0:
            return False
        s = math.sqrt(disc)
        lo = max(lo, rad * c - s)
        hi = min(hi, rad * c + s)
        return True

    if not _tighten(R_max):
        return (float("nan"), float("nan"))
    if not _tighten(r_min):
        return (float("nan"), float("nan"))
    if hi <= lo or hi <= 0:
        return (float("nan"), float("nan"))
    return (max(lo, 0.0), hi)


def analytic_max_dist(d: float, psi_deg: float,
                      R_max: float = G.R_RECV_MAX,
                      r_min: float = G.NEAR_R) -> float:
    """$\\max_{G\\in F_1}|S_2-G|$ 的**精确**解析值（源距离限制在 $[r_{\\min},R_{max}]$）

    略去目标圆域先验，故数值上略大于真实值（保守）。
    """
    c = _cos_theta(psi_deg)

    def f(r):
        return d * d + r * r - 2.0 * d * r * c

    return math.sqrt(max(0.0, f(r_min), f(R_max)))


def analytic_diam_coeff(R_lo: float = G.R_RECV_MIN,
                        R_max: float = G.R_RECV_MAX) -> Dict[str, float]:
    """推荐解摘要：$\\psi^*$ 取可行上界的 0.95 倍附近，$d^*=R_{lo}$（见 §4.4 数值最优）"""
    pm = analytic_psi_max(R_lo, R_max)
    return {"psi_max_deg": pm, "d_star_m": R_lo,
            "min_d_at_psi_max_m": analytic_d_feasible_range(pm * 0.98, R_lo, R_max)[0]}


def worst_diam_analytic(r1: float, r2: float, gamma_deg: float,
                        delta_deg: float = G.SV_DELTA) -> float:
    """定位区域直径的解析上界 $2\\delta(r_1+r_2)/\\sin\\gamma$（$\\delta,\\gamma$ 用度输入）

    数值上比真实直径高约 10%，可作保守估计。
    """
    s = math.sin(gamma_deg * G.D2R)
    if s <= 1e-12:
        return INF
    return 2.0 * (delta_deg * G.D2R) * (r1 + r2) / s


def precision_threshold(gamma_deg: float, target: float = 2 * G.CLEAR_R,
                        delta_deg: float = G.SV_DELTA) -> float:
    """两点交会区域直径 ≤ ``target`` 时允许的最大作用距离 r（$r_1\\approx r_2\\approx r$）

    由 $2\\delta\\cdot 2r/\\sin\\gamma\\le target$ 得 $r\\le \\dfrac{target\\sin\\gamma}{4\\delta}$。
    与"直径 ≈ 2·tanδ·r/sinγ"的几何式同阶；本文档采用 $r\\le \\dfrac{(target/2)\\sin\\gamma}{\\tan\\delta}$
    （单站横向误差 $r\\tan\\delta$ 的量级），二者相差常数因子，论文中给数值表即可。
    """
    return (target / 2.0) * math.sin(gamma_deg * G.D2R) / math.tan(delta_deg * G.D2R)


def precision_table(gammas: Sequence[float] = (30, 45, 60, 75, 90),
                    target: float = 2 * G.CLEAR_R) -> List[Dict[str, float]]:
    out = []
    for g in gammas:
        out.append({"gamma_deg": float(g),
                    "r_max_m": precision_threshold(g, target),
                    "dia_m_per_r": 2 * math.tan(G.SV_DELTA * G.D2R) / math.sin(g * G.D2R)})
    return out


def linearized_diam_coeff(gamma_deg: float, delta_deg: float = G.SV_DELTA) -> float:
    """垂直基线构型下"直径 / 真值距离"的经验系数（数值实验拟合，见 fig_q1）"""
    # 由 2δ(r1+r2)/sinγ 在 r1=r, r2=r√2, γ=45° 处标定，再乘 0.9 的经验修正
    r1, r2 = 1.0, math.sqrt(2.0)
    return 0.9 * worst_diam_analytic(r1, r2, gamma_deg, delta_deg)


# ================================================================ F1 可行域
@dataclass
class F1Set:
    """第一次测量后的源可行域采样集

    Attributes
    ----------
    fine_x, fine_y : 精细采样（用于"可检测性/最远距离"这类需要精确极值的判定）
    coarse_x, coarse_y : 粗采样（用于最坏直径扫描，降低计算量的同时保持极值覆盖）
    """

    S1: Tuple[float, float]
    theta1: float
    r_max: float = G.R_RECV_MAX
    r_min: float = G.NEAR_R
    arena_r: float = G.ARENA_R
    delta: float = G.SV_DELTA
    fine_x: np.ndarray = field(default=None, repr=False)
    fine_y: np.ndarray = field(default=None, repr=False)
    coarse_x: np.ndarray = field(default=None, repr=False)
    coarse_y: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        self._build()

    def _sample(self, dr: float, db: float):
        r = np.arange(self.r_min, self.r_max + 1e-9, dr)
        b = np.arange(-self.delta, self.delta + 1e-9, db)
        RR, BB = np.meshgrid(r, b, indexing="ij")
        ang = (self.theta1 + BB) * G.D2R
        x = self.S1[0] + RR * np.cos(ang)
        y = self.S1[1] + RR * np.sin(ang)
        x, y = x.ravel(), y.ravel()
        keep = x * x + y * y <= self.arena_r ** 2 + 1e-9
        return x[keep], y[keep]

    def _build(self):
        self.fine_x, self.fine_y = self._sample(max(5.0, self.r_min), 0.25)
        self.coarse_x, self.coarse_y = self._sample(250.0, 0.5)
        # 保证最远点一定被采到：把 r_max 处的样本补进粗采样
        extra_ang = np.arange(-self.delta, self.delta + 1e-9, 0.5) * G.D2R
        ex = self.S1[0] + self.r_max * np.cos(self.theta1 * G.D2R + extra_ang)
        ey = self.S1[1] + self.r_max * np.sin(self.theta1 * G.D2R + extra_ang)
        m = ex * ex + ey * ey <= self.arena_r ** 2 + 1e-9
        self.coarse_x = np.concatenate([self.coarse_x, ex[m]])
        self.coarse_y = np.concatenate([self.coarse_y, ey[m]])

    # ---- 几何量 ----
    def max_dist(self, P: Sequence[float]) -> float:
        """F1 中到 P 的最远距离（可检测性判据）"""
        if len(self.fine_x) == 0:
            return 0.0
        return float(np.hypot(self.fine_x - P[0], self.fine_y - P[1]).max())

    def min_dist(self, P: Sequence[float]) -> float:
        if len(self.fine_x) == 0:
            return 0.0
        return float(np.hypot(self.fine_x - P[0], self.fine_y - P[1]).min())

    def centroid(self) -> np.ndarray:
        if len(self.fine_x) == 0:
            return np.asarray(self.S1, float)
        return np.array([float(self.fine_x.mean()), float(self.fine_y.mean())])

    def extent_radius(self) -> float:
        if len(self.fine_x) == 0:
            return 0.0
        c = self.centroid()
        return float(np.hypot(self.fine_x - c[0], self.fine_y - c[1]).max())

    def polygon(self, n: int = 240) -> G.Polygon:
        """F1 的凸多边形表示（楔形 ∩ 圆盘 ∩ 圆域），用于绘图"""
        return G.locate_region([(self.S1[0], self.S1[1], self.theta1)],
                               arena=True, arena_radius=self.arena_r,
                               arena_sides=n)

    def contains(self, P: Sequence[float], tol: float = 1e-6) -> bool:
        if not G.in_wedge(P, self.S1, self.theta1, self.delta, tol):
            return False
        if math.hypot(P[0] - self.S1[0], P[1] - self.S1[1]) > self.r_max + tol:
            return False
        return math.hypot(P[0], P[1]) <= self.arena_r + tol


# ================================================================ 单点评价
def worst_diameter(S1: Sequence[float], theta1: float, S2: Sequence[float],
                   F1: F1Set, err: Sequence[float] = (-1.0, 0.0, 1.0),
                   clip_arena: bool = False,
                   arena_sides: int = 90) -> float:
    """最坏情况定位区域直径（遍历 F1 粗采样点 × 误差组合）

    ``clip_arena=False`` 时用解析求交（不含圆域约束），是真实值的**保守上界**且极快，
    适合大规模扫描；报告最终结果时设 ``clip_arena=True`` 重新精算。
    """
    if len(F1.coarse_x) == 0:
        return INF
    worst = 0.0
    for gx, gy in zip(F1.coarse_x, F1.coarse_y):
        base2 = G.bearing(S2, (gx, gy))
        for e1 in err:
            for e2 in err:
                ms = [(S1[0], S1[1], theta1 + e1), (S2[0], S2[1], base2 + e2)]
                if clip_arena:
                    poly = G.locate_region(ms, arena=True, arena_sides=arena_sides)
                    if len(poly) < 3:
                        return INF
                    d = G.polygon_diameter(poly)
                else:
                    poly = G.fast_region_vertices(ms)
                    if len(poly) < 3:
                        return INF
                    d = G.polygon_diameter(poly)
                if d > worst:
                    worst = d
    return worst


def evaluate(S1: Sequence[float], theta1: float, S2: Sequence[float], F1: F1Set,
             R_lo: float = G.R_RECV_MIN, err=(-1.0, 0.0, 1.0),
             clip_arena: bool = False) -> Dict[str, float]:
    """评价一个候选第二检测点"""
    md = F1.max_dist(S2)
    feasible = md <= R_lo + 1e-9
    rec = {
        "d": G.dist(S1, S2),
        "psi_deg": G.ang_diff(G.bearing(S1, S2), theta1),
        "max_dist_F1": md,
        "feasible": bool(feasible),
        "move_time_s": G.move_time(S1, S2),
        "worst_diam_m": (worst_diameter(S1, theta1, S2, F1, err, clip_arena)
                         if feasible else INF),
    }
    return rec


# ================================================================ 网格扫描
def scan(S1: Sequence[float] = (0.0, 0.0), theta1: float = 0.0,
         R_lo: float = G.R_RECV_MIN,
         psi_grid: Optional[Sequence[float]] = None,
         d_grid: Optional[Sequence[float]] = None,
         F1: Optional[F1Set] = None,
         err=(-1.0, 0.0, 1.0)) -> List[Dict[str, float]]:
    """在 (ψ, d) 网格上扫描，返回每条记录（含可行性、最坏直径、移动耗时）"""
    if F1 is None:
        F1 = F1Set(tuple(S1), theta1)
    if psi_grid is None:
        psi_grid = np.arange(-85, 85.1, 5.0)
    if d_grid is None:
        d_grid = np.arange(200, 2600.1, 100.0)
    rows: List[Dict[str, float]] = []
    for psi in psi_grid:
        for d in d_grid:
            S2 = (S1[0] + d * math.cos((theta1 + psi) * G.D2R),
                  S1[1] + d * math.sin((theta1 + psi) * G.D2R))
            rec = evaluate(S1, theta1, S2, F1, R_lo, err)
            rec["S2"] = S2
            rows.append(rec)
    return rows


def pareto_front(rows: Sequence[Dict[str, float]],
                 keys: Tuple[str, str] = ("move_time_s", "worst_diam_m")) -> List[Dict]:
    """帕累托前沿（两目标同时最小）"""
    ok = [r for r in rows if r["feasible"] and r["worst_diam_m"] < INF]
    ok.sort(key=lambda r: (r[keys[0]], r[keys[1]]))
    front, best = [], INF
    for r in ok:
        if r[keys[1]] < best - 1e-12:
            best = r[keys[1]]
            front.append(r)
    return front


def best_by(rows: Sequence[Dict[str, float]], key: str = "worst_diam_m") -> Optional[Dict]:
    ok = [r for r in rows if r["feasible"] and r["worst_diam_m"] < INF]
    return min(ok, key=lambda r: r[key]) if ok else None


# ================================================================ 候选区域
def candidate_region(S1: Sequence[float], theta1: float,
                     R_lo: float = G.R_RECV_MIN,
                     psi_step: float = 1.0,
                     d_step: float = 25.0) -> Dict[str, object]:
    """第二检测点候选区域（**解析刻画** + 多边形表示，便于绘图）

    候选集
        C = { S1 + d·u(θ1+ψ) : |ψ| ≤ ψ_max(R_lo), d ∈ [d1(ψ), d2(ψ)] }

    其中
        ψ_max = arcsin(R_lo/R_max) − δ
        d1/d2 由 ``d² − 2R_max·cos(|ψ|+δ)·d + (R_max²−R_lo²) ≤ 0`` 与 ``d ≤ R_lo`` 共同给出。
    """
    psi_max = analytic_psi_max(R_lo)
    pts: List[Tuple[float, float]] = []
    rows = []
    psi = -psi_max
    while psi <= psi_max + 1e-9:
        d1, d2 = analytic_d_feasible_range(psi, R_lo)
        if not math.isnan(d1) and d2 > d1:
            rows.append({"psi_deg": psi, "d_min": d1, "d_max": d2})
            # 采样区域边界
            for d in np.arange(d1, d2 + 1e-9, d_step):
                a = (theta1 + psi) * G.D2R
                pts.append((S1[0] + d * math.cos(a), S1[1] + d * math.sin(a)))
        psi += psi_step
    hull = G.convex_hull(pts) if len(pts) >= 3 else []
    return {"psi_max_deg": psi_max, "R_lo": R_lo, "rows": rows,
            "polygon": hull, "points": pts}


def candidate_region_area(reg: Dict[str, object]) -> float:
    poly = reg.get("polygon") or []
    return G.polygon_area(poly) if len(poly) >= 3 else 0.0


# ================================================================ 结论汇总
def recommend(S1: Sequence[float] = (0.0, 0.0), theta1: float = 0.0,
              R_lo_values: Sequence[float] = (1000.0, 1250.0, 1500.0),
              psi_step: float = 5.0,
              d_step: float = 100.0) -> Dict[str, object]:
    """对给定 $\\theta_1$，输出各保守可检测半径下的推荐第二点与候选区域摘要"""
    F1 = F1Set(tuple(S1), theta1)
    out: Dict[str, object] = {"F1_extent_radius": F1.extent_radius(),
                              "cases": []}
    for R_lo in R_lo_values:
        rows = scan(S1, theta1, R_lo, np.arange(-85, 85.1, psi_step),
                    np.arange(200, 2600.1, d_step), F1)
        b = best_by(rows)
        feas = [r for r in rows if r["feasible"]]
        psi_lo = min((r["psi_deg"] for r in feas), default=float("nan"))
        psi_hi = max((r["psi_deg"] for r in feas), default=float("nan"))
        reg = candidate_region(S1, theta1, R_lo)
        out["cases"].append({
            "R_lo": R_lo,
            "psi_max_analytic_deg": analytic_psi_max(R_lo),
            "psi_feasible_range_deg": (psi_lo, psi_hi),
            "n_feasible": len(feas),
            "best": b,
            "candidate_area_m2": candidate_region_area(reg),
        })
    return out
