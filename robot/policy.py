# -*- coding: utf-8 -*-
"""
问题3：全向干扰源的自动搜索—定位—清除策略

三层结构
--------
**L1 巡测网**：中心 + 6 环（$\rho=1200$ m）共 7 点。每个巡测点对全部"未结案"频道依次检测。
    该布网满足"目标圆内任一点到最近巡测点 ≤ R_min=1000 m"，因此
    * **检测完备**：任何全向源至少被检测到一次；
    * **证否完备**：若某频道在 7 个点全部 no_signal，则其可行域为空 —— 该频道被**证明无源**。

**L2 位置估计与逼近**：对每个未结案频道，用可行域质心作为源位置估计，
    朝估计点前进（单步 ≤1100 m，保证最后一步的横向误差 ≤1100·tan1° ≈ 19.2 m），到达后复测更新估计。

**L3 清除**：当可行域的"最小包围圆半径 + 栅格容差 ≤ 20 m"时，到该圆心 `/clear` **必然成功**。

终止条件：20 个频道全部结案（cleared ∪ proven_empty）或时间/动作数预算耗尽。
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import geometry as G
from .client import ActionResult, BaseRobotClient, LocalRobotClient
from .feasible import CHANNELS, FeasibleTracker


@dataclass
class Q3Config:
    """策略参数"""

    # 巡测网
    census_layout: str = "ring"       # ring（问题3）| hex（问题4：定向源需要更密的网）
    census_radius: float = 1200.0
    census_ring_n: int = 6
    use_center_point: bool = True
    hex_step: float = 1000.0          # 六角点阵最近邻距离；≤1000 才能保证任意朝向可检出
    hex_extent: float = 2800.0        # 单层六角点阵半径（hex 布局用）
    # hex_ring 布局（问题4 推荐）：密集内层（内部包围）+ 紧贴区域外的外环（边界朝外的源）
    hex_ring_inner_step: float = 1000.0
    hex_ring_inner_extent: float = 1800.0
    hex_ring_outer_n: int = 12
    hex_ring_outer_rho: float = 1900.0

    # 可行域
    grid_cell: float = 10.0
    directional: bool = False
    svd_delta: float = 1.0            # 示向度误差半幅 δ（外近似必须知道它）
    q_outlier: int = 0                # 允许的离群读数个数（q-松弛交集；0=硬外近似）

    # ---- 规划策略（仅用于隔离对照，默认＝完整方法）----
    plan_visit_order: str = "tsp"          # tsp | sequential | random
    plan_channel_order: str = "nearest"    # nearest | roundrobin | uncertainty
    plan_clear_target: str = "mec"         # mec | centroid

    # 逼近
    homing_step_max: float = 400.0    # 单步上限（灵敏度扫描：100 会因迭代上限失败，300~500 最优）
    homing_step_min: float = 120.0
    max_iter_per_channel: int = 25

    # 清除
    clear_radius: float = G.CLEAR_R
    try_clear_spread: float = 50.0    # 可行域外接盒对角线 ≤ 该值时，试探性清除（失败也有信息）
    probe_step: float = 120.0         # 估计点与当前位置重合时的垂向探针步长
    probe_spacing: float = 400.0      # 定向源"扫过可行域"的探测点间距上限
    probe_min_spacing: float = 40.0   # 间距下限（自适应时使用）
    probe_limit: int = 60             # 每个频道最多生成的探测点数
    max_speculative_clear: int = 4    # 每个频道每次解决流程内允许的试探性清除次数
    max_iter_per_channel: int = 40
    clear_sweep_max_spread: float = 900.0   # 可行域外接盒对角线 ≤ 该值时启用 /clear 栅格扫描
    clear_sweep_max_points: int = 1600
    max_retry_rounds: int = 2               # 停滞频道允许的整体重试轮数

    # ---- 消融开关（仅用于实验，默认全关＝完整方法）----
    ablate_centroid: bool = False   # A1 去掉可行域质心估计，改为纯示向度定步长归航
    ablate_mec: bool = False        # A2 去掉 MEC<=20m 可清除判据，改为"见到方向就试清"
    ablate_certify: bool = False    # A3 关闭证否论证（失去终止证书）
    ablate_probe: bool = False      # A5 关闭自适应探测点序列
    ablate_sweep: bool = False      # A6 关闭 /clear 栅格扫描兜底
    validation_gate: bool = True    # 读数验证门：拒收与证据集矛盾的示向度
    adaptive_relax: bool = True     # 清除失败时自适应放宽 q 并重建证据集
    adaptive_relax_cap: int = 6     # 放宽上限
    census_rounds: int = 1          # 巡测重复轮数（抗间歇突发：漏检概率 <= (1-p_on)^R）

    # 预算与安全
    real_reserve_s: float = 30.0
    max_actions: int = 3000
    max_virtual_s: float = 350000.0

    # 策略变体
    strategy: str = "pipeline"        # pipeline | fused
    fuse_clear_radius: float = 260.0  # fused：巡测途中顺路清除的代价门限
    census_remeasure_spread: float = 1e9
    """巡测时：若频道已有方向证据且可行域外接盒对角线 ≤ 该值，则本点跳过复测。
    ``1e9`` 表示不跳过（每个巡测点都复测所有未结案频道）。"""


@dataclass
class EpisodeResult:
    """一局演练的统计量（对应题目表 1）"""

    seed: Optional[int]
    n_sources: int
    n_cleared: int
    clear_ratio: float
    virtual_time_s: float
    avg_locate_clear_time_s: float
    real_elapsed_s: float
    program_run_time_s: float
    n_measure: int
    n_clear_ok: int
    n_clear_fail: int
    n_switch: int
    n_actions: int
    n_requests: int
    census_points_visited: int
    ended_reason: Optional[str]
    unresolved: List[int]
    proven_empty: List[int]
    action_log_size: int
    wall_s: float

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- 检测网保证
_GUARANTEE_CACHE: Dict[tuple, bool] = {}


def _in_convex_hull(p, pts, tol: float = 1e-6) -> bool:
    """点 p 是否落在点集 pts 的凸包内（二维极角间隙判据）"""
    if len(pts) == 0:
        return False
    d = np.asarray(pts, float) - np.asarray(p, float)
    r = np.hypot(d[:, 0], d[:, 1])
    if (r <= tol).any():
        return True
    a = np.sort(np.arctan2(d[:, 1], d[:, 0]))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * math.pi]]))
    return bool(gaps.max() <= math.pi + 1e-9)


def network_guarantee(wps: Sequence[Tuple[float, float]], directional: bool,
                      n_sample: int = 900, seed: int = 20260913) -> bool:
    """检测网是否满足"**任意位置、任意朝向**的源都可被检出"

    条件：对目标区域内任一点 $G$，$G\\in\\mathrm{conv}(S_G)$，
    $S_G$ = 网中到 $G$ 距离 ≤ $R_{\\min}$ 的点。
    ``directional=False``（全向源）时该条件放宽为覆盖条件（存在一点距离 ≤ R_min）。
    """
    P = np.asarray(wps, float)
    if len(P) == 0:
        return False
    rng = np.random.default_rng(seed)
    r = G.ARENA_R * np.sqrt(rng.random(n_sample))
    a = rng.random(n_sample) * 2 * math.pi
    for gx, gy in zip(r * np.cos(a), r * np.sin(a)):
        d = np.hypot(P[:, 0] - gx, P[:, 1] - gy)
        near = P[d <= G.R_RECV_MIN + 1e-9]
        if not directional:
            if len(near) == 0:
                return False
        elif not _in_convex_hull((gx, gy), near):
            return False
    return True


class Q3Policy:
    """问题3 机器狗策略"""

    def __init__(self, client: BaseRobotClient, cfg: Optional[Q3Config] = None,
                 verbose: bool = False):
        self.cfg = cfg or Q3Config()
        self.cl = client
        self.tr = FeasibleTracker(cell=self.cfg.grid_cell,
                                  directional=self.cfg.directional,
                                  delta=self.cfg.svd_delta,
                                  q_outlier=self.cfg.q_outlier,
                                  gate=self.cfg.validation_gate)
        self.verbose = verbose
        self.visited_census = 0
        self.aborted = False
        self.stop_reason: Optional[str] = None
        self.stalled: set = set()        # 反复逼近仍无法结案的频道，避免空转
        self.census_seen: Dict[int, set] = {}
        self.census_complete = False
        self.network_guaranteed = False
        self.last_res: Dict[int, Tuple[Optional[str], Tuple[float, float]]] = {}
        self.probe_visited: Dict[int, set] = {}
        self.clear_visited: Dict[int, set] = {}

    # ------------------------------------------------------------ 工具
    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _virtual_time(self) -> float:
        eng = getattr(self.cl, "engine", None)
        if eng is not None:
            return float(eng.virtual_time)
        for rec in reversed(self.cl.log):
            resp = rec.get("response") or {}
            if resp.get("accepted"):
                return float(resp.get("virtual_time_s", 0.0))
        return 0.0

    def census_points(self) -> List[Tuple[float, float]]:
        """巡测点集合（不含访问顺序）"""
        c = self.cfg
        if c.census_layout == "hex_ring":
            inner = self._hex_lattice(c.hex_ring_inner_step, c.hex_ring_inner_extent)
            n_out = int(c.hex_ring_outer_n)
            outer = [(c.hex_ring_outer_rho * math.cos(math.pi / n_out + 2 * math.pi * i / n_out),
                      c.hex_ring_outer_rho * math.sin(math.pi / n_out + 2 * math.pi * i / n_out))
                     for i in range(n_out)]
            return inner + outer
        if c.census_layout == "hex":
            return self._hex_lattice(c.hex_step, c.hex_extent)
        pts: List[Tuple[float, float]] = []
        if c.use_center_point:
            pts.append((0.0, 0.0))
        for k in range(c.census_ring_n):
            a = 2 * math.pi * k / c.census_ring_n
            pts.append((c.census_radius * math.cos(a), c.census_radius * math.sin(a)))
        return pts

    def census_waypoints(self) -> List[Tuple[float, float]]:
        """巡测点 + **访问顺序**（由 plan_visit_order 决定）"""
        pts = self.census_points()
        order = getattr(self.cfg, "plan_visit_order", "tsp")
        if order == "sequential":
            return list(pts)
        if order == "random":
            q = list(pts)
            random.Random(20260913).shuffle(q)
            return q
        return self._tsp_order(pts, start=(0.0, 0.0))

    @staticmethod
    def _hex_lattice(step: float, extent: float) -> List[Tuple[float, float]]:
        """六角点阵（最近邻距离 = step），半径 extent 内

        只要 ``step ≤ R_min = 1000`` 且 ``extent`` 足够大（把边界外一圈也覆盖到），
        就满足 $G\\in\\mathrm{conv}(S_G)$ —— 任意位置、任意朝向的定向源都可被检出。
        """
        pts: List[Tuple[float, float]] = []
        dy = step * math.sqrt(3) / 2.0
        ny = int(2 * extent / dy) + 2
        nx = int(2 * extent / step) + 2
        for j in range(-ny, ny + 1):
            y = j * dy
            off = (step / 2.0) if (j % 2) else 0.0
            for i in range(-nx, nx + 1):
                x = i * step + off
                if math.hypot(x, y) <= extent + 1e-9:
                    pts.append((x, y))
        return pts

    @staticmethod
    def _tsp_order(pts: Sequence[Tuple[float, float]],
                   start: Tuple[float, float] = (0.0, 0.0)
                   ) -> List[Tuple[float, float]]:
        """最近邻 + 2-opt 开环路径（从 start 出发，不必回到起点）"""
        P = [np.asarray(p, float) for p in pts]
        n = len(P)
        if n == 0:
            return []
        s = np.asarray(start, float)
        k = int(np.argmin([np.linalg.norm(p - s) for p in P]))
        order = [k]
        left = set(range(n)) - {k}
        cur = P[k]
        while left:
            j = min(left, key=lambda i: float(np.linalg.norm(P[i] - cur)))
            order.append(j)
            left.discard(j)
            cur = P[j]
        seq = [s] + [P[i] for i in order]
        improved = True
        guard = 0
        while improved and guard < 200:
            improved = False
            guard += 1
            for i in range(1, len(seq) - 1):
                for j in range(i + 1, len(seq)):
                    old = float(np.linalg.norm(seq[i - 1] - seq[i]))
                    new = float(np.linalg.norm(seq[i - 1] - seq[j]))
                    if j + 1 < len(seq):
                        old += float(np.linalg.norm(seq[j] - seq[j + 1]))
                        new += float(np.linalg.norm(seq[i] - seq[j + 1]))
                    if new < old - 1e-9:
                        seq[i:j + 1] = seq[i:j + 1][::-1]
                        improved = True
        return [(float(p[0]), float(p[1])) for p in seq[1:]]

    def _budget_ok(self) -> bool:
        if self.aborted:
            return False
        if self.cl.n_requests >= self.cfg.max_actions:
            self.stop_reason = "max_actions"
            return False
        if self._virtual_time() >= self.cfg.max_virtual_s:
            self.stop_reason = "virtual_budget"
            return False
        left = self.cl.wall_budget_left()
        if left <= self.cfg.real_reserve_s:
            self.stop_reason = "real_budget"
            return False
        return True

    # ------------------------------------------------------------ 动作封装
    def _measure(self, x: float, y: float, ch: int) -> ActionResult:
        r = self.cl.measure(x, y, ch)
        if not r.accepted:
            self.aborted = True
            return r
        mr = r.measure_result
        if mr == "direction" and r.svd_deg is not None:
            self.tr.on_direction(ch, (x, y), r.svd_deg)
        elif mr == "near":
            self.tr.on_near(ch, (x, y))
        elif mr == "no_signal":
            self.tr.on_no_signal(ch, (x, y))
        self.last_res[ch] = (mr, (float(x), float(y)))
        self.probe_visited.setdefault(ch, set()).add(
            (int(round(x / 10.0)), int(round(y / 10.0))))
        return r

    def _clear(self, x: float, y: float, ch: int) -> ActionResult:
        r = self.cl.clear(x, y, ch)
        if not r.accepted:
            self.aborted = True
            return r
        if r.clear_result == "success":
            self.tr.on_clear_success(ch)
        else:
            self.tr.on_clear_fail(ch, (x, y))
        # 清除位置同样计入"已访问"，避免在同一个点反复试探；
        # 注意：**探测（/measure）与清除（/clear）必须分开记录**——
        # 探测过不等于清除过，若混用会让"28.28 m 栅格扫描必然清除"的覆盖保证失效。
        key = (int(round(x / 10.0)), int(round(y / 10.0)))
        self.probe_visited.setdefault(ch, set()).add(key)
        self.clear_visited.setdefault(ch, set()).add(key)
        return r

    # ------------------------------------------------------------ L1 巡测
    def scan_at(self, wp: Tuple[float, float], wp_index: int = -1) -> None:
        """在巡测点 wp 对全部未结案频道依次检测（移动由第一条指令完成）

        ``census_remeasure_spread`` 生效时，已经"定位得足够好"的频道在本点跳过复测，
        以省下一次 5~6 s 的检测；它们留给后续的逼近/清除阶段处理。

        ``fused`` 策略下，**扫描全部做完后**再考虑顺路清除：否则清除会把机器狗
        带离巡测点，后续检测还得折返，反而更慢。
        """
        active = sorted(self.tr.unresolved())
        for ch in active:
            if not self._budget_ok():
                return
            if (self.cfg.census_remeasure_spread < 1e8
                    and self.tr.has_direction(ch)
                    and self.tr.spread(ch) <= self.cfg.census_remeasure_spread):
                continue
            r = self._measure(wp[0], wp[1], ch)
            if r.accepted and wp_index >= 0:
                self.census_seen[ch].add(wp_index)
        if self.cfg.strategy == "fused":
            while self._budget_ok() and self._try_fused_clear(wp):
                pass

    def _try_fused_clear(self, pos) -> bool:
        """顺路清除：若某频道已可清除且代价小于门限，就地绕一下"""
        best = None
        for ch in self.tr.unresolved():
            ok, c, md, src = self.tr.clear_candidate(
                ch, self.cfg.clear_radius, prefilter=self.cfg.fuse_clear_radius)
            if not ok or c is None:
                continue
            d = math.dist(pos, c)
            if d <= self.cfg.fuse_clear_radius and (best is None or d < best[0]):
                best = (d, ch, c)
        if best is None:
            return False
        _, ch, c = best
        self._clear(c[0], c[1], ch)
        return True

    # ------------------------------------------------------------ L2 逼近
    def _approach_target(self, ch: int, pos) -> Optional[Tuple[float, float]]:
        """该频道的下一个检测点：朝可行域质心前进，单步 ≤ homing_step_max

        当质心与当前位置重合（估计已收敛但还不能保证清除）时，
        沿**垂直于最近示向度**的方向打一个探针，以获取最大角度增益。
        """
        s = self.tr.st[ch]
        if self.cfg.ablate_centroid:
            # A1：不用可行域，只沿最近一次示向度定步长前进
            if s.last_theta is None or s.last_point is None:
                return None
            if math.dist(s.last_point, pos) > 1.0:
                return None
            a = math.radians(s.last_theta)
            stp = self.cfg.homing_step_max
            return (pos[0] + stp * math.cos(a), pos[1] + stp * math.sin(a))
        if getattr(self.cfg, "plan_clear_target", "mec") == "bbox":
            cen = self.tr.bbox_center(ch)
        else:
            cen = self.tr.centroid(ch)
        if cen is None:
            return None
        d = math.dist(pos, cen)
        if d <= max(1.0, 0.05 * self.cfg.probe_step):
            spread = self.tr.spread(ch)
            step = min(max(2.0 * spread, self.cfg.probe_step), 300.0)
            if s.last_theta is None:
                ang = 0.0
            else:
                ang = s.last_theta + 90.0        # 垂直于最近视线
            u = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
            return (pos[0] + step * u[0], pos[1] + step * u[1])
        step = min(d, self.cfg.homing_step_max)
        u = ((cen[0] - pos[0]) / d, (cen[1] - pos[1]) / d)
        return (pos[0] + step * u[0], pos[1] + step * u[1])

    def _next_probe(self, ch: int, pos) -> Optional[Tuple[float, float]]:
        """可行域内尚未访问过的探测点（最近优先）

        专门对付定向源：朝向背对时 ``no_signal`` 不提供任何信息、可行域完全冻结，
        此时只能"逐个探测点扫过可行域"，直到重新测得方向或遍历完。
        间距按可行域尺度自适应：``spacing = clip(spread/2, 40, probe_spacing)``，
        否则小可行域只会生成 1 个探测点、来回死循环。
        """
        if self.cfg.ablate_probe:
            return None
        spread = self.tr.spread(ch)
        major, minor = self.tr.shape(ch)
        spacing = max(minor * 1.2, major / 8.0)
        spacing = min(self.cfg.probe_spacing,
                      max(self.cfg.probe_min_spacing, spacing))
        pts = self.tr.probe_points(ch, spacing=spacing, limit=self.cfg.probe_limit)
        if not pts:
            return None
        vis = self.probe_visited.setdefault(ch, set())
        cand = [p for p in pts
                if (int(round(p[0] / 10.0)), int(round(p[1] / 10.0))) not in vis]
        if not cand:
            return None
        cand.sort(key=lambda p: math.dist(p, pos))
        return cand[0]

    def resolve_channel(self, ch: int) -> bool:
        """把单个频道追到可清除并清除；返回是否清除成功"""
        n_spec = 0
        for _ in range(self.cfg.max_iter_per_channel):
            if not self._budget_ok():
                return False
            if self.tr.resolved(ch):
                return self.tr.is_cleared(ch)
            if self.cfg.ablate_mec:
                # A2：不要几何判据——只要此刻有方向证据就试着清除
                mr0, mp0 = self.last_res.get(ch, (None, None))
                if mr0 in ("direction", "near") and mp0 is not None:
                    r = self._clear(mp0[0], mp0[1], ch)
                    if r.accepted and r.clear_result == "success":
                        return True
                tgt0 = self._approach_target(ch, self.cl_pos())
                if tgt0 is None:
                    return False
                self._measure(tgt0[0], tgt0[1], ch)
                continue
            ok, c, md, src = self.tr.clear_candidate(
                ch, self.cfg.clear_radius,
                prefilter=max(2.0 * self.cfg.clear_radius, self.cfg.try_clear_spread))
            if ok and c is not None:
                r = self._clear(c[0], c[1], ch)
                self._log("    clear  ch=%2d at (%7.1f,%7.1f) → %s (%s md=%.1f)"
                          % (ch, c[0], c[1], r.clear_result, src, md))
                if r.clear_result == "success":
                    return True
                # 判定"必然可清除"却失败了 ⟹ 证据集已被离群读数污染（相干鬼源的典型后果）。
                # 该判据**不依赖"野值独立"**：清除失败本身就是物理证据。
                # 处理：逐步放宽松弛量 q 并用保存的全部约束重建集合，直到清除成功或到达上限。
                if (self.cfg.adaptive_relax and r.accepted
                        and r.clear_result != "success" and md is not None
                        and md <= self.cfg.clear_radius):
                    for _try in range(self.cfg.adaptive_relax_cap):
                        n_rej = self.tr.reject_cluster(ch, c)
                        if n_rej == 0:
                            break
                        ok2, c2, md2, _ = self.tr.clear_candidate(
                            ch, self.cfg.clear_radius, prefilter=None)
                        if not (ok2 and c2 is not None) or md2 > self.cfg.clear_radius:
                            continue
                        r2 = self._clear(c2[0], c2[1], ch)
                        if r2.accepted and r2.clear_result == "success":
                            return True
                        if not self._budget_ok():
                            return False
                        c = c2
                continue
            # 不保证可清除，但可行域已经很小 → 试探一次（失败也会剪掉 B(P,20)）
            spread = self.tr.spread(ch)
            if (c is not None and spread <= self.cfg.try_clear_spread
                    and n_spec < self.cfg.max_speculative_clear):
                n_spec += 1
                r = self._clear(c[0], c[1], ch)
                self._log("    clear? ch=%2d at (%7.1f,%7.1f) → %s (试探, spread=%.0f)"
                          % (ch, c[0], c[1], r.clear_result, spread))
                if r.clear_result == "success":
                    return True
                continue

            pos = self.cl_pos()
            mr, mp = self.last_res.get(ch, (None, None))
            tgt = None
            if mr == "no_signal" and mp is not None and math.dist(mp, pos) < 1.0:
                # 当前位置刚测过、没有信号 → 沿"探测点序列"扫过可行域
                tgt = self._next_probe(ch, pos)
                if tgt is None:
                    # 探测点已遍历完 → 兜底：对可行域做 28.28 m 间距的 /clear 栅格扫描。
                    # 源必在可行域内，故只要扫描覆盖整个可行域就**必然清除**。
                    if spread <= self.cfg.clear_sweep_max_spread:
                        if self._clear_sweep(ch):
                            return True
                        if self.tr.is_empty(ch):
                            return False
                        continue
                    return False
            if tgt is None:
                tgt = self._approach_target(ch, pos)
            if tgt is None:
                return False
            r = self._measure(tgt[0], tgt[1], ch)
            self._log("    %-8s ch=%2d → (%7.1f,%7.1f)  %s spread=%.1f dirs=%d"
                      % ("probe" if (mr == "no_signal") else "approach",
                         ch, tgt[0], tgt[1], r.measure_result,
                         self.tr.spread(ch), len(self.tr.dirs[ch])))
        return False

    def _clear_sweep(self, ch: int) -> bool:
        """对可行域做 28.28 m 间距的 /clear 栅格扫描（可证明的兜底）

        格距 $2\\cdot20/\\sqrt2$ 保证可行域内任一点到最近格点 ≤ 20 m，
        而源必落在可行域内，因此**只要能扫完就必然清除**。
        用于定向源朝向背对、``no_signal`` 不提供信息、可行域冻结的僵局。
        """
        if self.cfg.ablate_sweep:
            return False
        spacing = self.cfg.clear_radius * math.sqrt(2.0)
        pts = self.tr.probe_points(ch, spacing=spacing,
                                   limit=self.cfg.clear_sweep_max_points)
        if not pts:
            return False
        # 只用"清除过没有"过滤（探测过不算），否则覆盖保证会被破坏
        vis = self.clear_visited.setdefault(ch, set())
        cand = [p for p in pts
                if (int(round(p[0] / 10.0)), int(round(p[1] / 10.0))) not in vis]
        if not cand:
            return False
        cur = self.cl_pos()
        left = list(cand)
        n = 0
        while left:
            if not self._budget_ok():
                return False
            j = min(range(len(left)), key=lambda i: math.dist(left[i], cur))
            cur = left.pop(j)
            r = self._clear(cur[0], cur[1], ch)
            n += 1
            if r.clear_result == "success":
                self._log("    sweep  ch=%2d 第 %d 个格点清除成功" % (ch, n))
                return True
        self._log("    sweep  ch=%2d 扫描 %d 个格点未命中（可行域已收缩）" % (ch, n))
        return False

    def cl_pos(self) -> Tuple[float, float]:
        """当前虚拟位置：用客户端日志中最后一次被接受的动作位置近似"""
        for rec in reversed(self.cl.log):
            resp = rec.get("response") or {}
            if not resp.get("accepted"):
                continue
            pos = (rec.get("request") or {}).get("position")
            if pos is not None:
                return (float(pos["x"]), float(pos["y"]))
        return (0.0, 0.0)

    # ------------------------------------------------------------ 主循环
    def run(self, already_entered: bool = False) -> EpisodeResult:
        t_wall = time.monotonic()
        if already_entered:
            # 真实联调时 /enter 由主程序在"等待接口就绪"阶段完成
            if self.cl.remaining_real_duration_s is None:
                self.cl.remaining_real_duration_s = 1200
            self.cl._enter_wall = self.cl._enter_wall or time.monotonic()
        else:
            r = self.cl.enter()
            if not r.accepted:
                raise RuntimeError("进入目标区域失败: %r" % (r.body,))

        # ---- L1：巡测网 ----
        wps = self.census_waypoints()
        self.census_seen = {ch: set() for ch in CHANNELS}
        # 只有在**检测网确实满足"任意位置任意朝向可检出"**时，
        # "全部检测点均无信号 ⟹ 无源"才是有效论证；否则绝不能据此判空。
        key = (round(self.cfg.census_radius, 3), self.cfg.census_ring_n,
               self.cfg.use_center_point, round(self.cfg.hex_step, 3),
               round(self.cfg.hex_extent, 3), self.cfg.census_layout,
               bool(self.cfg.directional))
        if key not in _GUARANTEE_CACHE:
            _GUARANTEE_CACHE[key] = network_guarantee(wps, bool(self.cfg.directional))
        self.network_guaranteed = _GUARANTEE_CACHE[key]
        census_complete = True
        for idx, wp in enumerate(wps):
            if self.tr.all_resolved():
                break
            if not self._budget_ok():
                census_complete = False
                break
            if self.tr.unresolved():
                self.scan_at(wp, idx)
                self.visited_census += 1

        # ---- 由巡测网完备性给出的"证否" ----
        # 仅当检测网满足"任意位置、任意朝向的源都可被检出"时，
        # "在网络全部检测点都没测得方向的频道 ⟹ 无源"才成立。
        if (census_complete and self.network_guaranteed
                and not self.cfg.ablate_certify
                and self.cfg.census_remeasure_spread >= 1e8):
            for ch in list(self.tr.unresolved()):
                if (len(self.census_seen.get(ch, ())) == len(wps)
                        and self.tr.st[ch].n_detect == 0
                        and self.tr.near_pt.get(ch) is None):
                    self.tr.mark_proven_empty(ch)
        self.census_complete = census_complete

        # ---- L2/L3：逐频道逼近与清除（最近优先）----
        retry_rounds = 0
        while not self.tr.all_resolved() and self._budget_ok():
            pos = self.cl_pos()
            cand = []
            _rr = getattr(self, "_rr_ptr", 0)
            _active = [ch for ch in self.tr.unresolved()
                       if ch not in self.stalled and self.tr.centroid(ch) is not None]
            order_cfg = getattr(self.cfg, "plan_channel_order", "nearest")
            if order_cfg == "roundrobin" and _active:
                # 轮转：从上次位置继续，公平但无视距离
                _active.sort()
                _rr = _rr % len(_active)
                _active = _active[_rr:] + _active[:_rr]
                self._rr_ptr = (_rr + 1) % max(1, len(_active))
                cand = [(0.0, ch) for ch in _active]
            elif order_cfg == "uncertainty" and _active:
                # 不确定度最大优先（可行域格数最多）
                cand = [(-float(self.tr.count(ch)), ch) for ch in _active]
            else:
                for ch in _active:
                    cand.append((math.dist(pos, self.tr.centroid(ch)), ch))
            if not cand:
                # 所有未结案频道都曾失败过：给它们**第二次机会**再试一轮，
                # 因为第一次失败往往只是因为当时离得太远、或预算分配给了别的频道。
                if self.stalled and retry_rounds < self.cfg.max_retry_rounds:
                    retry_rounds += 1
                    self.stalled.clear()
                    continue
                self.stop_reason = self.stop_reason or "all_stalled"
                break
            cand.sort()
            ch = cand[0][1]
            self._log("  → 处理频道 %d（估计距离 %.0f m，%d 个候选）"
                      % (ch, cand[0][0], len(cand)))
            if not self.resolve_channel(ch):
                self.stalled.add(ch)
            if not self._budget_ok():
                break

        self.cl.exit()
        return self._report(t_wall)

    # ------------------------------------------------------------ 报告
    def _report(self, t_wall: float) -> EpisodeResult:
        """汇总一局结果

        * 进程内演练：直接读引擎的权威计数；
        * **真实 HTTP 链路**：引擎在本机不存在，改为从自身记录的动作日志重算
          （虚拟时刻取最后一次 accepted 响应的 ``virtual_time_s``；
          切频道次数按"相邻两次 /measure 的频道是否变化"重建，与附件2 §4.3 一致）。
        """
        eng = getattr(self.cl, "engine", None)
        sc = getattr(eng, "scenario", None) or getattr(self.cl, "truth_scenario", None)
        n_src = (sc.n_sources if sc is not None else len(self.tr.cleared_channels()))
        n_cleared = len(self.tr.cleared_channels())

        vt = 0.0
        n_meas = n_ok = n_fail = n_sw = 0
        last_ch = None
        for rec in self.cl.log:
            resp = rec.get("response") or {}
            if not resp.get("accepted"):
                continue
            vt = max(vt, float(resp.get("virtual_time_s", 0.0)))
            if rec.get("path") == "/measure":
                n_meas += 1
                ch = (rec.get("request") or {}).get("channel")
                if last_ch is not None and ch != last_ch:
                    n_sw += 1
                last_ch = ch
            elif rec.get("path") == "/clear":
                if resp.get("clear_result") == "success":
                    n_ok += 1
                else:
                    n_fail += 1
        if eng is not None:                     # 进程内：用引擎权威值（含 /enter 前的初始频道）
            vt = float(eng.virtual_time)
            n_meas, n_ok, n_fail, n_sw = (eng.n_measure, eng.n_clear_ok,
                                          eng.n_clear_fail, eng.n_switch)
        return EpisodeResult(
            seed=getattr(sc, "seed", None),
            n_sources=n_src,
            n_cleared=n_cleared,
            clear_ratio=(n_cleared / n_src if n_src else 0.0),
            virtual_time_s=vt,
            avg_locate_clear_time_s=(vt / n_cleared if n_cleared else float("nan")),
            real_elapsed_s=self.cl.wall_elapsed(),
            program_run_time_s=self.cl.wall_elapsed(),
            n_measure=n_meas,
            n_clear_ok=n_ok,
            n_clear_fail=n_fail,
            n_switch=n_sw,
            n_actions=n_meas + n_ok + n_fail,
            n_requests=self.cl.n_requests,
            census_points_visited=self.visited_census,
            ended_reason=(getattr(eng, "end_reason", None) if eng is not None else None),
            unresolved=self.tr.unresolved(),
            proven_empty=self.tr.proven_empty(),
            action_log_size=len(self.cl.log),
            wall_s=time.monotonic() - t_wall,
        )


# ---------------------------------------------------------------- 便捷函数
def run_episode(scenario, robot_id: str = "TEAM-TEST",
                cfg: Optional[Q3Config] = None, verbose: bool = False,
                engine_kwargs: Optional[dict] = None) -> Tuple[EpisodeResult, object]:
    """在给定场景上跑一局（进程内，无 HTTP）"""
    from sim.engine import ArenaEngine
    from .client import LocalRobotClient
    eng = ArenaEngine(scenario, robot_id, **(engine_kwargs or {}))
    eng.arm()
    cl = LocalRobotClient(eng, robot_id)
    pol = Q3Policy(cl, cfg, verbose=verbose)
    res = pol.run()
    return res, eng
