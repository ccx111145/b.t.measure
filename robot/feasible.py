# -*- coding: utf-8 -*-
"""
频道可行域追踪（问题3/4 的"证据模型"）

对每个频道 c 维护一个**外近似**可行域 ``E_c``（10 m 栅格上的布尔掩码），
每次动作反馈按 §2.1 的算子裁剪：

===============  ==========================================  ==========================
反馈             全向（问题3）                                定向（问题4）
===============  ==========================================  ==========================
``direction``    ``E ∩ W(P,θ̄) ∩ B(P,1500)``                 同左（且 P 在覆盖半平面内）
``near``         ``E ∩ B(P,5)``                              同左
``no_signal``    ``E \\ B(P,1000)``                          **不改**（角度遮挡）
``clear`` 未发现  ``E \\ B(P,20)``                           同左
``clear`` 成功    该频道结案                                  同左
===============  ==========================================  ==========================

外近似保证「真值所在的栅格单元永不被剔除」，因此
``best_clear_point`` 返回 ``ok=True`` 时一次 ``/clear`` **必然成功**；
而空掩码则意味着该频道被**证明无源**。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from . import geometry as G

CHANNELS = tuple(range(1, 21))


@dataclass
class ChannelState:
    mask: np.ndarray
    viol: Optional[np.ndarray] = None   # 每个栅格被"读数"违反的次数（q-松弛交集用）
    n_dir: int = 0                 # 收到的 direction 次数
    n_detect: int = 0              # direction 或 near 的次数（**任何**"发现了源"的证据）
    last_theta: Optional[float] = None
    last_point: Optional[Tuple[float, float]] = None
    cleared: bool = False
    proven_empty: bool = False     # 由巡测网完备性证书判定的"无源"（对伪装示向度免疫）
    n_rejected: int = 0            # 被验证门拒收的读数条数


class FeasibleTracker:
    """20 个频道的可行域集合

    同时保留两套表示：
    * **栅格掩码**（10 m 外近似）：能处理 ``no_signal`` / ``clear`` 失败这类非凸约束；
    * **方向证据多边形**（问题1 的精确算法）：只由 ``direction`` 楔形求交，
      无栅格容差，因此判定"能否一次清除"更紧。
    两者都是真值集合的**外近似**，取二者中更优的判据即可。
    """

    def __init__(self, cell: float = 10.0, radius: float = G.ARENA_R,
                 directional: bool = False, delta: float = G.SV_DELTA,
                 q_outlier: int = 0, gate: bool = True):
        self.r = G.get_raster(cell=cell, radius=radius)
        self.directional = bool(directional)
        self.delta = float(delta)
        # q = 允许的离群读数个数（0 = 原来的硬外近似）
        self.q_outlier = int(max(0, q_outlier))
        # 验证门：与现有证据集矛盾的示向度一律拒收（在线异常值剔除）
        self.gate = bool(gate)
        self.n_rejected_total = 0
        # 每个频道的全部约束 (kind, pos, val)，用于在清除失败后重建证据集
        self.constraints: Dict[int, List[tuple]] = {ch: [] for ch in CHANNELS}
        self.q_used: Dict[int, int] = {ch: self.q_outlier for ch in CHANNELS}
        self.n_rebuild: Dict[int, int] = {ch: 0 for ch in CHANNELS}
        self.st: Dict[int, ChannelState] = {
            ch: ChannelState(mask=np.ones(self.r.n, dtype=bool),
                             viol=np.zeros(self.r.n, dtype=np.int16))
            for ch in CHANNELS
        }
        self.n_updates: Dict[int, int] = {ch: 0 for ch in CHANNELS}
        self.dirs: Dict[int, List[Tuple[Tuple[float, float], float]]] = {
            ch: [] for ch in CHANNELS}
        self.near_pt: Dict[int, Optional[Tuple[float, float]]] = {
            ch: None for ch in CHANNELS}

    # ------------------------------------------------------------ 查询
    def mask(self, ch: int) -> np.ndarray:
        return self.st[ch].mask

    def count(self, ch: int) -> int:
        return int(self.st[ch].mask.sum())

    def is_empty(self, ch: int) -> bool:
        """该频道是否被**证明**无源。

        两条路径要分开看（回应多径野值导致的"假阴性证书"问题）：

        * 巡测网完备性证书 —— 只需"全部检测点都无信号"，而伪装示向度只会**增加**检出，
          不会掩盖真检出，故对伪装测向角免疫；
        * 栅格裁剪为空 —— 依赖"每条读数都可信"。一旦出现超出 δ 的野值（多径跳变），
          真值可能被错误剔除，集合变空 ⟹ **假阴性证书**。

        因此 q_outlier > 0 时不再承认栅格为空作为无源证明。
        """
        st = self.st[ch]
        if st.proven_empty:
            return True
        if self.q_outlier > 0:
            return False
        return not st.mask.any()

    def is_cleared(self, ch: int) -> bool:
        return self.st[ch].cleared

    def resolved(self, ch: int) -> bool:
        return self.st[ch].cleared or self.is_empty(ch)

    def unresolved(self) -> List[int]:
        return [ch for ch in CHANNELS if not self.resolved(ch)]

    def cleared_channels(self) -> List[int]:
        return [ch for ch in CHANNELS if self.st[ch].cleared]

    def proven_empty(self) -> List[int]:
        return [ch for ch in CHANNELS if self.is_empty(ch) and not self.st[ch].cleared]

    def all_resolved(self) -> bool:
        return all(self.resolved(ch) for ch in CHANNELS)

    def has_direction(self, ch: int) -> bool:
        return self.st[ch].n_dir > 0

    # ------------------------------------------------------------ 估计量
    def bbox_center(self, ch: int) -> Optional[Tuple[float, float]]:
        """可行点集外接盒的中心（另一种点估计，用于规划策略的隔离对照）"""
        idx = np.flatnonzero(self.st[ch].mask)
        if len(idx) == 0:
            return None
        x, y = self.r.x[idx], self.r.y[idx]
        return (float(x.min() + x.max()) / 2.0, float(y.min() + y.max()) / 2.0)

    def centroid(self, ch: int) -> Optional[np.ndarray]:
        idx = np.flatnonzero(self.st[ch].mask)
        if len(idx) == 0:
            return None
        return np.array([float(self.r.x[idx].mean()), float(self.r.y[idx].mean())])

    def spread(self, ch: int) -> float:
        """可行域的外接盒对角线长（廉价的范围度量）"""
        idx = np.flatnonzero(self.st[ch].mask)
        if len(idx) == 0:
            return 0.0
        x, y = self.r.x[idx], self.r.y[idx]
        return float(math.hypot(x.max() - x.min(), y.max() - y.min()))

    def clear_point(self, ch: int, radius: float = G.CLEAR_R):
        """栅格判据：``(ok, center, maxdist)``"""
        return self.r.best_clear_point(radius, mask=self.st[ch].mask)

    # ------------------------------------------------------------ 精确多边形判据
    def evidence_polygon(self, ch: int):
        """只由 ``direction`` 楔形求交得到的定位区域（限制在栅格掩码包围盒内）。

        由于先验矩形按"栅格可行域的外接盒 + pad"取得，而真值必落在某个可行单元内，
        故该多边形**仍是真值的上界**；但不受栅格容差影响，判据更紧。
        """
        ev = self.dirs[ch]
        if len(ev) < 2:
            return []
        ext = self.r.extent() if False else None
        idx = np.flatnonzero(self.st[ch].mask)
        if len(idx) == 0:
            return []
        x, y = self.r.x[idx], self.r.y[idx]
        pad = self.r.pad
        rect = G.rect_polygon(float(x.min()) - pad, float(x.max()) + pad,
                              float(y.min()) - pad, float(y.max()) + pad)
        return G.locate_region([(p[0], p[1], th) for (p, th) in ev],
                               initial_poly=rect, delta=self.delta)

    def polygon_clear_point(self, ch: int, radius: float = G.CLEAR_R):
        """精确多边形判据：``(ok, center, maxdist)``，无栅格容差"""
        poly = self.evidence_polygon(ch)
        if len(poly) < 3:
            return False, None, float("inf")
        c, r = G.mec(poly)
        if c is None:
            return False, None, float("inf")
        md = max(math.dist(v, (c[0], c[1])) for v in poly)
        return (md <= radius + 1e-9), (float(c[0]), float(c[1])), float(md)

    def clear_candidate(self, ch: int, radius: float = G.CLEAR_R,
                        prefilter: Optional[float] = None):
        """综合判据：``near`` > 精确多边形 > 栅格，返回更优者

        Returns ``(guaranteed, center, maxdist, source)``；``guaranteed=True`` 时
        到 ``center`` 执行 ``/clear`` **必然成功**。

        先用廉价预筛：设可行点集外接盒对角线为 ``d``，则其最小包围圆半径 ``≥ d/2``，
        故 ``d > 2·radius`` 时**必不可清除**，直接返回，避免昂贵的多边形求交。
        """
        if self.st[ch].cleared:
            return True, None, 0.0, "cleared"
        np_ = self.near_pt[ch]
        if np_ is not None:
            return True, np_, 0.0, "near"          # ≤5 m ⟹ 必然可清除

        idx = np.flatnonzero(self.st[ch].mask)
        if len(idx) == 0:
            return False, None, float("inf"), "empty"
        x, y = self.r.x[idx], self.r.y[idx]
        d = math.hypot(float(x.max() - x.min()), float(y.max() - y.min()))
        lim = prefilter if prefilter is not None else 2.0 * radius
        if d > lim:
            return False, None, float("inf"), "prefilter"

        ok1, c1, md1 = self.polygon_clear_point(ch, radius)
        if ok1:
            return True, c1, md1, "polygon"
        ok2, c2, md2 = self.clear_point(ch, radius)
        if ok2:
            return True, c2, md2, "raster"
        # 都不保证：返回更紧的一个作为"试探点"
        if md1 <= md2 and c1 is not None:
            return False, c1, md1, "polygon"
        if c2 is not None:
            return False, c2, md2, "raster"
        return False, None, float("inf"), "none"

    def near_point(self, ch: int) -> Optional[Tuple[float, float]]:
        """可行域中离其质心最近的代表点"""
        c = self.centroid(ch)
        if c is None:
            return None
        idx = np.flatnonzero(self.st[ch].mask)
        x, y = self.r.x[idx], self.r.y[idx]
        k = int(np.argmin((x - c[0]) ** 2 + (y - c[1]) ** 2))
        return float(x[k]), float(y[k])

    def farthest_point(self, ch: int, p) -> Optional[Tuple[float, float]]:
        """可行域中离 p 最远的点（用于"穿过整个可行域"的扫掠）"""
        idx = np.flatnonzero(self.st[ch].mask)
        if len(idx) == 0:
            return None
        x, y = self.r.x[idx], self.r.y[idx]
        k = int(np.argmax((x - p[0]) ** 2 + (y - p[1]) ** 2))
        return float(x[k]), float(y[k])

    def shape(self, ch: int) -> Tuple[float, float]:
        """可行域的 (长轴尺度, 短轴尺度)。

        对"细长楔形"（例如 1500 m 长、50 m 宽）必须按**短轴**定探测间距，
        否则按外接盒对角线定间距会造成严重欠采样、探测点全部落在可行域外沿。
        """
        idx = np.flatnonzero(self.st[ch].mask)
        if len(idx) <= 1:
            return 0.0, 0.0
        x, y = self.r.x[idx].astype(np.float64), self.r.y[idx].astype(np.float64)
        x = x - x.mean()
        y = y - y.mean()
        cxx = float((x * x).mean())
        cyy = float((y * y).mean())
        cxy = float((x * y).mean())
        tr = cxx + cyy
        det = cxx * cyy - cxy * cxy
        disc = max(0.0, tr * tr / 4.0 - det)
        l1 = tr / 2.0 + math.sqrt(disc)
        l2 = max(0.0, tr / 2.0 - math.sqrt(disc))
        return 2.0 * math.sqrt(max(l1, 0.0)) * 1.732, 2.0 * math.sqrt(l2) * 1.732

    def probe_points(self, ch: int, spacing: float = 400.0,
                     limit: int = 24) -> List[Tuple[float, float]]:
        """在可行域内生成间距 **严格为** ``spacing`` 的网格点（bbox 对齐 + KD 过滤）

        注意：不能用"掩码格点抽样 + 取整去重"——细长/对角走向的可行域会把多个格点
        折叠到同一个桶里，只剩 2~3 个点，**栅格扫描的覆盖保证就失效了**。
        这里改为在可行域外接盒上铺规则网格，再用 KD 树保留距可行域 ≤ 0.75·spacing 的点，
        这样可行域内任一点到某个保留点的距离 ≤ spacing/√2。
        """
        idx = np.flatnonzero(self.st[ch].mask)
        if len(idx) == 0:
            return []
        x, y = self.r.x[idx], self.r.y[idx]
        pts = np.stack([x, y], axis=1)
        x0, x1 = float(x.min()), float(x.max())
        y0, y1 = float(y.min()), float(y.max())
        nx = int((x1 - x0) / spacing) + 2
        ny = int((y1 - y0) / spacing) + 2
        if nx * ny > limit * 40:
            return []
        gx = x0 + np.arange(nx) * spacing
        gy = y0 + np.arange(ny) * spacing
        GX, GY = np.meshgrid(gx, gy, indexing="ij")
        G = np.stack([GX.ravel(), GY.ravel()], 1)
        try:
            from scipy.spatial import cKDTree
            d, _ = cKDTree(pts).query(G)
            keep = d <= spacing * 0.75 + 1e-9
        except Exception:
            keep = np.ones(len(G), bool)
        out = G[keep]
        if len(out) == 0:
            return []
        cx, cy = float(x.mean()), float(y.mean())
        order = np.argsort((out[:, 0] - cx) ** 2 + (out[:, 1] - cy) ** 2)
        out = out[order][:limit]
        return [(float(p[0]), float(p[1])) for p in out]

    def grid_count(self, ch: int, spacing: float) -> int:
        idx = np.flatnonzero(self.st[ch].mask)
        if len(idx) == 0:
            return 0
        x, y = self.r.x[idx], self.r.y[idx]
        nx = int((float(x.max()) - float(x.min())) / spacing) + 2
        ny = int((float(y.max()) - float(y.min())) / spacing) + 2
        return nx * ny

    # ------------------------------------------------------------ 更新算子
    def _record(self, ch: int, kind: str, pos, val=None) -> None:
        self.constraints[ch].append((kind, (float(pos[0]), float(pos[1])), val))

    def rebuild(self, ch: int, q: int) -> None:
        """用保存的全部约束以松弛量 q 重建证据集（q 越大集合越大）"""
        st = self.st[ch]
        st.viol = np.zeros(self.r.n, dtype=np.int16)
        st.mask = np.ones(self.r.n, dtype=bool)
        st.proven_empty = False
        self.q_used[ch] = int(q)
        keep_all: Optional[np.ndarray] = None
        for (kind, p, val) in self.constraints[ch]:
            if kind == "dir":
                keep = self.r.keep_wedge(p, val, self.delta, mask=None)
                keep = keep & self.r.keep_ball(p, self.r.radius)
            elif kind == "near":
                keep = self.r.keep_ball(p, val)
            elif kind == "nosig":
                keep = self.r.keep_outside_ball(p, val)
            elif kind == "clearfail":
                keep = self.r.keep_outside_ball(p, val)
            else:
                continue
            st.viol += (~keep)
        st.mask = st.viol <= int(q)
        self.n_rebuild[ch] += 1

    def wedge_contains(self, p, theta_deg, target, tol_deg: float = 1e-9) -> bool:
        """楔形 W(p,θ) 是否包含 target"""
        vx, vy = target[0] - p[0], target[1] - p[1]
        d = math.hypot(vx, vy)
        if d < 1e-9:
            return True
        beta = math.degrees(math.atan2(vy, vx))
        diff = abs((beta - theta_deg + 180.0) % 360.0 - 180.0)
        return diff <= self.delta + 1e-9

    def reject_cluster(self, ch: int, c_star, cap_frac: float = 0.9) -> int:
        """否决投票给 c_star 的示向度读数并重建证据集

        返回被否决的读数条数；若否决比例过高（>cap_frac）则不否决，
        以免把全部证据删光（那说明问题不在某一簇）。
        """
        cons = self.constraints[ch]
        dirs = [(k, pp, vv) for (k, pp, vv) in cons if k == "dir"]
        if len(dirs) < 2:
            return 0
        doomed = [(k, pp, vv) for (k, pp, vv) in dirs
                  if self.wedge_contains(pp, vv, c_star)]
        if not doomed or len(doomed) > cap_frac * len(dirs):
            return 0
        self.constraints[ch] = [c for c in cons if c not in doomed]
        self.rebuild(ch, self.q_used.get(ch, self.q_outlier))
        self.n_rejected_total += len(doomed)
        return len(doomed)

    def _apply(self, ch: int, keep: np.ndarray):
        """施加一条约束。

        q = 0 时退化为原来的硬交集；q > 0 时为 **q-松弛交集**：
        只有某栅格被 *超过 q* 条读数违反时才被剔除，从而容忍至多 q 个离群读数
        （例如多径造成的假示向角）。若真值只会被离群读数违反，则它必然留在集合中，
        于是定理 1 的可清除判据在 q > 0 下原样成立。
        """
        st = self.st[ch]
        if st.viol is None:
            st.viol = np.zeros(self.r.n, dtype=np.int16)
            st.mask = np.ones(self.r.n, dtype=bool)
        st.viol += (~keep)
        st.mask = st.viol <= self.q_outlier
        self.n_updates[ch] += 1

    def on_direction(self, ch: int, p: Tuple[float, float], theta_deg: float,
                     delta: Optional[float] = None):
        """示向度读数：先过验证门，再更新证据集

        验证门：若新的楔形与现有证据集**完全不相交**，说明该读数与已积累的全部
        证据矛盾（典型成因是多径造成的假示向），此时拒收，不更新集合。
        """
        dlt = self.delta if delta is None else float(delta)
        keep = self.r.keep_wedge(p, theta_deg, dlt, mask=None)
        self._record(ch, "dir", p, float(theta_deg))
        st = self.st[ch]
        if self.gate and not bool((st.mask & keep).any()):
            st.n_rejected += 1
            self.n_rejected_total += 1
            return False
        self._apply(ch, keep)
        s = self.st[ch]
        s.n_dir += 1
        s.n_detect += 1
        s.last_theta = float(theta_deg)
        s.last_point = (float(p[0]), float(p[1]))
        self.dirs[ch].append(((float(p[0]), float(p[1])), float(theta_deg)))
        return True

    def on_near(self, ch: int, p: Tuple[float, float],
                near_r: float = G.NEAR_R):
        self._apply(ch, self.r.keep_ball(p, near_r))
        self._record(ch, "near", p, near_r)
        self.near_pt[ch] = (float(p[0]), float(p[1]))
        self.st[ch].n_detect += 1        # near 同样是"发现了源"，绝不能被当成无源

    def on_no_signal(self, ch: int, p: Tuple[float, float],
                     r_lo: float = G.R_RECV_MIN):
        if self.directional:
            return                      # 定向源：no_signal 不提供距离信息
        if self.q_outlier > 0:
            return                      # 抗离群值模式：漏检也会造成假 no_signal，不裁剪
        self._record(ch, "nosig", p, r_lo)
        self._apply(ch, self.r.keep_outside_ball(p, r_lo))

    def on_clear_fail(self, ch: int, p: Tuple[float, float],
                      clear_r: float = G.CLEAR_R):
        self._record(ch, "clearfail", p, clear_r)
        self._apply(ch, self.r.keep_outside_ball(p, clear_r))

    def on_clear_success(self, ch: int):
        self.st[ch].cleared = True
        self.st[ch].mask[:] = False

    def mark_proven_empty(self, ch: int) -> None:
        """由巡测网完备性给出的证否：直接把集合置空。

        定向源的 ``no_signal`` 不提供距离信息，因此栅格裁剪无法把可行域裁空。
        但若检测网满足“任意位置、任意朝向的源都可被检出”，则
        “该频道在网络全部检测点均无信号” ⟹ 该频道无源。
        """
        st = self.st[ch]
        st.proven_empty = True
        if st.viol is None:
            st.viol = np.zeros(self.r.n, dtype=np.int16)
        st.viol[:] = self.q_outlier + 1
        st.mask = np.zeros(self.r.n, dtype=bool)

    # ------------------------------------------------------------ 诊断
    def summary(self) -> dict:
        return {
            "cleared": self.cleared_channels(),
            "proven_empty": self.proven_empty(),
            "unresolved": self.unresolved(),
            "counts": {ch: self.count(ch) for ch in CHANNELS},
        }
