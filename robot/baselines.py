# -*- coding: utf-8 -*-
"""
基线策略：用于与本文方法（可行域证据 + 认证式终止）做**同信息量**对照

公平性约定
----------
所有基线都只能使用与本文方法**相同的信息**（示向度读数、no_signal、near、清除结果），
不允许使用本文特有的"有界误差可行域"几何。差别只在**决策逻辑**：

  HomingOnly      无普查：直接对最近检测到的频道做测向归航，逐个击破
  Lawnmower       犁地式全覆盖扫描 + 遇到方向就归航清除（**无认证**，走完即停）
  RandomWaypoints 随机航点 + 同样的归航清除逻辑（**无认证**，耗尽航点即停）
  GridClear       纯 /clear 方格扫描（格距 28.28 m），不使用任何测向信息（蛮力下界）

每个基线都继承 `Q3Policy` 以复用通信、日志与报告逻辑。
"""
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

from . import geometry as G
from .policy import Q3Config, Q3Policy, EpisodeResult


class _Replay(Q3Policy):
    """基线公共部分：航点序列 + 到达后扫描 + 对"当前可听到"的频道尝试归航清除"""

    name = "base"

    def __init__(self, client, cfg: Optional[Q3Config] = None, verbose: bool = False,
                 **kw):
        super().__init__(client, cfg, verbose=verbose)
        self.ever_detected: set = set()
        self.cur_detect: set = set()

    # ---- 记录"当前是否听得到" ----
    def _measure(self, x, y, ch):
        r = super()._measure(x, y, ch)
        if r.accepted:
            if r.measure_result in ("direction", "near"):
                self.ever_detected.add(ch)
                self.cur_detect.add(ch)
            else:
                self.cur_detect.discard(ch)
        return r

    def _clear(self, x, y, ch):
        r = super()._clear(x, y, ch)
        if r.accepted and r.clear_result == "success":
            self.cur_detect.discard(ch)
            self.ever_detected.discard(ch)
        return r

    # ---- 归航：沿最新示向度前进，边走边复测，直到 near 再清除 ----
    def _chase(self, ch: int, step: float = 500.0, max_iter: int = 40) -> bool:
        """不含可行域推理的朴素归航

        规则只用"最近一次读数"：direction 就沿它走，no_signal 就折返半步。
        终止于 near（距离 ≤5 m，此时 /clear 必然成功）或迭代/预算耗尽。
        """
        step_cur = step
        prev_th = None
        for _ in range(max_iter):
            if not self._budget_ok():
                return False
            if self.tr.is_cleared(ch) or self.tr.is_empty(ch):
                return self.tr.is_cleared(ch)
            pos = self.cl_pos()
            mr, mp = self.last_res.get(ch, (None, None))
            if mr is None:
                return False
            if mr == "near" and mp is not None:
                r = self._clear(mp[0], mp[1], ch)
                if r.accepted and r.clear_result == "success":
                    return True
                continue
            if mp is None or math.dist(mp, pos) > 1.0:
                self._measure(pos[0], pos[1], ch)      # 先就地复测
                continue
            th = self.tr.st[ch].last_theta
            if th is None:
                return False
            if mr == "no_signal":
                # 刚复测也没信号：沿上次示向度反向走（多半是走过头了）
                th += 180.0
            # 方向相对上一腿反转 >120° ⟹ 过冲，步长折半（自适应归航的关键）
            if prev_th is not None:
                d = abs((th - prev_th + 180.0) % 360.0 - 180.0)
                if d > 120.0:
                    step_cur = max(step_cur * 0.5, 15.0)
            prev_th = th
            if step_cur < 15.0:                        # 已经非常近，直接试探清除
                r = self._clear(pos[0], pos[1], ch)
                if r.accepted and r.clear_result == "success":
                    return True
            a = math.radians(th)
            self._measure(pos[0] + step_cur * math.cos(a),
                          pos[1] + step_cur * math.sin(a), ch)
        return False

    # ---- 站点扫描 + 归航清除（不含可行域推理）----
    def _station(self, x: float, y: float, chase: bool = True) -> None:
        """到达一个站点：扫描全部未结案频道；随后对"当前能听到"的频道逐一归航清除"""
        active = [c for c in sorted(self.tr.unresolved()) if c not in self.stalled]
        for ch in active:
            if not self._budget_ok():
                return
            self._measure(x, y, ch)
        if not chase:
            return
        for ch in sorted(self.cur_detect):
            if not self._budget_ok():
                return
            if self.tr.is_cleared(ch) or self.tr.is_empty(ch):
                continue
            if self._chase(ch):
                self.cur_detect.discard(ch)
                return                       # 一次只追一个，追完回主循环重新决策
            self.stalled.add(ch)             # 这个频道追不到，本轮不再纠缠

    def _finish(self, t_wall: float) -> EpisodeResult:
        self.cl.exit()
        return self._report(t_wall)


# ==================================================================== 基线 1
class HomingOnly(_Replay):
    """无普查：从原点开始，反复选一个能听到的频道归航清除"""

    name = "HomingOnly"

    def run(self, already_entered: bool = False) -> EpisodeResult:
        import time
        t0 = time.monotonic()
        if not already_entered:
            r = self.cl.enter()
            if not r.accepted:
                raise RuntimeError("enter failed")
        while not self.tr.all_resolved() and self._budget_ok():
            pos = self.cl_pos()
            self._station(pos[0], pos[1])
            if not self.cur_detect:
                # 听不到任何东西：换一个远一点的点（简单的确定性游走）
                k = getattr(self, "_hop", 0) + 1
                self._hop = k
                a = 2 * math.pi * k * 0.61803398875
                r_ = 600.0
                self._measure(pos[0] + r_ * math.cos(a), pos[1] + r_ * math.sin(a),
                              self.tr.unresolved()[0] if self.tr.unresolved() else 1)
        return self._finish(t0)


# ==================================================================== 基线 2
class Lawnmower(_Replay):
    """犁地式全覆盖：按行扫描整个目标区域，行距 ≤ 2·R_min"""

    name = "Lawnmower"

    def __init__(self, client, cfg=None, row_step: float = 1800.0,
                 stop_step: float = 300.0, **kw):
        super().__init__(client, cfg, **kw)
        self.row_step = row_step
        self.stop_step = stop_step

    def path(self) -> List[Tuple[float, float]]:
        pts: List[Tuple[float, float]] = []
        R = G.ARENA_R
        y = -R
        row = 0
        while y <= R + 1e-9:
            xmax = math.sqrt(max(0.0, R * R - y * y))
            xs = []
            x = -xmax
            while x <= xmax + 1e-9:
                xs.append(x)
                x += self.stop_step
            if row % 2:
                xs = xs[::-1]
            pts += [(xx, y) for xx in xs]
            y += self.row_step
            row += 1
        return pts

    def run(self, already_entered: bool = False) -> EpisodeResult:
        import time
        t0 = time.monotonic()
        if not already_entered:
            r = self.cl.enter()
            if not r.accepted:
                raise RuntimeError("enter failed")
        for (x, y) in self.path():
            if not self._budget_ok() or self.tr.all_resolved():
                break
            self._station(x, y)
        return self._finish(t0)


# ==================================================================== 基线 3
class RandomWaypoints(_Replay):
    """随机航点 + 同样的归航清除逻辑"""

    name = "RandomWaypoints"

    def __init__(self, client, cfg=None, n_waypoints: int = 40, seed: int = 0, **kw):
        super().__init__(client, cfg, **kw)
        self.n_waypoints = n_waypoints
        self.seed = seed

    def path(self) -> List[Tuple[float, float]]:
        rng = random.Random(self.seed)
        pts = []
        for _ in range(self.n_waypoints):
            r = G.ARENA_R * math.sqrt(rng.random())
            a = rng.random() * 2 * math.pi
            pts.append((r * math.cos(a), r * math.sin(a)))
        return pts

    def run(self, already_entered: bool = False) -> EpisodeResult:
        import time
        t0 = time.monotonic()
        if not already_entered:
            r = self.cl.enter()
            if not r.accepted:
                raise RuntimeError("enter failed")
        for (x, y) in self.path():
            if not self._budget_ok() or self.tr.all_resolved():
                break
            self._station(x, y)
        return self._finish(t0)


# ==================================================================== 基线 4
class GridClear(Q3Policy):
    """纯 /clear 方格扫描（格距 28.28 m）：完全不使用测向信息

    这是"蛮力"参照：它必然清除（覆盖保证），代价是极慢。
    """

    name = "GridClear"

    def __init__(self, client, cfg=None, spacing: float = 28.28,
                 max_points: int = 200000, **kw):
        super().__init__(client, cfg, **kw)
        self.spacing = spacing
        self.max_points = max_points

    def path(self) -> List[Tuple[float, float]]:
        pts = []
        R = G.ARENA_R
        y = -R
        row = 0
        while y <= R + 1e-9:
            xmax = math.sqrt(max(0.0, R * R - y * y))
            xs = []
            x = -xmax
            while x <= xmax + 1e-9:
                xs.append(x)
                x += self.spacing
            if row % 2:
                xs = xs[::-1]
            pts += [(xx, y) for xx in xs]
            if len(pts) > self.max_points:
                return pts[:self.max_points]
            y += self.spacing
            row += 1
        return pts

    def run(self, already_entered: bool = False) -> EpisodeResult:
        import time
        t0 = time.monotonic()
        if not already_entered:
            r = self.cl.enter()
            if not r.accepted:
                raise RuntimeError("enter failed")
        for (x, y) in self.path():
            if not self._budget_ok() or self.tr.all_resolved():
                break
            for ch in sorted(self.tr.unresolved()):
                r = self._clear(x, y, ch)
                if not self._budget_ok():
                    break
        return self._finish(t0)


ALL_BASELINES = {
    "HomingOnly": HomingOnly,
    "Lawnmower": Lawnmower,
    "RandomWaypoints": RandomWaypoints,
}
