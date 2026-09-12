# -*- coding: utf-8 -*-
"""
场景生成：目标区域中干扰源的真值数据

按题目与附件约定：
* 目标区域为半径 1800 m 的圆，圆心为原点，x 正东、y 正北；
* 干扰源总数在 10–16 个之间（正式测试不通过接口返回）；
* 每个干扰源的频道互不相同，取自 {1,...,20}；
* 有效接收半径在 1000–1500 m 之间，逐源不同；
* 全向源覆盖 360°；定向源覆盖定向方向两侧各 90°（含边界），定向方向未知。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

ARENA_R = 1800.0
R_RECV_MIN = 1000.0
R_RECV_MAX = 1500.0
N_CHANNELS = 20
N_SRC_MIN, N_SRC_MAX = 10, 16
NEAR_R = 5.0
CLEAR_R = 20.0


@dataclass
class Source:
    """一个干扰源（真值，机器狗不可见）"""

    channel: int
    x: float
    y: float
    recv_radius: float
    kind: str = "omni"                 # "omni" | "dir"
    dir_deg: Optional[float] = None    # 定向方向（度），全向源为 None
    cleared: bool = False

    # ---- 几何 ----
    @property
    def pos(self):
        return (self.x, self.y)

    def dist_to(self, p: Sequence[float]) -> float:
        return math.hypot(p[0] - self.x, p[1] - self.y)

    def in_sector(self, p: Sequence[float], tol: float = 1e-9) -> bool:
        """检测点 p 是否落在该源的信号有效覆盖角度范围内（含边界）"""
        if self.kind == "omni":
            return True
        if self.dir_deg is None:
            return True
        # 指向 p 的方向（从源看出去）
        ang = math.degrees(math.atan2(p[1] - self.y, p[0] - self.x)) % 360.0
        d = abs((ang - self.dir_deg + 180.0) % 360.0 - 180.0)
        return d <= 90.0 + tol

    def detectable(self, p: Sequence[float]) -> bool:
        """距离条件 + 角度条件都满足才可检测到信号"""
        return (self.dist_to(p) <= self.recv_radius + 1e-9) and self.in_sector(p)

    def as_truth(self) -> dict:
        return {"channel": self.channel, "x": self.x, "y": self.y,
                "recv_radius": self.recv_radius, "kind": self.kind,
                "dir_deg": self.dir_deg, "cleared": self.cleared}


@dataclass
class Scenario:
    """一局测试的完整真值"""

    sources: List[Source]
    seed: Optional[int] = None
    arena_r: float = ARENA_R

    # ---- 查询 ----
    @property
    def n_sources(self) -> int:
        return len(self.sources)

    def by_channel(self, ch: int) -> Optional[Source]:
        for s in self.sources:
            if s.channel == ch:
                return s
        return None

    def alive(self) -> List[Source]:
        return [s for s in self.sources if not s.cleared]

    def remaining(self) -> int:
        return sum(1 for s in self.sources if not s.cleared)

    def truth_table(self) -> List[dict]:
        return [s.as_truth() for s in self.sources]

    def summary(self) -> dict:
        return {
            "n_sources": self.n_sources,
            "n_omni": sum(1 for s in self.sources if s.kind == "omni"),
            "n_dir": sum(1 for s in self.sources if s.kind == "dir"),
            "channels": sorted(s.channel for s in self.sources),
            "seed": self.seed,
        }


def _sample_disk(rng: np.random.Generator, radius: float) -> tuple:
    """圆盘内均匀采样"""
    r = radius * math.sqrt(rng.random())
    a = rng.random() * 2 * math.pi
    return r * math.cos(a), r * math.sin(a)


def generate_scenario(seed: Optional[int] = None,
                      n_sources: Optional[int] = None,
                      directional_ratio: float = 0.0,
                      arena_r: float = ARENA_R,
                      r_recv: tuple = (R_RECV_MIN, R_RECV_MAX),
                      min_sep: float = 0.0,
                      max_tries: int = 2000) -> Scenario:
    """随机生成一局场景

    Parameters
    ----------
    seed : 随机种子（None 则用系统熵）
    n_sources : 干扰源个数，None 时在 [10,16] 均匀随机
    directional_ratio : 定向源占比（问题3 取 0，问题4 取 (0,1]）
    min_sep : 源之间最小间距（0 表示不做约束，忠实于题面）
    """
    rng = np.random.default_rng(seed)
    if n_sources is None:
        n_sources = int(rng.integers(N_SRC_MIN, N_SRC_MAX + 1))
    n_sources = int(max(1, min(N_CHANNELS, n_sources)))

    channels = rng.choice(np.arange(1, N_CHANNELS + 1), size=n_sources, replace=False)
    channels = sorted(int(c) for c in channels)

    # 定向源个数：按比例四舍五入，再随机选哪些频道是定向
    n_dir = int(round(directional_ratio * n_sources))
    n_dir = max(0, min(n_sources, n_dir))
    dir_flags = np.zeros(n_sources, dtype=bool)
    if n_dir:
        dir_flags[rng.choice(n_sources, size=n_dir, replace=False)] = True

    srcs: List[Source] = []
    for i, ch in enumerate(channels):
        for _ in range(max_tries):
            x, y = _sample_disk(rng, arena_r)
            if min_sep > 0 and any(math.hypot(x - s.x, y - s.y) < min_sep for s in srcs):
                continue
            break
        rr = float(rng.uniform(r_recv[0], r_recv[1]))
        if dir_flags[i]:
            srcs.append(Source(ch, x, y, rr, "dir", float(rng.uniform(0, 360))))
        else:
            srcs.append(Source(ch, x, y, rr, "omni", None))
    return Scenario(srcs, seed=seed, arena_r=arena_r)


def positional_scenario(sources: Sequence[dict], arena_r: float = ARENA_R) -> Scenario:
    """手工指定场景（用于单元测试）

    每个 dict: ``{channel, x, y, recv_radius?, kind?, dir_deg?}``
    """
    srcs = []
    for s in sources:
        srcs.append(Source(
            channel=int(s["channel"]), x=float(s["x"]), y=float(s["y"]),
            recv_radius=float(s.get("recv_radius", 1200.0)),
            kind=str(s.get("kind", "omni")),
            dir_deg=(None if s.get("dir_deg") is None else float(s["dir_deg"])),
        ))
    return Scenario(srcs, seed=None, arena_r=arena_r)
