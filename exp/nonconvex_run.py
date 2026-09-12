# -*- coding: utf-8 -*-
"""
③ 非凸/有障碍工作区实验
  1) 在无障碍圆盘上验证判据实现与论文一致
  2) 加入矩形障碍：原 25 点网络失效多少
  3) 贪心沿遮挡补点后能否恢复
"""
from __future__ import annotations

import sys
import os
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp"))

from nonconvex import (ARENA_R, Rect, certify, greedy_augment,  # noqa: E402
                       hex_lattice, los)


def ring_net():
    import math
    return np.array([(0.0, 0.0)] + [(1200 * math.cos(2 * math.pi * i / 6),
                                     1200 * math.sin(2 * math.pi * i / 6))
                                    for i in range(6)])


def two_layer():
    import math
    inner = hex_lattice(1000.0, 1800.0)
    outer = np.array([(1900 * math.cos(math.pi / 12 + 2 * math.pi * i / 12),
                       1900 * math.sin(math.pi / 12 + 2 * math.pi * i / 12))
                      for i in range(12)])
    return np.vstack([inner, outer])


OBSTACLES = [
    Rect(-1200.0, -300.0, -600.0, 900.0),      # 一道竖墙
    Rect(400.0, -1300.0, 1500.0, -700.0),      # 一道横墙
    Rect(-200.0, 900.0, 500.0, 1400.0),        # 一个方块
]


if __name__ == "__main__":
    P = two_layer()
    print("=" * 78)
    print("A. 无障碍（对照）：应全部通过")
    print("=" * 78)
    t0 = time.monotonic()
    ok, bad, first = certify(P, [], h=40.0)
    print("  25 点两层网：通过 %d，失败 %d，首个失败 %s（用时 %.1f s）"
          % (ok, bad, first, time.monotonic() - t0))

    print()
    print("=" * 78)
    print("B. 有障碍：原来的网络还剩多少有效")
    print("=" * 78)
    t0 = time.monotonic()
    ok, bad, first = certify(P, OBSTACLES, h=40.0)
    tot = ok + bad
    print("  25 点两层网：通过 %d / %d（%.1f%%），失败 %d，首个失败位置 %s"
          % (ok, tot, 100.0 * ok / tot, bad, first))
    print("  用时 %.1f s" % (time.monotonic() - t0))

    print()
    print("=" * 78)
    print("C. 贪心沿遮挡补点")
    print("=" * 78)
    t0 = time.monotonic()
    P2 = greedy_augment(P, OBSTACLES, h=60.0, max_add=40, cand_step=200.0)
    print("  补点：%d → %d（新增 %d 个，用时 %.1f s）"
          % (len(P), len(P2), len(P2) - len(P), time.monotonic() - t0))
    ok2, bad2, first2 = certify(P2, OBSTACLES, h=40.0)
    tot2 = ok2 + bad2
    print("  补点后：通过 %d / %d（%.1f%%），失败 %d，首个失败 %s"
          % (ok2, tot2, 100.0 * ok2 / tot2, bad2, first2))

    print()
    print("=" * 78)
    print("D. 遮挡的定量刻画")
    print("=" * 78)
    # 统计每个障碍造成的"视线盲区"面积占比
    step = 40.0
    pts = [(x, y) for x in np.arange(-ARENA_R, ARENA_R + 1e-9, step)
           for y in np.arange(-ARENA_R, ARENA_R + 1e-9, step)
           if np.hypot(x, y) <= ARENA_R and not any(r.contains((x, y)) for r in OBSTACLES)]
    n_viable = 0
    for g in pts:
        cnt = sum(1 for Q in P if np.hypot(Q[0] - g[0], Q[1] - g[1]) <= 1000.0
                  and los(tuple(Q), g, OBSTACLES))
        if cnt >= 3:
            n_viable += 1
    print("  距障碍 1000 m 内仍有 ≥3 个可见网络点的位置占比：%.1f%%（%d/%d）"
          % (100.0 * n_viable / len(pts), n_viable, len(pts)))
    np.save("tab/nonconvex_net.npy", P2)
    print("  补点后的网络已存到 tab/nonconvex_net.npy")
