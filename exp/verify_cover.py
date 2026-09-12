# -*- coding: utf-8 -*-
"""
全向源场景（S1）的连续覆盖认证：
单元 C（中心 x、外接半径 η）若满足 min_i |x - P_i| <= R_min - η，则 C 全域被覆盖。
配合 S2 的精确凸性认证（verify_exact.py），两个场景都从"网格采样"升级为"连续保证"。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp"))

from verify_continuous import ARENA_R, R_LO, ring_net, two_layer   # noqa: E402


def certify_cover(P, h):
    eta = h * math.sqrt(2.0) / 2.0
    bad = tot = 0
    worst = None
    for x0 in np.arange(-ARENA_R, ARENA_R + 1e-9, h):
        for y0 in np.arange(-ARENA_R, ARENA_R + 1e-9, h):
            if math.hypot(x0, y0) + eta > ARENA_R + 1e-9:
                continue
            tot += 1
            dmin = float(np.hypot(P[:, 0] - x0, P[:, 1] - y0).min())
            if dmin > R_LO - eta:
                bad += 1
                if worst is None:
                    worst = (round(x0, 1), round(y0, 1), round(dmin, 1))
    return tot, bad, worst


if __name__ == "__main__":
    print("== S1（全向）连续覆盖认证：min_i |x-P_i| <= R_min - η ==")
    print("%-16s %8s %10s %10s %s" % ("网络", "单元(m)", "单元数", "未认证", "首个失败"))
    for name, P in (("七点环状网", ring_net()), ("25 点两层网", two_layer())):
        for h in (20.0, 10.0, 5.0):
            tot, bad, worst = certify_cover(P, h)
            print("%-16s %8.0f %10d %10d %s" % (name, h, tot, bad, worst))
        print()
