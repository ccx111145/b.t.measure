# -*- coding: utf-8 -*-
"""自适应细分能否闭合保守认证：对失败单元递归细分，看是否全部通过"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))

from verify_continuous import ARENA_R, R_LO, two_layer                # noqa: E402
from verify_conservative import cell_certified_conservative           # noqa: E402


def certify_adaptive(P, h0, max_depth=6):
    """先按 h0 网格认证；失败单元递归细分；统计最终叶单元与失败数"""
    stats = {"leaf_ok": 0, "leaf_bad": 0, "split": 0, "first_bad": None,
             "bad_area": 0.0}

    def rec(c, hh, depth):
        if math.hypot(c[0], c[1]) + hh * math.sqrt(2) / 2 > ARENA_R + 1e-9:
            return
        if cell_certified_conservative(P, c, hh):
            stats["leaf_ok"] += 1
            return
        if depth >= max_depth:
            stats["leaf_bad"] += 1
            stats["bad_area"] += hh * hh
            if stats["first_bad"] is None:
                stats["first_bad"] = (round(c[0], 1), round(c[1], 1), hh)
            return
        stats["split"] += 1
        q = hh / 2.0
        for sx in (-1, 1):
            for sy in (-1, 1):
                rec((c[0] + sx * q / 2, c[1] + sy * q / 2), q, depth + 1)

    half = h0 / 2.0
    for x in np.arange(-ARENA_R + half, ARENA_R, h0):
        for y in np.arange(-ARENA_R + half, ARENA_R, h0):
            rec((x, y), h0, 0)
    return stats


if __name__ == "__main__":
    P = two_layer()
    print("=" * 88)
    print("自适应细分：保守认证能否闭合整个圆盘（25 点两层网）")
    print("=" * 88)
    print("%8s %10s %8s %10s %12s %10s %s"
          % ("h0(m)", "叶单元", "通过", "失败", "失败面积(m²)", "细分次数", "首个失败"))
    for h0 in (40.0, 20.0, 10.0):
        for depth in (4, 6, 8):
            t0 = time.monotonic()
            st = certify_adaptive(P, h0, max_depth=depth)
            print("%8.0f %10d %8d %10d %12.3g %10d %s  [%.0fs]"
                  % (h0, st["leaf_ok"] + st["leaf_bad"], st["leaf_ok"], st["leaf_bad"],
                     st["bad_area"], st["split"], st["first_bad"],
                     time.monotonic() - t0), flush=True)
        print()
