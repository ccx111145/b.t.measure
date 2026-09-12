# -*- coding: utf-8 -*-
"""
全知（oracle）对照：给出一个"知道全部真值"的离线规划器的代价，衡量本方法离最优有多远

oracle 定义
-----------
已知全部干扰源的位置（离线全知），则最优做法就是：
从原点出发依次走到每个源所在处清除（每个源 5 s），无需任何检测。
其代价 = 开放 TSP 路径长度 / 5 + k × 5 s。
这是**任何在线策略都不可能低于的量级**（在线策略还要额外付出"发现"的代价），
可作为竞争力的参照基准。

用法::
    python exp/oracle.py --n 60 --problem 3
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as st
import sys
from typing import List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.policy import Q3Config, run_episode                     # noqa: E402
from sim.arena import generate_scenario                             # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tab")
os.makedirs(TAB, exist_ok=True)


def tsp_open_len(pts: List[Tuple[float, float]],
                 start: Tuple[float, float] = (0.0, 0.0)) -> float:
    """开放 TSP（最近邻 + 2-opt），从 start 出发不必回到起点"""
    P = [np.asarray(p, float) for p in pts]
    if not P:
        return 0.0
    s = np.asarray(start, float)
    k = int(np.argmin([np.linalg.norm(p - s) for p in P]))
    order, left, cur = [k], set(range(len(P))) - {k}, P[k]
    while left:
        j = min(left, key=lambda i: float(np.linalg.norm(P[i] - cur)))
        order.append(j)
        left.discard(j)
        cur = P[j]
    seq = [s] + [P[i] for i in order]
    improved, guard = True, 0
    while improved and guard < 200:
        improved, guard = False, guard + 1
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
    return float(sum(np.linalg.norm(seq[i + 1] - seq[i]) for i in range(len(seq) - 1)))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    args = ap.parse_args(argv)

    ratio = 1.0 if args.problem == 4 else 0.0
    cfg = Q3Config(census_layout=("hex_ring" if args.problem == 4 else "ring"),
                   directional=(args.problem == 4))
    rows = []
    for i in range(args.n):
        sc = generate_scenario(seed=args.seed0 + i, directional_ratio=ratio)
        pts = [(s.x, s.y) for s in sc.sources]
        L = tsp_open_len(pts)
        oracle = L / 5.0 + len(pts) * 5.0
        res, _ = run_episode(sc, "T", cfg)
        rows.append({"seed": args.seed0 + i, "k": len(pts),
                     "oracle_tour_m": L, "oracle_time_s": oracle,
                     "ours_time_s": res.virtual_time_s,
                     "ours_cleared": res.n_cleared,
                     "ratio": res.virtual_time_s / oracle if oracle > 0 else float("nan"),
                     "tsp_bound_s": L / 5.0 + len(pts) * 5.0})

    tag = "q%d" % args.problem
    with open(os.path.join(TAB, "oracle_%s.csv" % tag), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    r = np.array([x["ratio"] for x in rows if np.isfinite(x["ratio"])])
    ours = np.array([x["ours_time_s"] for x in rows])
    orc = np.array([x["oracle_time_s"] for x in rows])
    tour = np.array([x["oracle_tour_m"] for x in rows])
    kk = np.array([x["k"] for x in rows])
    print("=" * 92)
    print("全知（oracle）对照：问题 %d，%d 局" % (args.problem, len(rows)))
    print("=" * 92)
    print("  oracle 巡游长度      : %.0f m（均值，k 均值 %.1f）" % (tour.mean(), kk.mean()))
    print("  oracle 总时间        : %.0f s（移动 %.0f s + 清除 %.0f s）"
          % (orc.mean(), (tour / 5).mean(), (kk * 5).mean()))
    print("  本文方法总时间       : %.0f s" % ours.mean())
    print("  **竞争力比 ours/oracle**: %.3f ± %.3f（中位 %.3f，最好 %.3f，最差 %.3f）"
          % (r.mean(), r.std(), float(np.median(r)), r.min(), r.max()))
    print("  其中“发现代价”占比   : %.1f%%"
          % (100 * (ours.mean() - orc.mean()) / ours.mean()))
    print()
    print("  说明：oracle 已知全部源位置、无需任何检测，是在线策略不可能低于的量级；")
    print("       ours/oracle 越接近 1 说明策略越接近全知最优。")
    with open(os.path.join(TAB, "oracle_%s_stats.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump({"n": len(rows), "oracle_tour_m": float(tour.mean()),
                   "oracle_time_s": float(orc.mean()),
                   "ours_time_s": float(ours.mean()),
                   "ratio_mean": float(r.mean()), "ratio_std": float(r.std()),
                   "ratio_median": float(np.median(r)),
                   "discovery_overhead_pct":
                       float(100 * (ours.mean() - orc.mean()) / ours.mean())},
                  f, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
