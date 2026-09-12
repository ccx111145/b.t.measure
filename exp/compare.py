# -*- coding: utf-8 -*-
"""
策略对照实验：本文方法 vs 四个基线（同场景、同种子、同预算）

用法::
    python exp/compare.py --n 30 --problem 3
    python exp/compare.py --n 30 --problem 4

输出：tab/compare_<problem>.csv（逐局）、tab/compare_<problem>_stats.json（汇总 +
配对检验），并在屏幕上打印对照表。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics as stat
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.baselines import GridClear, HomingOnly, Lawnmower, RandomWaypoints  # noqa: E402
from robot.client import LocalRobotClient                                     # noqa: E402
from robot.policy import Q3Config, Q3Policy                                   # noqa: E402
from sim.arena import generate_scenario                                       # noqa: E402
from sim.engine import ArenaEngine                                            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tab")
os.makedirs(TAB, exist_ok=True)


def make(strategy: str, cl, cfg: Q3Config, seed: int):
    if strategy == "Ours":
        return Q3Policy(cl, cfg)
    if strategy == "HomingOnly":
        return HomingOnly(cl, cfg)
    if strategy == "Lawnmower":
        return Lawnmower(cl, cfg, row_step=2 * 1000.0, stop_step=300.0)
    if strategy == "RandomWaypoints":
        return RandomWaypoints(cl, cfg, n_waypoints=40, seed=seed)
    if strategy == "GridClear":
        return GridClear(cl, cfg)
    raise ValueError(strategy)


def run_one(strategy: str, seed: int, problem: int,
            max_virtual: float = 60_000.0) -> Dict:
    ratio = 1.0 if problem == 4 else 0.0
    sc = generate_scenario(seed=seed, directional_ratio=ratio)
    eng = ArenaEngine(sc, "T", max_virtual=max_virtual, idem_limit=200000)
    eng.arm()
    cl = LocalRobotClient(eng, "T")
    cfg = Q3Config(census_layout=("hex_ring" if problem == 4 else "ring"),
                   directional=(problem == 4))
    pol = make(strategy, cl, cfg, seed)
    t0 = time.monotonic()
    try:
        res = pol.run()
    except Exception as e:                                        # noqa: BLE001
        return {"strategy": strategy, "seed": seed, "error": repr(e),
                "n_sources": sc.n_sources, "n_cleared": 0, "clear_ratio": 0.0,
                "virtual_time_s": eng.virtual_time,
                "avg_locate_clear_time_s": float("nan"),
                "n_actions": eng.n_measure + eng.n_clear_ok + eng.n_clear_fail,
                "wall_s": time.monotonic() - t0,
                "certified": 0}
    certified = int(bool(pol.tr.all_resolved()))
    return {"strategy": strategy, "seed": seed, "error": None,
            "n_sources": res.n_sources, "n_cleared": res.n_cleared,
            "clear_ratio": res.clear_ratio, "virtual_time_s": res.virtual_time_s,
            "avg_locate_clear_time_s": res.avg_locate_clear_time_s,
            "n_actions": res.n_actions, "wall_s": time.monotonic() - t0,
            "certified": certified}


def wilcoxon_like(a: List[float], b: List[float]) -> Dict:
    """配对符号检验 + 差值 bootstrap 置信区间（不依赖 scipy）"""
    d = np.array(a, float) - np.array(b, float)
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return {"n": 0, "mean_diff": float("nan"), "ci95": [float("nan")] * 2,
                "wins": 0, "losses": 0, "sign_test_p": float("nan")}
    pos = int((d > 0).sum())
    neg = int((d < 0).sum())
    n = pos + neg
    # 双侧符号检验 p 值（精确二项，n 小）
    if n == 0:
        p = 1.0
    else:
        from math import comb
        k = min(pos, neg)
        p = 2.0 * sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
        p = min(1.0, p)
    rng = np.random.default_rng(20260913)
    boots = [float(np.mean(rng.choice(d, len(d), replace=True))) for _ in range(2000)]
    return {"n": int(len(d)), "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "wins": pos, "losses": neg, "sign_test_p": float(p)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    ap.add_argument("--strategies", nargs="*",
                    default=["Ours", "HomingOnly", "Lawnmower", "RandomWaypoints"])
    ap.add_argument("--max-virtual", type=float, default=60_000.0,
                    help="虚拟时间预算上限（各策略相同）")
    args = ap.parse_args(argv)

    rows: List[Dict] = []
    for st in args.strategies:
        t0 = time.monotonic()
        for i in range(args.n):
            rows.append(run_one(st, args.seed0 + i, args.problem, args.max_virtual))
        print("  %-16s 完成 %d 局，用时 %.1f s"
              % (st, args.n, time.monotonic() - t0), flush=True)

    tag = "q%d" % args.problem
    csv_path = os.path.join(TAB, "compare_%s.csv" % tag)
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("  逐局结果:", csv_path)

    print()
    print("=" * 104)
    print("策略对照（问题 %d，各 %d 局，虚拟时间预算 %.0f s）"
          % (args.problem, args.n, args.max_virtual))
    print("=" * 104)
    print("%-16s %10s %10s %14s %12s %10s %10s"
          % ("策略", "清除比例", "认证完成率", "平均定位清除时间", "总虚拟时间", "动作数", "本机(s)"))
    summary = {}
    for st in args.strategies:
        rs = [r for r in rows if r["strategy"] == st]
        cr = [r["clear_ratio"] for r in rs]
        ct = [r["certified"] for r in rs]
        t = [r["avg_locate_clear_time_s"] for r in rs if r["n_cleared"] > 0]
        vt = [r["virtual_time_s"] for r in rs]
        ac = [r["n_actions"] for r in rs]
        print("%-16s %10.3f %10.3f %14s %12.0f %10.0f %10.1f"
              % (st, stat.fmean(cr), stat.fmean(ct),
                 ("%.1f±%.1f" % (stat.fmean(t), stat.pstdev(t))) if t else "—",
                 stat.fmean(vt), stat.fmean(ac),
                 stat.fmean([r["wall_s"] for r in rs])))
        summary[st] = {"clear_ratio_mean": stat.fmean(cr),
                       "certified_mean": stat.fmean(ct),
                       "avg_time_mean": stat.fmean(t) if t else None,
                       "avg_time_std": stat.pstdev(t) if len(t) > 1 else None,
                       "virtual_time_mean": stat.fmean(vt),
                       "actions_mean": stat.fmean(ac)}

    # 与本文方法配对检验（清除比例、平均时间）
    if "Ours" in args.strategies:
        ours = {r["seed"]: r for r in rows if r["strategy"] == "Ours"}
        tests = {}
        for st in args.strategies:
            if st == "Ours":
                continue
            other = {r["seed"]: r for r in rows if r["strategy"] == st}
            seeds = sorted(set(ours) & set(other))
            if not seeds:
                continue
            a_cr = [ours[s]["clear_ratio"] for s in seeds]
            b_cr = [other[s]["clear_ratio"] for s in seeds]
            a_t = [ours[s]["avg_locate_clear_time_s"] for s in seeds]
            b_t = [other[s]["avg_locate_clear_time_s"] for s in seeds]
            tests[st] = {"clear_ratio": wilcoxon_like(a_cr, b_cr),
                         "avg_time": wilcoxon_like(a_t, b_t)}
        summary["_paired_tests_vs_Ours"] = tests
        print()
        print("与本文方法的配对检验（Ours − 基线；正=本文更大/更慢）")
        for st, d in tests.items():
            c = d["clear_ratio"]
            t = d["avg_time"]
            print("  %-16s 清除比例差 %+.4f (95%%CI [%+.4f,%+.4f], p=%.3g, 胜/负 %d/%d) | "
                  "平均时间差 %+.1f s (p=%.3g)"
                  % (st, c["mean_diff"], c["ci95"][0], c["ci95"][1], c["sign_test_p"],
                     c["wins"], c["losses"], t["mean_diff"], t["sign_test_p"]))

    with open(os.path.join(TAB, "compare_%s_stats.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("  汇总:", os.path.join(TAB, "compare_%s_stats.json" % tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

