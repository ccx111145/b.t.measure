# -*- coding: utf-8 -*-
"""
抗离群值实验：多径野值下的硬外近似 vs q-松弛交集

对 p_outlier ∈ {0, 0.02, 0.05, 0.10} 与 q ∈ {0, 2, 4} 的每组组合跑 N 局，
统计：满清除率、**假证否次数**（把有源判为无源 —— 致命错误）、认证完成率、总虚拟时间。

用法: python exp/outlier.py --n 20 --problem 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
import gc
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.policy import Q3Config, run_episode        # noqa: E402
from sim.arena import generate_scenario               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tab")
os.makedirs(TAB, exist_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--problem", type=int, default=4, choices=[3, 4])
    ap.add_argument("--cell", type=float, default=10.0)
    args = ap.parse_args(argv)

    ratio = 1.0 if args.problem == 4 else 0.0
    base = dict(census_layout=("hex_ring" if args.problem == 4 else "ring"),
                directional=(args.problem == 4))
    rows = []
    print("%-8s %-3s %-9s %-9s %-9s %-10s %-9s"
          % ("p_out", "q", "满清除率", "假证否", "认证率", "总时间(s)", "野值数"))
    for p_out in (0.0, 0.02, 0.05, 0.10):
        for q in (0, 2, 4):
            full = fe = cert = 0
            ts, outs = [], 0
            t0 = time.monotonic()
            for i in range(args.n):
                seed = args.seed0 + i
                sc = generate_scenario(seed=seed, directional_ratio=ratio)
                cfg = Q3Config(q_outlier=q, grid_cell=args.cell, **base)
                r, eng = run_episode(sc, "T", cfg,
                                     engine_kwargs={"p_outlier": p_out})
                src_ch = {s.channel for s in sc.sources}
                fe += len([c for c in r.proven_empty if c in src_ch])
                full += int(r.n_cleared == r.n_sources)
                cert += int(r.n_cleared == r.n_sources and not r.unresolved)
                ts.append(r.virtual_time_s)
                outs += eng.n_outlier
            rows.append({"p_outlier": p_out, "q": q, "n": args.n,
                         "full_clear": full / args.n, "false_empty": fe,
                         "certified": cert / args.n,
                         "virtual_time_mean": st.fmean(ts), "n_outlier": outs})
            print("%-8.2f %-3d %-9.3f %-9d %-9.3f %-10.0f %-9d"
                  % (p_out, q, full / args.n, fe, cert / args.n,
                     st.fmean(ts), outs), flush=True)
            del sc, r, eng
            gc.collect()
    tag = "q%d" % args.problem
    with open(os.path.join(TAB, "outlier_%s.csv" % tag), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(TAB, "outlier_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)
    print("结果:", os.path.join(TAB, "outlier_%s.csv" % tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
