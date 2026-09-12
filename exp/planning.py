# -*- coding: utf-8 -*-
"""
① 规划策略隔离对照：**固定证据集与证书**，只替换规划策略

每个变体只改一个规划选择，证据集推断、可清除判据、证否证书完全不变。
因此各变体之间的差异**只能**归因于规划。

  Full                tsp 访问顺序 / 最近优先频道 / 质心点估计（完整方法）
  P1-sequential       巡测点按生成顺序访问（不做 TSP）
  P2-random           巡测点随机顺序
  P3-roundrobin       频道轮转处理（不看距离）
  P4-uncertainty      优先处理可行域最大的频道
  P5-bbox             用可行域外接盒中心作点估计（而非质心）

用法: python exp/planning.py --n 30 --problem 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.policy import Q3Config, run_episode          # noqa: E402
from sim.arena import generate_scenario                 # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tab")
os.makedirs(TAB, exist_ok=True)

VARIANTS = [
    ("Full", {}),
    ("P1-sequential", {"plan_visit_order": "sequential"}),
    ("P2-random", {"plan_visit_order": "random"}),
    ("P3-roundrobin", {"plan_channel_order": "roundrobin"}),
    ("P4-uncertainty", {"plan_channel_order": "uncertainty"}),
    ("P5-bbox", {"plan_clear_target": "bbox"}),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    args = ap.parse_args(argv)

    ratio = 1.0 if args.problem == 4 else 0.0
    base = dict(census_layout=("hex_ring" if args.problem == 4 else "ring"),
                directional=(args.problem == 4))
    rows = []
    print("%-16s %9s %9s %14s %12s %8s" %
          ("变体", "清除比例", "认证率", "平均定位清除时间", "总虚拟时间", "动作数"))
    for name, over in VARIANTS:
        cfg_kw = dict(base)
        cfg_kw.update(over)
        t0 = time.monotonic()
        recs = []
        for i in range(args.n):
            sc = generate_scenario(seed=args.seed0 + i, directional_ratio=ratio)
            r, eng = run_episode(sc, "T", Q3Config(**cfg_kw))
            recs.append(r)
            del sc, r, eng
        cr = st.fmean([r.clear_ratio for r in recs])
        cert = st.fmean([1.0 if (r.n_cleared == r.n_sources and not r.unresolved)
                         else 0.0 for r in recs])
        tt = [r.avg_locate_clear_time_s for r in recs if r.n_cleared > 0]
        vt = st.fmean([r.virtual_time_s for r in recs])
        ac = st.fmean([r.n_actions for r in recs])
        rows.append({"variant": name, "n": args.n, "clear_ratio": cr,
                     "certified": cert,
                     "avg_locate_clear_time_s": st.fmean(tt) if tt else None,
                     "avg_time_std": st.pstdev(tt) if len(tt) > 1 else 0.0,
                     "virtual_time_mean": vt, "actions_mean": ac,
                     "wall_s": time.monotonic() - t0})
        print("%-16s %9.3f %9.3f %14s %12.0f %8.0f"
              % (name, cr, cert,
                 ("%.1f±%.1f" % (st.fmean(tt), st.pstdev(tt))) if tt else "-",
                 vt, ac), flush=True)

    tag = "q%d" % args.problem
    with open(os.path.join(TAB, "planning_%s.csv" % tag), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(TAB, "planning_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=1)

    full = rows[0]
    print()
    print("相对完整方法的代价变化（固定证据集与证书，只改规划）：")
    for r in rows[1:]:
        d = r["virtual_time_mean"] - full["virtual_time_mean"]
        print("  %-16s %+8.0f s (%+.1f%%)，动作 %+.0f"
              % (r["variant"], d, 100 * d / full["virtual_time_mean"],
                 r["actions_mean"] - full["actions_mean"]))
    print("结果:", os.path.join(TAB, "planning_%s.csv" % tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
