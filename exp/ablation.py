# -*- coding: utf-8 -*-
"""
消融实验（G3）：逐个移除本文方法的组件，量化每个组件的贡献

消融项
------
Full            完整方法
A1-no-feasible  去掉可行域质心估计 → 纯示向度定步长归航
A2-no-mec       去掉 MEC≤20 m 可清除判据 → "见到方向就试着清"
A3-no-certify   关闭证否论证（失去"证明做完了"的终止证书）
A4-ring-net     仅问题4：密集两层网 → 问题3 的环状网
A5-no-probe     关闭自适应探测点序列（定向源朝向背对时只能靠猜）
A6-no-sweep     关闭 /clear 栅格扫描兜底

用法::
    python exp/ablation.py --n 30 --problem 3
    python exp/ablation.py --n 30 --problem 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.client import LocalRobotClient                       # noqa: E402
from robot.policy import Q3Config, Q3Policy                     # noqa: E402
from sim.arena import generate_scenario                         # noqa: E402
from sim.engine import ArenaEngine                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tab")
os.makedirs(TAB, exist_ok=True)

ABLATIONS = [
    ("Full",            {}),
    ("A1-no-feasible",  {"ablate_centroid": True}),
    ("A2-no-mec",       {"ablate_mec": True}),
    ("A3-no-certify",   {"ablate_certify": True}),
    ("A5-no-probe",     {"ablate_probe": True}),
    ("A6-no-sweep",     {"ablate_sweep": True}),
]
ABLATIONS_Q4 = ABLATIONS[:1] + [("A4-ring-net", {"_ring": True})] + ABLATIONS[1:]


def base_cfg(problem: int) -> dict:
    return dict(census_layout=("hex_ring" if problem == 4 else "ring"),
                directional=(problem == 4))


def run_one(variant: str, over: Dict, seed: int, problem: int,
            max_virtual: float = 60000.0) -> Dict:
    ratio = 1.0 if problem == 4 else 0.0
    sc = generate_scenario(seed=seed, directional_ratio=ratio)
    eng = ArenaEngine(sc, "T", max_virtual=max_virtual, idem_limit=200000)
    eng.arm()
    cl = LocalRobotClient(eng, "T")
    kw = base_cfg(problem)
    over = dict(over)
    if over.pop("_ring", False):
        kw["census_layout"] = "ring"
    kw.update(over)
    cfg = Q3Config(**kw)
    cfg.max_virtual_s = max_virtual - 1.0
    pol = Q3Policy(cl, cfg)
    t0 = time.monotonic()
    try:
        res = pol.run()
    except Exception as e:                                        # noqa: BLE001
        return {"variant": variant, "seed": seed, "n_sources": sc.n_sources,
                "n_cleared": 0, "clear_ratio": 0.0, "certified": 0,
                "virtual_time_s": eng.virtual_time,
                "avg_locate_clear_time_s": float("nan"),
                "n_actions": 0, "error": repr(e)}
    return {"variant": variant, "seed": seed, "n_sources": res.n_sources,
            "n_cleared": res.n_cleared, "clear_ratio": res.clear_ratio,
            "certified": int(res.n_cleared == res.n_sources and not res.unresolved),
            "virtual_time_s": res.virtual_time_s,
            "avg_locate_clear_time_s": res.avg_locate_clear_time_s,
            "n_actions": res.n_actions, "error": None,
            "wall_s": time.monotonic() - t0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    ap.add_argument("--max-virtual", type=float, default=60000.0)
    args = ap.parse_args(argv)

    variants = ABLATIONS_Q4 if args.problem == 4 else ABLATIONS
    rows: List[Dict] = []
    for name, over in variants:
        t0 = time.monotonic()
        for i in range(args.n):
            rows.append(run_one(name, over, args.seed0 + i, args.problem,
                                args.max_virtual))
        print("  %-16s 完成 %d 局，用时 %.1f s"
              % (name, args.n, time.monotonic() - t0), flush=True)

    tag = "q%d" % args.problem
    with open(os.path.join(TAB, "ablation_%s.csv" % tag), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print()
    print("=" * 100)
    print("消融实验（问题 %d，各 %d 局，虚拟预算 %.0f s）" % (args.problem, args.n,
                                                              args.max_virtual))
    print("=" * 100)
    print("%-16s %10s %10s %16s %12s %10s" %
          ("变体", "清除比例", "认证完成率", "平均定位清除时间", "总虚拟时间", "动作数"))
    summary = {}
    full = None
    for name, _ in variants:
        rs = [r for r in rows if r["variant"] == name]
        cr = [r["clear_ratio"] for r in rs]
        ct = [r["certified"] for r in rs]
        t = [r["avg_locate_clear_time_s"] for r in rs if r["n_cleared"] > 0]
        vt = [r["virtual_time_s"] for r in rs]
        ac = [r["n_actions"] for r in rs]
        m = st.fmean(cr)
        if name == "Full":
            full = m
        print("%-16s %10.3f %10.3f %16s %12.0f %10.0f" %
              (name, m, st.fmean(ct),
               ("%.1f±%.1f" % (st.fmean(t), st.pstdev(t))) if t else "—",
               st.fmean(vt), st.fmean(ac)))
        summary[name] = {"clear_ratio_mean": m, "certified_mean": st.fmean(ct),
                         "avg_time_mean": st.fmean(t) if t else None,
                         "virtual_time_mean": st.fmean(vt),
                         "actions_mean": st.fmean(ac)}
    print()
    print("相对完整方法的清除比例下降：")
    for name, _ in variants:
        if name == "Full" or full is None:
            continue
        d = full - summary[name]["clear_ratio_mean"]
        print("  %-16s %+.4f  (%.0f%%)" % (name, -d, 100 * d / max(full, 1e-9)))
    with open(os.path.join(TAB, "ablation_%s_stats.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print("  汇总:", os.path.join(TAB, "ablation_%s_stats.json" % tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
