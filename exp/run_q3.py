# -*- coding: utf-8 -*-
"""
问题3 批量演练：跑 N 局，统计题目要求的两个指标，并做参数灵敏度

用法::
    python exp/run_q3.py --n 100 --seed0 1
    python exp/run_q3.py --n 60 --sweep census_radius 900 1000 1100 1200 1400
    python exp/run_q3.py --n 1 --trace 1 --verbose          # 单局全流程跟踪
    python exp/run_q3.py --n 60 --strategy fused
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
from dataclasses import asdict
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.policy import Q3Config, Q3Policy, run_episode      # noqa: E402
from robot.client import LocalRobotClient                     # noqa: E402
from sim.arena import generate_scenario                       # noqa: E402
from sim.engine import ArenaEngine                            # noqa: E402

TABDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tab")
os.makedirs(TABDIR, exist_ok=True)


def _pct(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    k = (len(ys) - 1) * q
    lo, hi = int(math.floor(k)), int(math.ceil(k))
    return ys[lo] if lo == hi else ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


def summarize(results: List) -> Dict:
    ok = [r for r in results]
    ratios = [r.clear_ratio for r in ok]
    avg_t = [r.avg_locate_clear_time_s for r in ok if r.n_cleared > 0]
    vt = [r.virtual_time_s for r in ok]
    acts = [r.n_actions for r in ok]
    reqs = [r.n_requests for r in ok]
    walls = [r.wall_s for r in ok]
    return {
        "n_episodes": len(ok),
        "full_clear_rate": sum(1 for r in ok if r.n_cleared == r.n_sources) / max(1, len(ok)),
        "clear_ratio_mean": stat.fmean(ratios) if ratios else float("nan"),
        "clear_ratio_min": min(ratios) if ratios else float("nan"),
        "avg_time_mean": stat.fmean(avg_t) if avg_t else float("nan"),
        "avg_time_std": (stat.pstdev(avg_t) if len(avg_t) > 1 else 0.0),
        "avg_time_p50": _pct(avg_t, 0.50),
        "avg_time_p90": _pct(avg_t, 0.90),
        "avg_time_max": max(avg_t) if avg_t else float("nan"),
        "virtual_time_mean": stat.fmean(vt) if vt else float("nan"),
        "virtual_time_p90": _pct(vt, 0.90),
        "actions_mean": stat.fmean(acts) if acts else float("nan"),
        "actions_max": max(acts) if acts else 0,
        "requests_mean": stat.fmean(reqs) if reqs else float("nan"),
        "requests_max": max(reqs) if reqs else 0,
        "wall_total_s": sum(walls),
        "wall_mean_s": stat.fmean(walls) if walls else float("nan"),
        "unresolved_total": sum(len(r.unresolved) for r in ok),
        "proven_empty_mean": stat.fmean([len(r.proven_empty) for r in ok]) if ok else float("nan"),
    }


def run_batch(n: int, seed0: int, cfg: Q3Config, directional_ratio: float = 0.0,
              min_sep: float = 0.0, verbose: bool = False,
              progress: int = 0) -> List:
    import gc
    out = []
    for i in range(n):
        sc = generate_scenario(seed=seed0 + i, directional_ratio=directional_ratio,
                               min_sep=min_sep)
        try:
            res, eng = run_episode(sc, "TEAM-LOCAL", cfg)
        except MemoryError:
            print("  [警告] 第 %d 局内存不足，提前结束（已完成 %d 局）"
                  % (i + 1, len(out)))
            break
        out.append(res)
        del eng, sc
        if (i + 1) % 10 == 0:
            gc.collect()
        if progress and (i + 1) % progress == 0:
            print("    ... 已完成 %d/%d 局" % (i + 1, n), flush=True)
    gc.collect()
    return out


def write_csv(results: List, tag: str, cfg: Q3Config):
    path = os.path.join(TABDIR, "q3_runs_%s.csv" % tag)
    rows = [asdict(r) for r in results]
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["unresolved"] = " ".join(str(x) for x in r["unresolved"])
            r["proven_empty"] = " ".join(str(x) for x in r["proven_empty"])
            w.writerow(r)
    print("  逐局结果:", path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--directional-ratio", type=float, default=0.0)
    ap.add_argument("--min-sep", type=float, default=0.0)
    ap.add_argument("--strategy", default="pipeline", choices=["pipeline", "fused"])
    ap.add_argument("--census-layout", default="ring", choices=["ring", "hex", "hex_ring"])
    ap.add_argument("--hex-step", type=float, default=1000.0)
    ap.add_argument("--hex-extent", type=float, default=2800.0)
    ap.add_argument("--census-radius", type=float, default=1200.0)
    ap.add_argument("--ring-n", type=int, default=6)
    ap.add_argument("--grid-cell", type=float, default=10.0)
    ap.add_argument("--homing-step-max", type=float, default=1100.0)
    ap.add_argument("--remeasure-spread", type=float, default=1e9)
    ap.add_argument("--fuse-radius", type=float, default=260.0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--trace", type=int, default=None, help="对某局打印全流程")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--sweep", nargs="+", default=None,
                    help="参数名 取值1 取值2 ... 做一维灵敏度")
    args = ap.parse_args(argv)

    tag = args.tag or ("%s_r%.0f_k%d" % (args.strategy, args.census_radius, args.ring_n))

    def make_cfg(**over):
        kw = dict(strategy=args.strategy, census_layout=args.census_layout,
                  hex_step=args.hex_step, hex_extent=args.hex_extent,
                  census_radius=args.census_radius,
                  census_ring_n=args.ring_n, grid_cell=args.grid_cell,
                  homing_step_max=args.homing_step_max,
                  census_remeasure_spread=args.remeasure_spread,
                  fuse_clear_radius=args.fuse_radius,
                  directional=(args.directional_ratio > 0))
        kw.update(over)
        return Q3Config(**kw)

    # ---- 单局跟踪 ----
    if args.trace is not None:
        sc = generate_scenario(seed=args.trace, directional_ratio=args.directional_ratio,
                               min_sep=args.min_sep)
        print("场景:", json.dumps(sc.summary(), ensure_ascii=False))
        print("真值:", [(s.channel, round(s.x), round(s.y), round(s.recv_radius))
                        for s in sc.sources])
        eng = ArenaEngine(sc, "TEAM-LOCAL")
        eng.arm()
        cl = LocalRobotClient(eng, "TEAM-LOCAL")
        pol = Q3Policy(cl, make_cfg(), verbose=True)
        res = pol.run()
        print(json.dumps(res.as_dict(), ensure_ascii=False, indent=1))
        return 0

    # ---- 一维灵敏度 ----
    if args.sweep:
        name = args.sweep[0]
        vals = [float(v) for v in args.sweep[1:]]
        table = []
        for v in vals:
            over = {name: (int(v) if name in ("census_ring_n",) else float(v))}
            t0 = time.time()
            rs = run_batch(args.n, args.seed0, make_cfg(**over),
                           args.directional_ratio, args.min_sep)
            s = summarize(rs)
            s[name] = v
            s["wall_s"] = time.time() - t0
            table.append(s)
            print("  %s=%-8s 满清除率=%.3f  平均时间=%.1f±%.1f s  动作=%.0f  用时=%.1fs"
                  % (name, v, s["full_clear_rate"], s["avg_time_mean"], s["avg_time_std"],
                     s["actions_mean"], s["wall_s"]))
        path = os.path.join(TABDIR, "q3_sweep_%s.csv" % name)
        keys = [name] + [k for k in table[0].keys() if k != name]
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in table:
                w.writerow(r)
        print("  灵敏度结果:", path)
        return 0

    # ---- 常规批量 ----
    t0 = time.time()
    results = run_batch(args.n, args.seed0, make_cfg(), args.directional_ratio,
                        args.min_sep, args.verbose)
    s = summarize(results)
    s["tag"] = tag
    s["wall_s"] = time.time() - t0
    s["config"] = asdict(make_cfg())
    print("=" * 78)
    print("问题3 演练结果  tag=%s  n=%d  seed0=%d" % (tag, args.n, args.seed0))
    print("=" * 78)
    print("  满清除率        : %.4f" % s["full_clear_rate"])
    print("  清除比例 均值/最小: %.4f / %.4f" % (s["clear_ratio_mean"], s["clear_ratio_min"]))
    print("  平均定位清除时间  : %.1f ± %.1f s  (P50 %.1f, P90 %.1f, max %.1f)"
          % (s["avg_time_mean"], s["avg_time_std"], s["avg_time_p50"],
             s["avg_time_p90"], s["avg_time_max"]))
    print("  定位清除总时间    : 均值 %.0f s, P90 %.0f s" % (s["virtual_time_mean"],
                                                              s["virtual_time_p90"]))
    print("  动作数            : 均值 %.1f, 最大 %d" % (s["actions_mean"], s["actions_max"]))
    print("  请求数            : 均值 %.1f, 最大 %d" % (s["requests_mean"], s["requests_max"]))
    print("  未结案频道总数    : %d" % s["unresolved_total"])
    print("  证明无源频道/局   : %.1f" % s["proven_empty_mean"])
    print("  本机用时          : %.1f s (%.1f ms/局)" % (s["wall_s"], 1000 * s["wall_mean_s"]))
    write_csv(results, tag, make_cfg())
    sp = os.path.join(TABDIR, "q3_stats_%s.json" % tag)
    with open(sp, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=1)
    print("  汇总:", sp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

