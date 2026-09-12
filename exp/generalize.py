# -*- coding: utf-8 -*-
"""
泛化实验（G5）：逐个维度扫描，检验方法的适用范围

维度
----
k            干扰源个数 10/12/14/16
Rc           接收半径分布：均匀[1000,1500] / 固定1000 / 固定1500
dir_ratio    定向源占比 0 / 0.25 / 0.5 / 0.75 / 1.0
delta        示向度误差半幅 δ = 0.25/0.5/1/2 度
cell         可行域栅格边长 5/10/15/20 m

用法::
    python exp/generalize.py --n 20 --dim all
    python exp/generalize.py --n 20 --dim k
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics as st
import sys
import time
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.client import LocalRobotClient                       # noqa: E402
from robot.policy import Q3Config, Q3Policy                     # noqa: E402
from sim.arena import generate_scenario                         # noqa: E402
from sim.engine import ArenaEngine                              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tab")
os.makedirs(TAB, exist_ok=True)


def run_cell(seed: int, problem: int, *, k=None, rc=None, dir_ratio=0.0,
             delta=1.0, cell=10.0, max_virtual=60000.0) -> Dict:
    sc = generate_scenario(seed=seed, directional_ratio=dir_ratio, n_sources=k)
    if rc is not None:
        for s in sc.sources:
            s.recv_radius = float(rc)
    eng = ArenaEngine(sc, "T", max_virtual=max_virtual, svd_delta=delta,
                      idem_limit=200000)
    eng.arm()
    cl = LocalRobotClient(eng, "T")
    cfg = Q3Config(census_layout=("hex_ring" if problem == 4 else "ring"),
                   directional=(problem == 4), svd_delta=delta, grid_cell=cell)
    cfg.max_virtual_s = max_virtual - 1.0
    pol = Q3Policy(cl, cfg)
    try:
        res = pol.run()
    except Exception as e:                                        # noqa: BLE001
        return {"n": sc.n_sources, "cleared": 0, "ratio": 0.0, "certified": 0,
                "vt": eng.virtual_time, "avg": float("nan"), "err": repr(e)}
    return {"n": res.n_sources, "cleared": res.n_cleared, "ratio": res.clear_ratio,
            "certified": int(res.n_cleared == res.n_sources and not res.unresolved),
            "vt": res.virtual_time_s, "avg": res.avg_locate_clear_time_s, "err": None}


def sweep(name: str, values: List[Tuple[str, Dict]], problem: int, n: int,
          seed0: int) -> List[Dict]:
    out = []
    for label, kw in values:
        rows = [run_cell(seed0 + i, problem, **kw) for i in range(n)]
        ratio = [r["ratio"] for r in rows]
        cert = [r["certified"] for r in rows]
        t = [r["avg"] for r in rows if r["cleared"] > 0]
        rec = {"dim": name, "value": label,
               "clear_ratio": st.fmean(ratio),
               "certified": st.fmean(cert),
               "avg_time": st.fmean(t) if t else float("nan"),
               "avg_time_std": st.pstdev(t) if len(t) > 1 else 0.0,
               "total_time": st.fmean([r["vt"] for r in rows]),
               "n_sources_mean": st.fmean([r["n"] for r in rows])}
        out.append(rec)
        print("    %-14s 清除比例=%.3f 认证=%.3f 平均时间=%s 总时间=%.0f"
              % (label, rec["clear_ratio"], rec["certified"],
                 ("%.1f±%.1f" % (rec["avg_time"], rec["avg_time_std"]))
                 if t else "—", rec["total_time"]), flush=True)
    return out


DIMS = {
    "k": ("干扰源个数 k", [(str(v), {"k": v}) for v in (10, 12, 14, 16)]),
    "rc": ("接收半径分布", [("uniform[1000,1500]", {"rc": None}),
                            ("fixed 1000 (最坏)", {"rc": 1000.0}),
                            ("fixed 1500 (最好)", {"rc": 1500.0})]),
    "dir": ("定向源占比", [(str(v), {"dir_ratio": v})
                           for v in (0.0, 0.25, 0.5, 0.75, 1.0)]),
    "delta": ("示向度误差 δ(度)", [(str(v), {"delta": v})
                                   for v in (0.25, 0.5, 1.0, 2.0)]),
    "cell": ("栅格边长 (m)", [(str(v), {"cell": v}) for v in (5.0, 10.0, 15.0, 20.0)]),
}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    ap.add_argument("--dim", default="all")
    args = ap.parse_args(argv)

    dims = list(DIMS) if args.dim == "all" else [args.dim]
    rows: List[Dict] = []
    for d in dims:
        title, values = DIMS[d]
        print("=" * 90)
        print("维度：%s（问题 %d，各 %d 局）" % (title, args.problem, args.n))
        print("=" * 90)
        t0 = time.monotonic()
        rows += sweep(d, values, args.problem, args.n, args.seed0)
        print("  用时 %.1f s" % (time.monotonic() - t0), flush=True)

    tag = "q%d" % args.problem
    csv_path = os.path.join(TAB, "generalize_%s.csv" % tag)
    # 合并：保留本次未扫描的维度，覆盖本次扫描的维度（避免多次运行互相覆盖）
    done = {r["dim"] for r in rows}
    old_rows = []
    if os.path.exists(csv_path):
        old_rows = [r for r in csv.DictReader(open(csv_path, encoding="utf-8-sig"))
                    if r["dim"] not in done]
    all_rows = old_rows + rows
    order = list(DIMS)
    all_rows.sort(key=lambda r: (order.index(r["dim"]) if r["dim"] in order else 99,
                                 str(r["value"])))
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)
    with open(os.path.join(TAB, "generalize_%s.json" % tag), "w",
              encoding="utf-8") as f:
        json.dump(all_rows, f, ensure_ascii=False, indent=1)
    print()
    print("结果:", os.path.join(TAB, "generalize_%s.csv" % tag))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
