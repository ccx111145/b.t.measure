# -*- coding: utf-8 -*-
"""
End-to-end runs with the array-processing chain in place of the abstract bearing model.

Reproduces Table 6 of the manuscript (``tab:rfrun``): the complete search-and-neutralize
method is run in scenario S2 (directional emitters) with the physical direction-finding
chain --- uniform circular array, specular multipath, snapshot covariance, MUSIC ---
feeding the evidence update, for five sensor/channel configurations.

Columns of ``tab/rf_end2end.csv``
  cleared          episodes in which every emitter was neutralized, out of ``--n``
  false_absence    channels wrongly certified absent (a count, summed over episodes)
  certified        episodes that terminated with a completion certificate
  virtual_time_s   mean virtual mission time

Usage
  python exp/rf_end2end.py --pilot      # one episode of the successful row, timing estimate
  python exp/rf_end2end.py              # full table (10 episodes per row)
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.policy import Q3Config, run_episode        # noqa: E402
from sim.arena import generate_scenario               # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tab")
os.makedirs(TAB, exist_ok=True)

SNR_DB = 12.0
N_SNAPSHOTS = 48
N_MULTIPATH = 2

# (label, M, R_lambda, rho, delta_deg); rho=None means a line-of-sight channel
CONFIGS = [
    ("M=8, R=0.35, rho=0.6", 8, 0.35, 0.6, 5.0),
    ("M=16, R=1.0, rho=0.6", 16, 1.0, 0.6, 5.0),
    ("M=16, R=1.0, rho=0.3", 16, 1.0, 0.3, 3.0),
    ("M=16, R=1.0, rho=0.3", 16, 1.0, 0.3, 5.0),
    ("M=16, R=1.0, LOS only", 16, 1.0, None, 5.0),
]


def run_row(m, r_wl, rho, delta, n, seed0):
    engine = {"rf_chain": {"array_m": m, "array_radius_wl": r_wl,
                           "n_multipath": 0 if rho is None else N_MULTIPATH,
                           "rho": 0.0 if rho is None else rho,
                           "snr_db": SNR_DB, "n_snapshots": N_SNAPSHOTS}}
    cfg = Q3Config(census_layout="hex_ring", directional=True, svd_delta=delta)
    n_false = n_clear = n_cert = 0
    tt = []
    for i in range(n):
        sc = generate_scenario(seed=seed0 + i, directional_ratio=1.0)
        r, eng = run_episode(sc, "T", cfg, engine_kwargs=engine)
        src_ch = {s.channel for s in sc.sources}
        n_false += len([c for c in r.proven_empty if c in src_ch])
        n_clear += int(r.n_cleared == r.n_sources and not r.unresolved)
        n_cert += int(r.n_cleared == r.n_sources and not r.unresolved)
        tt.append(r.virtual_time_s)
        del sc, r, eng
    return {"n_episodes": n, "cleared": n_clear, "false_absence": n_false,
            "certified": n_cert, "virtual_time_s": st.fmean(tt) if tt else 0.0}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args(argv)

    if args.pilot:
        lab, m, rw, rho, d = CONFIGS[3]
        t0 = time.monotonic()
        r = run_row(m, rw, rho, d, 1, args.seed0)
        dt = time.monotonic() - t0
        print("pilot %s delta=%.0f: cleared %d/1, false abs %d, %.0f s virtual, %.1f s wall"
              % (lab, d, r["cleared"], r["false_absence"], r["virtual_time_s"], dt))
        print("full table (%d rows x %d episodes) ~ %.0f min"
              % (len(CONFIGS), args.n, len(CONFIGS) * args.n * dt / 60))
        return 0

    rows = []
    for lab, m, rw, rho, d in CONFIGS:
        t0 = time.monotonic()
        r = run_row(m, rw, rho, d, args.n, args.seed0)
        row = {"sensor": lab, "rho": "" if rho is None else rho, "delta_deg": d}
        row.update(r)
        rows.append(row)
        print("%-24s delta=%3.0f  cleared %2d/%d  false abs %4d  certified %2d/%d  %8.0f s  [%.0fs]"
              % (lab, d, r["cleared"], args.n, r["false_absence"], r["certified"],
                 args.n, r["virtual_time_s"], time.monotonic() - t0), flush=True)

    path = os.path.join(TAB, "rf_end2end.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("saved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
