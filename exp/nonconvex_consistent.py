# -*- coding: utf-8 -*-
"""
Obstructed workspace: one evaluation, two tables.

Reproduces Table 8 (``tab:obstacles``) of the manuscript and the comparison table of
Section 6.11 from a single measurement, so that the two can never disagree.

The quantity measured
---------------------
For a network ``P`` and a workspace ``W``, the predicate is the detectability condition
of the manuscript with line-of-sight filtering,

    G  in  conv( V_G ),      V_G = { Q in P : |Q - G| <= R_min , LOS(Q, G) },

evaluated at every position of a **common** square grid of free-space positions.  The
reported number is the fraction of those positions at which the predicate is FALSE.

This is a pointwise evaluation on a grid, *not* the conservative cell certificate of
Section 5.2 of the manuscript.  The two answer different questions and are deliberately
not mixed here.

Output
  tab/nonconvex_eval.csv     one row per (workspace, network)

Usage
  python exp/nonconvex_consistent.py                # uses the saved repair network
  python exp/nonconvex_consistent.py --rebuild      # re-runs the greedy repair first
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "exp"))

from nonconvex import ARENA_R, detectable, greedy_augment, hex_lattice       # noqa: E402
from nonconvex_run import OBSTACLES, two_layer                               # noqa: E402

TAB = os.path.join(ROOT, "tab")
NET = os.path.join(TAB, "nonconvex_net.npy")
GRID_STEP = 80.0


def free_grid(step, obstacles):
    return [(x, y) for x in np.arange(-ARENA_R, ARENA_R + 1e-9, step)
            for y in np.arange(-ARENA_R, ARENA_R + 1e-9, step)
            if math.hypot(x, y) <= ARENA_R
            and not any(r.contains((x, y)) for r in obstacles)]


def evaluate(P, G, obstacles):
    """Return (n_failing, n_total, first_failing_position)."""
    n, first = 0, None
    for g in G:
        if not detectable(P, g, obstacles):
            n += 1
            if first is None:
                first = (round(g[0], 1), round(g[1], 1))
    return n, len(G), first


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true",
                    help="re-run the greedy shadow-filling before evaluating")
    ap.add_argument("--step", type=float, default=GRID_STEP)
    args = ap.parse_args(argv)

    P25 = two_layer()
    if args.rebuild or not os.path.exists(NET):
        print("rebuilding the greedy repair (this takes a few minutes) ...", flush=True)
        t0 = time.monotonic()
        P52 = greedy_augment(P25, OBSTACLES, h=60.0, max_add=40, cand_step=200.0)
        os.makedirs(TAB, exist_ok=True)
        np.save(NET, P52)
        print("  %d -> %d points  [%.0fs]" % (len(P25), len(P52), time.monotonic() - t0))
    else:
        P52 = np.load(NET)
    print("networks: %d-point disk-optimal, %d-point greedy repair"
          % (len(P25), len(P52)))

    G_obs = free_grid(args.step, OBSTACLES)
    G_free = free_grid(args.step, [])
    print("evaluation grid %.0f m: %d free positions with obstacles, %d without"
          % (args.step, len(G_obs), len(G_free)))

    ROWS = [
        ("no obstacles", "two-layer disk-optimal", P25, []),
        ("with obstacles", "two-layer disk-optimal", P25, OBSTACLES),
        ("with obstacles", "greedy shadow-filling", P52, OBSTACLES),
        ("with obstacles", "uniform hex, 600 m", hex_lattice(600.0, 2800.0, OBSTACLES), OBSTACLES),
        ("with obstacles", "uniform hex, 500 m", hex_lattice(500.0, 2800.0, OBSTACLES), OBSTACLES),
        ("with obstacles", "uniform hex, 450 m", hex_lattice(450.0, 2800.0, OBSTACLES), OBSTACLES),
    ]

    rows = []
    print()
    for world, name, P, obs in ROWS:
        G = G_free if not obs else G_obs
        t0 = time.monotonic()
        n, tot, first = evaluate(P, G, obs)
        rows.append({"workspace": world, "network": name, "points": len(P),
                     "positions": tot, "failing": n, "failing_fraction": n / tot,
                     "passing_fraction": 1.0 - n / tot,
                     "first_failure": "" if first is None else "%g,%g" % first})
        print("  %-15s %-22s %4d pts  failing %5d / %5d = %6.2f%%  first=%s  [%.0fs]"
              % (world, name, len(P), n, tot, 100 * n / tot, first or "---",
                 time.monotonic() - t0), flush=True)

    os.makedirs(TAB, exist_ok=True)
    path = os.path.join(TAB, "nonconvex_eval.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\nsaved", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
