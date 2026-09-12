# -*- coding: utf-8 -*-
"""
证书可靠性图：从信道/传感器条件到"误证率"的定量映射

测量学问题
----------
给定 Rician K 因子、阵列孔径 M·R_lambda、每阵元 SNR、信标占空比 p_on，
"把有源判为无源"（误证，false absence）的概率是多少？

输出
  tab/reliability_map.csv     主图：孔径 × K 因子
  tab/reliability_snr.csv     SNR 切片
  tab/reliability_duty.csv    占空比切片
  figs/reliability_map.pdf    图

约定
  * 误证率 = 被错误认证为无源的源数 / 源总数（**唯一不可恢复的错误**）
  * 认证率 = 全部源被清除且无未结案的局数占比
  * 示向度误差界取固定 delta = 5 度（一个合理的工程设计值）
"""
from __future__ import annotations

import argparse
import csv
import json
import math
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

# 孔径 = M * R_lambda；保持阵元间距 2*pi*R/M <= 0.5 波长以避免栅瓣模糊
APERTURES = [(8, 0.20), (8, 0.35), (8, 0.60), (16, 0.90), (16, 1.20)]
KS_DB = [3.0, 5.0, 7.0, 9.0, 11.0]          # rho = 10^(-K/20)


def run_cell(m, r_wl, k_db, snr_db, duty, n, seed0, delta=5.0, problem=4):
    rho = 10 ** (-k_db / 20.0)
    engine = {"rf_chain": {"array_m": m, "array_radius_wl": r_wl,
                           "n_multipath": 2, "rho": rho,
                           "snr_db": snr_db, "n_snapshots": 48}}
    if duty < 1.0:
        engine["duty_cycle"] = duty
        engine["duty_slot_s"] = 20.0
    cfg = Q3Config(census_layout="hex_ring", directional=True, svd_delta=delta)
    n_src = n_false = n_clear = n_cert = 0
    tt = []
    for i in range(n):
        sc = generate_scenario(seed=seed0 + i, directional_ratio=1.0)
        r, eng = run_episode(sc, "T", cfg, engine_kwargs=engine)
        src_ch = {s.channel for s in sc.sources}
        n_src += r.n_sources
        n_false += len([c for c in r.proven_empty if c in src_ch])
        n_clear += r.n_cleared
        n_cert += int(r.n_cleared == r.n_sources and not r.unresolved)
        tt.append(r.virtual_time_s)
        del sc, r, eng
    return {"m": m, "r_wl": r_wl, "aperture": m * r_wl, "k_db": k_db, "rho": rho,
            "snr_db": snr_db, "duty": duty, "n_episodes": n, "n_sources": n_src,
            "false_absence": n_false,
            "false_absence_rate": n_false / max(1, n_src),
            "cleared_fraction": n_clear / max(1, n_src),
            "certified_rate": n_cert / n,
            "virtual_time_mean": st.fmean(tt)}


def sweep(tag, rows, fn):
    path = os.path.join(TAB, fn)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("  已保存", path)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args(argv)

    if args.pilot:
        t0 = time.monotonic()
        r = run_cell(8, 0.35, 7.0, 12.0, 1.0, 3, 1)
        dt = time.monotonic() - t0
        print("试跑 3 局用时 %.1f s ⟹ 每局 %.2f s" % (dt, dt / 3))
        print("  误证 %d / 源 %d，清除率 %.3f，认证率 %.3f"
              % (r["false_absence"], r["n_sources"], r["cleared_fraction"],
                 r["certified_rate"]))
        n_cells = len(APERTURES) * len(KS_DB) + len(APERTURES) * 3 + 4
        print("  预计全部 %d 个单元 × %d 局 ≈ %.0f 分钟"
              % (n_cells, args.n, n_cells * args.n * dt / 3 / 60))
        return 0

    # ---------------- 主图：孔径 × K 因子 ----------------
    print("=" * 84)
    print("主图：孔径 × Rician K 因子（SNR=12 dB，连续发射）")
    print("=" * 84)
    rows = []
    for (m, rw) in APERTURES:
        for k in KS_DB:
            t0 = time.monotonic()
            r = run_cell(m, rw, k, 12.0, 1.0, args.n, args.seed0)
            rows.append(r)
            print("  M=%2d R=%.2fλ (孔径 %5.1f)  K=%4.1f dB 误证率 %6.4f  清除 %.3f  认证 %.3f  [%.0fs]"
                  % (m, rw, m * rw, k, r["false_absence_rate"], r["cleared_fraction"],
                     r["certified_rate"], time.monotonic() - t0), flush=True)
    sweep("map", rows, "reliability_map.csv")

    # ---------------- SNR 切片 ----------------
    print()
    print("=" * 84)
    print("SNR 切片（K=7 dB，连续发射）")
    print("=" * 84)
    rows2 = []
    for (m, rw) in APERTURES:
        for snr in (6.0, 12.0, 20.0):
            r = run_cell(m, rw, 7.0, snr, 1.0, args.n, args.seed0)
            rows2.append(r)
            print("  M=%2d R=%.2fλ  SNR=%4.1f dB 误证率 %6.4f  清除 %.3f  认证 %.3f"
                  % (m, rw, snr, r["false_absence_rate"], r["cleared_fraction"],
                     r["certified_rate"]), flush=True)
    sweep("snr", rows2, "reliability_snr.csv")

    # ---------------- 占空比切片 ----------------
    print()
    print("=" * 84)
    print("占空比切片（M=16, R=1.2λ, K=7 dB, SNR=12 dB）")
    print("=" * 84)
    rows3 = []
    for duty in (1.0, 0.8, 0.6, 0.4):
        r = run_cell(16, 1.2, 7.0, 12.0, duty, args.n, args.seed0)
        rows3.append(r)
        print("  p_on=%.1f  误证率 %6.4f  清除 %.3f  认证 %.3f"
              % (duty, r["false_absence_rate"], r["cleared_fraction"],
                 r["certified_rate"]), flush=True)
    sweep("duty", rows3, "reliability_duty.csv")

    with open(os.path.join(TAB, "reliability_all.json"), "w",
              encoding="utf-8") as f:
        json.dump({"map": rows, "snr": rows2, "duty": rows3}, f,
                  ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
