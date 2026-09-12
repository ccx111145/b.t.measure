# -*- coding: utf-8 -*-
"""
② 测向链路的误差特性：纯 LOS vs 多径

输出
  tab/rf_bearing_error.csv    各配置下的误差分位数与离群率
  figs/rf_error.png           误差分布直方图 + 离群率随多径强度变化
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.rf import ChannelConfig, DirectionFinder, UniformCircularArray   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tab")
FIG = os.path.join(ROOT, "figs")
os.makedirs(TAB, exist_ok=True)


def wrap_err(a, b):
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def sweep(n_trial=4000, seed=7):
    rng = np.random.default_rng(seed)
    arr = UniformCircularArray(m=8, radius_wl=0.35)
    rows, samples = [], {}
    configs = [
        ("LOS only, SNR 20 dB", dict(n_multipath=0, snr_db=20.0)),
        ("LOS only, SNR 12 dB", dict(n_multipath=0, snr_db=12.0)),
        ("1 multipath, SNR 12 dB", dict(n_multipath=1, rho=0.6, snr_db=12.0)),
        ("2 multipath, SNR 12 dB", dict(n_multipath=2, rho=0.6, snr_db=12.0)),
        ("3 multipath, SNR 12 dB", dict(n_multipath=3, rho=0.8, snr_db=12.0)),
    ]
    for name, kw in configs:
        df = DirectionFinder(arr, ChannelConfig(**kw))
        errs = []
        for _ in range(n_trial):
            th = rng.uniform(0, 360.0)
            est, share = df.estimate(math.radians(th), rng)
            errs.append(wrap_err(est, th))
        e = np.array(errs)
        rows.append({
            "config": name, "n": n_trial,
            "err_p50": float(np.percentile(e, 50)),
            "err_p90": float(np.percentile(e, 90)),
            "err_p99": float(np.percentile(e, 99)),
            "err_max": float(e.max()),
            "frac_le_1deg": float((e <= 1.0).mean()),
            "frac_gt_5deg": float((e > 5.0).mean()),
            "frac_gt_20deg": float((e > 20.0).mean()),
            "frac_gt_45deg": float((e > 45.0).mean()),
        })
        samples[name] = e
        print("%-24s p50=%5.2f p90=%6.2f p99=%7.2f  >5deg %5.2f%%  >20deg %5.2f%%  >45deg %5.2f%%"
              % (name, rows[-1]["err_p50"], rows[-1]["err_p90"], rows[-1]["err_p99"],
                 100 * rows[-1]["frac_gt_5deg"], 100 * rows[-1]["frac_gt_20deg"],
                 100 * rows[-1]["frac_gt_45deg"]), flush=True)
    return rows, samples


def figure(samples):
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, os.path.join(ROOT, "exp"))
    from plot_style import C, save, setup          # noqa: E402
    import matplotlib.pyplot as plt                # noqa: E402
    setup()
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "SimHei"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    ax = axes[0]
    names = list(samples)
    data = [samples[n] for n in names]
    bp = ax.boxplot(data, vert=True, widths=0.6, showfliers=False, patch_artist=True)
    for p in bp["boxes"]:
        p.set_facecolor(C["s2"]); p.set_alpha(0.7)
    ax.set_xticklabels([n.split(",")[0] for n in names], fontsize=6, rotation=15)
    ax.set_ylabel("Bearing error (deg, log)", fontsize=7.5)
    ax.set_yscale("log")
    ax.tick_params(labelsize=7)
    ax.set_title("(a) Error distribution of the DF chain", fontsize=8)

    ax = axes[1]
    for n in names:
        e = np.sort(samples[n])
        y = np.arange(1, len(e) + 1) / len(e)
        ax.plot(e, 1 - y, lw=1.2, label=n.split(",")[0])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Bearing error (deg)", fontsize=7.5)
    ax.set_ylabel("P(error > x)", fontsize=7.5)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=5.5)
    ax.set_title("(b) Tail probability (spurious bearings)", fontsize=8)
    fig.tight_layout()
    save(fig, "rf_error.png")


if __name__ == "__main__":
    print("② 测向链路误差特性 ...")
    rows, samples = sweep()
    with open(os.path.join(TAB, "rf_bearing_error.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("数据:", os.path.join(TAB, "rf_bearing_error.csv"))
    figure(samples)
