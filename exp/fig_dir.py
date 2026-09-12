# -*- coding: utf-8 -*-
"""
关键对照图：定向源占比 vs 网络结构
  figs/en_dir.png
  (a) 清除比例：环状网（S1 配置）随定向占比崩塌，两层密集网（S2 配置）保持 1.000
  (b) 认证完成率
  (c) 每源平均时间
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.plot_style import C, TABDIR, save, setup          # noqa: E402
import matplotlib.pyplot as plt                            # noqa: E402

setup()
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "SimHei"]


def rows_of(tag, dim="dir"):
    p = os.path.join(TABDIR, "generalize_%s.csv" % tag)
    return [r for r in csv.DictReader(open(p, encoding="utf-8-sig"))
            if r["dim"] == dim]


def main():
    r1, r2 = rows_of("q3"), rows_of("q4")
    x = np.arange(len(r1))
    lab = [r["value"] for r in r1]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    for ax, key, ylab in ((axes[0], "clear_ratio", "Cleared fraction"),
                          (axes[1], "certified", "Certified completion"),
                          (axes[2], "avg_time", "Cost per emitter (s)")):
        y1 = [float(r[key]) for r in r1]
        y2 = [float(r[key]) for r in r2]
        ax.plot(x, y1, "o-", color=C["s2"], ms=4, lw=1.5,
                label="Single-layer ring network (7 pts)")
        ax.plot(x, y2, "s-", color=C["s1"], ms=4, lw=1.5,
                label="Two-layer dense network (25 pts)")
        ax.set_xticks(x)
        ax.set_xticklabels(lab, fontsize=7)
        ax.set_xlabel("Fraction of directional emitters", fontsize=7.5)
        ax.set_ylabel(ylab, fontsize=7.5)
        ax.tick_params(labelsize=7)
        if key != "avg_time":
            ax.set_ylim(-0.02, 1.08)
    axes[2].set_ylim(0, None)
    axes[0].legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    save(fig, "en_dir.png")


if __name__ == "__main__":
    main()
