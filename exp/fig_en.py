# -*- coding: utf-8 -*-
"""
英文版实验图（供 SCI 稿使用）
  figs/en_cmp.png      基线对照：清除比例 / 认证完成率
  figs/en_abl.png      消融：两个场景
  figs/en_gen.png      泛化：5 个维度
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.plot_style import C, TABDIR, save, setup          # noqa: E402
import matplotlib.pyplot as plt                            # noqa: E402

setup()
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

ORDER = ["Ours", "Lawnmower", "RandomWaypoints", "HomingOnly"]
LABEL = {"Ours": "Proposed", "Lawnmower": "Lawnmower", "RandomWaypoints": "Random waypoints",
         "HomingOnly": "Homing only"}
COLOR = {"Ours": C["s2"], "Lawnmower": C["s1"], "RandomWaypoints": C["hl"],
         "HomingOnly": C["bad"]}


def load(fn):
    p = os.path.join(TABDIR, fn)
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    out = {}
    for st in ORDER:
        rs = [r for r in rows if r["strategy"] == st]
        if rs:
            out[st] = {"cr": np.array([float(r["clear_ratio"]) for r in rs]),
                       "ct": np.array([float(r["certified"]) for r in rs]),
                       "vt": np.array([float(r["virtual_time_s"]) for r in rs])}
    return out


def fig_cmp():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, (fn, title) in zip(axes, (("compare_q3.csv", "(a) Scenario S1 (omni)"),
                                      ("compare_q4.csv", "(b) Scenario S2 (directional)"))):
        d = load(fn)
        if not d:
            continue
        xs = np.arange(len(ORDER))
        w = 0.36
        cr = [d[s]["cr"].mean() if s in d else 0 for s in ORDER]
        ct = [d[s]["ct"].mean() if s in d else 0 for s in ORDER]
        ax.bar(xs - w / 2, [100 * v for v in cr], w, color=[COLOR[s] for s in ORDER],
               label="Cleared fraction")
        ax.bar(xs + w / 2, [100 * v for v in ct], w, color=[COLOR[s] for s in ORDER],
               alpha=0.45, hatch="//", label="Certified")
        for i, v in enumerate(cr):
            ax.annotate("%.0f" % (100 * v), (i - w / 2, 100 * v), ha="center",
                        va="bottom", fontsize=7)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL[s] for s in ORDER], fontsize=7, rotation=12)
        ax.set_ylabel("Percentage (%)", fontsize=8)
        ax.set_ylim(0, 118)
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=8.5)
        ax.legend(fontsize=6.5, loc="upper right")
    fig.tight_layout()
    save(fig, "en_cmp.png")


def fig_abl():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    SHORT = {"Full": "Full", "A1-no-feasible": "A1", "A2-no-mec": "A2",
             "A3-no-certify": "A3", "A4-ring-net": "A4", "A5-no-probe": "A5",
             "A6-no-sweep": "A6"}
    for ax, (tag, title) in zip(axes, (("q3", "(a) Scenario S1"), ("q4", "(b) Scenario S2"))):
        p = os.path.join(TABDIR, "ablation_%s_stats.json" % tag)
        if not os.path.exists(p):
            continue
        d = json.load(open(p, encoding="utf-8"))
        names = list(d.keys())
        cr = [100 * d[n]["clear_ratio_mean"] for n in names]
        ct = [100 * d[n]["certified_mean"] for n in names]
        vt = [d[n]["virtual_time_mean"] for n in names]
        xs = np.arange(len(names))
        w = 0.38
        ax.bar(xs - w / 2, cr, w, color=C["s2"], label="Cleared")
        ax.bar(xs + w / 2, ct, w, color=C["s1"], alpha=0.85, hatch="//",
               label="Certified")
        ax.set_xticks(xs)
        ax.set_xticklabels([SHORT.get(n, n) for n in names], fontsize=7)
        ax.set_ylabel("Percentage (%)", fontsize=8)
        ax.set_ylim(0, 118)
        ax.tick_params(labelsize=7)
        ax2 = ax.twinx()
        ax2.plot(xs, vt, "o--", color=C["bad"], ms=3.5, lw=1.2, label="Time")
        ax2.set_ylabel("Virtual time (s)", color=C["bad"], fontsize=8)
        ax2.tick_params(labelsize=7, colors=C["bad"])
        ax2.grid(False)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=6.5, loc="lower left")
        ax.set_title(title, fontsize=8.5)
    fig.tight_layout()
    save(fig, "en_abl.png")


DIM = [("k", "Number of emitters $k$"), ("rc", "Reception radius"),
       ("dir", "Fraction of directional"), ("delta", "Angular error $\\delta$ (deg)"),
       ("cell", "Raster cell (m)")]
XT = {"rc": ["uniform", "1000 m", "1500 m"]}


def fig_gen():
    fig, axes = plt.subplots(1, 5, figsize=(7.2, 2.1))
    for ax, (dim, xlabel) in zip(axes, DIM):
        for tag, col, lab in (("q3", C["s2"], "S1"), ("q4", C["s1"], "S2")):
            p = os.path.join(TABDIR, "generalize_%s.csv" % tag)
            if not os.path.exists(p):
                continue
            rows = [r for r in csv.DictReader(open(p, encoding="utf-8-sig"))
                    if r["dim"] == dim]
            if not rows:
                continue
            xs = np.arange(len(rows))
            ys = [float(r["avg_time"]) for r in rows]
            es = [float(r["avg_time_std"]) for r in rows]
            ax.errorbar(xs, ys, yerr=es, marker="o", ms=3, capsize=2, lw=1.2,
                        color=col, label=lab)
        p3 = os.path.join(TABDIR, "generalize_q3.csv")
        rows3 = [r for r in csv.DictReader(open(p3, encoding="utf-8-sig"))
                 if r["dim"] == dim] if os.path.exists(p3) else []
        if rows3:
            ax.set_xticks(np.arange(len(rows3)))
            ax.set_xticklabels(XT.get(dim, [r["value"] for r in rows3]), fontsize=6,
                               rotation=20)
        ax.set_xlabel(xlabel, fontsize=7)
        ax.set_ylabel("Cost per emitter (s)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    fig.tight_layout()
    save(fig, "en_gen.png")


if __name__ == "__main__":
    print("English figures ...")
    fig_cmp()
    fig_abl()
    fig_gen()
    print("done")
