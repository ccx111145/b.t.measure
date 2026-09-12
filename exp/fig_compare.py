# -*- coding: utf-8 -*-
"""
基线对照图
  figs/cmp_clear_ratio.png   清除比例 / 认证完成率 对照（问题3、问题4，两个预算档）
  figs/cmp_time.png          平均定位清除时间与总虚拟时间分布
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.plot_style import C, TABDIR, save, setup        # noqa: E402
import matplotlib.pyplot as plt                          # noqa: E402

setup()
FILES = [
    ("q3", "问题 3（全向源）", "compare_q3.csv"),
    ("q4", "问题 4（全定向）", "compare_q4.csv"),
]
ORDER = ["Ours", "Lawnmower", "RandomWaypoints", "HomingOnly"]
LABEL = {"Ours": "本文方法", "Lawnmower": "犁地扫描+归航",
         "RandomWaypoints": "随机航点+归航", "HomingOnly": "纯测向归航"}
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
        if not rs:
            continue
        out[st] = {
            "cr": np.array([float(r["clear_ratio"]) for r in rs]),
            "ct": np.array([float(r["certified"]) for r in rs]),
            "vt": np.array([float(r["virtual_time_s"]) for r in rs]),
            "t": np.array([float(r["avg_locate_clear_time_s"]) for r in rs
                           if r["n_cleared"] != "0" and
                           r["avg_locate_clear_time_s"] not in ("", "nan")]),
        }
    return out


def fig_ratio():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.6))
    for ax, (tag, title, fn) in zip(axes, FILES):
        d = load(fn)
        if not d:
            ax.set_title(title + "（无数据）")
            continue
        xs = np.arange(len(ORDER))
        w = 0.36
        cr = [d[s]["cr"].mean() if s in d else 0 for s in ORDER]
        ct = [d[s]["ct"].mean() if s in d else 0 for s in ORDER]
        ax.bar(xs - w / 2, [100 * v for v in cr], w, color=[COLOR[s] for s in ORDER],
               alpha=0.9, label="清除比例")
        ax.bar(xs + w / 2, [100 * v for v in ct], w, color=[COLOR[s] for s in ORDER],
               alpha=0.45, hatch="//", label="认证完成率")
        for i, (a, b) in enumerate(zip(cr, ct)):
            ax.annotate("%.0f%%" % (100 * a), (i - w / 2, 100 * a), ha="center",
                        va="bottom", fontsize=8.5)
            ax.annotate("%.0f%%" % (100 * b), (i + w / 2, 100 * b), ha="center",
                        va="bottom", fontsize=8.5, color="#444")
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL[s] for s in ORDER], fontsize=8.5)
        ax.set_ylabel("比例 (%)")
        ax.set_ylim(0, 118)
        ax.set_title("%s（各 30 局，虚拟时间预算 60000 s，为本文方法的 13 倍）" % title)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("与基线的对照：本文方法在同等信息下同时取得最高的清除比例"
                 "与唯一的“认证完成”（可证明做完了）", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "cmp_clear_ratio.png")


def fig_time():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.4))
    for ax, (tag, title, fn) in zip(axes, FILES):
        d = load(fn)
        if not d:
            ax.set_title(title + "（无数据）")
            continue
        data, labels, colors = [], [], []
        for s in ORDER:
            if s not in d or len(d[s]["vt"]) == 0:
                continue
            data.append(d[s]["vt"])
            labels.append(LABEL[s])
            colors.append(COLOR[s])
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False)
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("k")
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylabel("单局定位清除总时间 (s)")
        ax.axhline(60000, color="#888", ls="--", lw=1.0)
        ax.annotate("预算上限 60000 s", (0.5, 60000), textcoords="offset points",
                    xytext=(2, -12), fontsize=7.5, color="#666")
        ax.set_title("%s：总虚拟时间分布" % title)
    fig.tight_layout()
    save(fig, "cmp_time.png")


if __name__ == "__main__":
    print("对照图 ...")
    fig_ratio()
    fig_time()
    print("完成")
