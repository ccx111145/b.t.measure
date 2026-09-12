# -*- coding: utf-8 -*-
"""绘图统一样式（中文标题 + 论文级输出）"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figs")
TABDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tab")

C = {
    "s1": "#1f77b4",
    "s2": "#2ca02c",
    "src": "#d62728",
    "region": "#ff7f0e",
    "wedge1": "#1f77b4",
    "wedge2": "#2ca02c",
    "grid": "#bbbbbb",
    "ok": "#2ca02c",
    "bad": "#d62728",
    "hl": "#9467bd",
}


def setup():
    plt.rcParams.update({
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "figure.autolayout": False,
    })
    for d in (FIGDIR, TABDIR):
        os.makedirs(d, exist_ok=True)


def save(fig, name: str):
    import gc
    gc.collect()
    path = os.path.join(FIGDIR, name)
    fig.savefig(path)
    plt.close(fig)
    gc.collect()
    print("  图已保存:", path)
    return path


def draw_poly(ax, poly, **kw):
    if not poly:
        return
    xs = [p[0] for p in poly] + [poly[0][0]]
    ys = [p[1] for p in poly] + [poly[0][1]]
    ax.plot(xs, ys, **kw)


def fill_poly(ax, poly, **kw):
    if not poly:
        return
    ax.fill([p[0] for p in poly], [p[1] for p in poly], **kw)


def draw_arena(ax, radius=1800.0, label="目标区域 (R=1800 m)"):
    th = [i * 2 * 3.141592653589793 / 360 for i in range(361)]
    ax.plot([radius * __import__("math").cos(t) for t in th],
            [radius * __import__("math").sin(t) for t in th],
            color=C["grid"], lw=1.4, ls="--", label=label, zorder=1)
