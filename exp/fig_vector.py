# -*- coding: utf-8 -*-
"""
矢量版图表（PDF）+ 更大字号 + 去掉图内硬编码定理号

生成
  figs/en_net.pdf    图 1：判据示意 + 两种网络的可检出区域（子图标题不再写死定理号）
  figs/en_cmp.pdf    图 2：基线对照
  figs/en_gen.pdf    图 3：泛化 5 子图（放大字号）
  figs/en_dir.pdf    图 4：网络结构受控对照（放大字号）
"""
from __future__ import annotations

import csv
import itertools
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                             # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                               # noqa: E402
from matplotlib.patches import Circle, Polygon                # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tab")
FIG = os.path.join(ROOT, "figs")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
    "lines.linewidth": 1.4,
    "figure.dpi": 120,
})

C = {"s1": "#1f6fb4", "s2": "#2e8b57", "bad": "#c0392b", "hl": "#8e6fb0"}
R_LO, ARENA_R = 1000.0, 1800.0


def save(fig, name):
    for ext in ("pdf", "png"):
        p = os.path.join(FIG, name % ext)
        fig.savefig(p, format=ext, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("   保存 figs/%s" % (name % "pdf"))


# ------------------------------------------------------------------ 图 1
def hex_pts(step, extent):
    pts, dy = [], step * math.sqrt(3) / 2
    for j in range(-int(2 * extent / dy) - 2, int(2 * extent / dy) + 3):
        y = j * dy
        off = (step / 2) if (j % 2) else 0.0
        for i in range(-int(2 * extent / step) - 2, int(2 * extent / step) + 3):
            x = i * step + off
            if math.hypot(x, y) <= extent + 1e-9:
                pts.append((x, y))
    return np.array(pts)


def ring_net():
    return np.array([(0.0, 0.0)] + [(1200 * math.cos(2 * math.pi * i / 6),
                                     1200 * math.sin(2 * math.pi * i / 6))
                                    for i in range(6)])


def two_layer():
    inner = hex_pts(1000.0, 1800.0)
    outer = np.array([(1900 * math.cos(math.pi / 12 + 2 * math.pi * i / 12),
                       1900 * math.sin(math.pi / 12 + 2 * math.pi * i / 12))
                      for i in range(12)])
    return np.vstack([inner, outer])


def ok_angle(P, S):
    P32, S32 = P.astype(np.float32), S.astype(np.float32)
    dx = P32[None, :, 0] - S32[:, None, 0]
    dy = P32[None, :, 1] - S32[:, None, 1]
    d2 = dx * dx + dy * dy
    valid = d2 <= np.float32(R_LO * R_LO + 1e-3)
    ang = np.where(valid, np.arctan2(dy, dx), np.float32(np.inf))
    ang.sort(axis=1)
    cnt = valid.sum(axis=1)
    m = ang.shape[1]
    gaps = np.diff(ang, axis=1)
    gaps = np.where(np.isfinite(gaps), gaps, 0.0)
    mg = gaps.max(axis=1) if m > 1 else np.zeros(len(S), dtype=np.float32)
    first = ang[:, 0]
    last = ang[np.arange(len(S)), np.clip(cnt - 1, 0, m - 1)]
    wrap = np.where(cnt > 0, 2 * math.pi - (last - first), 2 * math.pi)
    mg = np.maximum(mg, wrap)
    ok = (cnt >= 2) & (mg <= math.pi + 1e-6)
    ok |= (d2.min(axis=1) <= np.float32(1e-6))
    return ok


def grid(step):
    xs = np.arange(-ARENA_R, ARENA_R + 1e-9, step)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    m = X * X + Y * Y <= ARENA_R ** 2
    return np.stack([X[m], Y[m]], 1)


def fig_net():
    G = grid(20.0)
    fig = plt.figure(figsize=(7.4, 2.7))
    ax1, ax2, ax3 = (fig.add_subplot(1, 3, i) for i in (1, 2, 3))
    for ax, P, title in ((ax1, ring_net(), "(a) Single-layer ring, 7 points"),
                         (ax2, two_layer(), "(b) Two-layer dense, 25 points")):
        ok = ok_angle(P, G)
        ax.scatter(G[~ok, 0], G[~ok, 1], s=1.4, c="#e06666", marker="s",
                   label="not detectable")
        ax.scatter(G[ok, 0], G[ok, 1], s=1.4, c="#6aa84f", marker="s",
                   label="detectable for all $d$")
        ax.add_patch(Circle((0, 0), ARENA_R, fill=False, lw=0.9, ec="k"))
        ax.plot(P[:, 0], P[:, 1], "o", ms=3.2, mfc="white", mec="k", mew=0.8)
        ax.set_title("%s\ndetectable fraction $=%.3f$" % (title, ok.mean()))
        ax.set_aspect("equal")
        ax.set_xlim(-2050, 2050); ax.set_ylim(-2050, 2050)
        ax.tick_params(labelsize=7)
        if ax is ax1:
            ax.legend(loc="upper right", markerscale=5, fontsize=6.5)
    G1 = np.array([-0.9, 0.55]); S1 = np.array([[-1.9, 0.1], [-0.4, 1.35], [-0.15, 0.0]])
    ax3.add_patch(Polygon(S1, closed=True, fc="#6aa84f", alpha=0.35, ec="#38761d"))
    ax3.plot(G1[0], G1[1], "*", ms=13, color="k")
    ax3.plot(S1[:, 0], S1[:, 1], "o", ms=6, mfc="white", mec="k")
    ax3.annotate("$G\\in\\mathrm{conv}(S_G)$", (-0.9, 0.28), fontsize=8,
                 ha="center", color="#38761d")
    G2 = np.array([0.95, 0.45]); S2 = np.array([[1.35, 0.05], [1.75, 0.75]])
    ax3.plot(G2[0], G2[1], "*", ms=13, color="k")
    ax3.plot(S2[:, 0], S2[:, 1], "o", ms=6, mfc="white", mec="k")
    a = math.radians(210)
    ax3.arrow(G2[0], G2[1], 1.0 * math.cos(a), 1.0 * math.sin(a),
              head_width=0.1, color="#cc0000", lw=1.4)
    ax3.annotate("$G\\notin\\mathrm{conv}(S_G)$", (1.15, -0.38), fontsize=8,
                 ha="center", color="#cc0000")
    ax3.text(0.0, -1.45,
             "red arrow: an orientation $d$ for which\nno point of $S_G$ is illuminated",
             fontsize=6.8, ha="center", color="#444")
    ax3.set_title("(c) Detectability condition")
    ax3.set_xlim(-2.2, 2.3); ax3.set_ylim(-1.72, 1.6)
    ax3.tick_params(labelsize=7)
    ax3.set_aspect("equal")
    fig.tight_layout()
    save(fig, "en_net.%s")


# ------------------------------------------------------------------ 图 2/3/4
ORDER = ["Ours", "Lawnmower", "RandomWaypoints", "HomingOnly"]
LABEL = {"Ours": "Proposed", "Lawnmower": "Lawnmower",
         "RandomWaypoints": "Random waypoints", "HomingOnly": "Homing only"}
COLOR = {"Ours": C["s2"], "Lawnmower": C["s1"], "RandomWaypoints": C["hl"],
         "HomingOnly": C["bad"]}


def fig_cmp():
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, (fn, title) in zip(axes, (("compare_q3.csv", "(a) Scenario S1 (omni)"),
                                      ("compare_q4.csv", "(b) Scenario S2 (directional)"))):
        rows = list(csv.DictReader(open(os.path.join(TAB, fn), encoding="utf-8-sig")))
        xs = np.arange(len(ORDER)); w = 0.36
        cr = [np.mean([float(r["clear_ratio"]) for r in rows if r["strategy"] == s])
              for s in ORDER]
        ct = [np.mean([float(r["certified"]) for r in rows if r["strategy"] == s])
              for s in ORDER]
        ax.bar(xs - w / 2, [100 * v for v in cr], w, color=[COLOR[s] for s in ORDER],
               label="Cleared fraction")
        ax.bar(xs + w / 2, [100 * v for v in ct], w, color=[COLOR[s] for s in ORDER],
               alpha=0.45, hatch="//", label="Certified")
        for i, v in enumerate(cr):
            ax.annotate("%.0f" % (100 * v), (i - w / 2, 100 * v), ha="center",
                        va="bottom", fontsize=7.5)
        ax.set_xticks(xs)
        ax.set_xticklabels([LABEL[s] for s in ORDER], fontsize=7.5, rotation=12)
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 120)
        ax.set_title(title)
        ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "en_cmp.%s")


DIM = [("k", "Number of emitters $k$"), ("rc", "Reception radius"),
       ("dir", "Fraction of directional"), ("delta", "Angular error $\\delta$ (deg)"),
       ("cell", "Raster cell (m)")]
XT = {"rc": ["uniform", "1000 m", "1500 m"]}


def fig_gen():
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.35))
    for ax, (dim, xlabel) in zip(axes, DIM):
        for tag, col, lab in (("q3", C["s2"], "S1"), ("q4", C["s1"], "S2")):
            rows = [r for r in csv.DictReader(
                open(os.path.join(TAB, "generalize_%s.csv" % tag), encoding="utf-8-sig"))
                if r["dim"] == dim]
            if not rows:
                continue
            xs = np.arange(len(rows))
            ax.errorbar(xs, [float(r["avg_time"]) for r in rows],
                        yerr=[float(r["avg_time_std"]) for r in rows],
                        marker="o", ms=3.4, capsize=2.5, color=col, label=lab)
        rows3 = [r for r in csv.DictReader(
            open(os.path.join(TAB, "generalize_q3.csv"), encoding="utf-8-sig"))
            if r["dim"] == dim]
        if rows3:
            ax.set_xticks(np.arange(len(rows3)))
            ax.set_xticklabels(XT.get(dim, [r["value"] for r in rows3]),
                               fontsize=7, rotation=25)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Cost per emitter (s)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(loc="best", fontsize=7)
    fig.tight_layout()
    save(fig, "en_gen.%s")


def fig_dir():
    def rows_of(tag):
        return [r for r in csv.DictReader(
            open(os.path.join(TAB, "generalize_%s.csv" % tag), encoding="utf-8-sig"))
            if r["dim"] == "dir"]
    r1, r2 = rows_of("q3"), rows_of("q4")
    x = np.arange(len(r1))
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for ax, key, ylab in ((axes[0], "clear_ratio", "Cleared fraction"),
                          (axes[1], "certified", "Certified completion"),
                          (axes[2], "avg_time", "Cost per emitter (s)")):
        ax.plot(x, [float(r[key]) for r in r1], "o-", color=C["s2"], ms=4.5,
                label="Single-layer ring (7 pts)")
        ax.plot(x, [float(r[key]) for r in r2], "s-", color=C["s1"], ms=4.5,
                label="Two-layer dense (25 pts)")
        ax.set_xticks(x)
        ax.set_xticklabels([r["value"] for r in r1], fontsize=7.5)
        ax.set_xlabel("Fraction of directional emitters")
        ax.set_ylabel(ylab)
        if key != "avg_time":
            ax.set_ylim(-0.02, 1.12)
    axes[2].set_ylim(0, None)
    axes[0].legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    save(fig, "en_dir.%s")


if __name__ == "__main__":
    print("矢量图导出 ...")
    fig_net()
    fig_cmp()
    fig_gen()
    fig_dir()
    print("完成")
