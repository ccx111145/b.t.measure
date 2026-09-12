# -*- coding: utf-8 -*-
"""
网络结构示意图（英文）
  figs/en_net.png
  (a) 七点单层环状网：区域中"任意朝向均可检出"的位置（蓝色）
  (b) 25 点两层密集网：同一判据下几乎全区域可检出
  (c) 检测完备性判据示意：G 在 conv(S_G) 内 vs G 不在 conv(S_G) 内
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.plot_style import C, save, setup                 # noqa: E402
import matplotlib.pyplot as plt                           # noqa: E402
from matplotlib.patches import Circle, Polygon            # noqa: E402

setup()
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "SimHei"]

R_LO, ARENA_R = 1000.0, 1800.0


def ok_angle(P, S):
    """G in conv(S_G) 判定（极角间隙法）"""
    P32, S32 = P.astype(np.float32), S.astype(np.float32)
    dx = P32[None, :, 0] - S32[:, None, 0]
    dy = P32[None, :, 1] - S32[:, None, 1]
    d2 = dx * dx + dy * dy
    valid = d2 <= np.float32(R_LO * R_LO + 1e-3)
    ang = np.where(valid, np.arctan2(dy, dx), np.float32(np.inf))
    ang.sort(axis=1)
    cnt = valid.sum(axis=1)
    m = ang.shape[1]
    gaps = np.where(np.isfinite(np.diff(ang, axis=1)), np.diff(ang, axis=1), 0.0)
    mg = gaps.max(axis=1) if m > 1 else np.zeros(len(S), dtype=np.float32)
    first = ang[:, 0]
    last = ang[np.arange(len(S)), np.clip(cnt - 1, 0, m - 1)]
    wrap = np.where(cnt > 0, 2 * math.pi - (last - first), 2 * math.pi)
    mg = np.maximum(mg, wrap)
    ok = (cnt >= 2) & (mg <= math.pi + 1e-6)
    ok |= (d2.min(axis=1) <= np.float32(1e-6))
    return ok


def ring_net():
    return np.array([(0.0, 0.0)] + [(1200 * math.cos(2 * math.pi * i / 6),
                                     1200 * math.sin(2 * math.pi * i / 6))
                                    for i in range(6)])


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


def two_layer():
    inner = hex_pts(1000.0, 1800.0)
    outer = np.array([(1900 * math.cos(math.pi / 12 + 2 * math.pi * i / 12),
                       1900 * math.sin(math.pi / 12 + 2 * math.pi * i / 12))
                      for i in range(12)])
    return np.vstack([inner, outer])


def grid(step=20.0):
    xs = np.arange(-ARENA_R, ARENA_R + 1e-9, step)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    X, Y = X.ravel(), Y.ravel()
    mk = X * X + Y * Y <= ARENA_R ** 2
    return np.stack([X[mk], Y[mk]], 1)


def main():
    G = grid(20.0)
    fig = plt.figure(figsize=(7.2, 2.55))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    for ax, P, title in ((ax1, ring_net(), "(a) Single-layer ring, 7 points"),
                         (ax2, two_layer(), "(b) Two-layer dense, 25 points")):
        ok = ok_angle(P, G)
        ax.scatter(G[~ok, 0], G[~ok, 1], s=1.2, c="#e06666", marker="s",
                   label="not detectable")
        ax.scatter(G[ok, 0], G[ok, 1], s=1.2, c="#6aa84f", marker="s",
                   label="detectable for all $d$")
        ax.add_patch(Circle((0, 0), ARENA_R, fill=False, lw=1.0, ec="k"))
        ax.plot(P[:, 0], P[:, 1], "o", ms=3.0, mfc="white", mec="k", mew=0.8)
        frac = ok.mean()
        ax.set_title("%s\ncoverage $=%.3f$" % (title, frac), fontsize=7.5)
        ax.set_aspect("equal")
        ax.set_xlim(-2000, 2000)
        ax.set_ylim(-2000, 2000)
        ax.tick_params(labelsize=6)
        if ax is ax1:
            ax.legend(fontsize=5.5, loc="upper right", markerscale=4)

    # (c) 判据示意
    ax3.set_aspect("equal")
    G1 = np.array([-0.9, 0.55])
    S1 = np.array([[-1.9, 0.1], [-0.4, 1.35], [-0.15, 0.0]])
    ax3.add_patch(Polygon(S1, closed=True, fc="#6aa84f", alpha=0.35, ec="#38761d"))
    ax3.plot(G1[0], G1[1], "*", ms=11, color="k")
    ax3.plot(S1[:, 0], S1[:, 1], "o", ms=5, mfc="white", mec="k")
    ax3.annotate("$G\\in\\mathrm{conv}(S_G)$", (-0.9, 0.35), fontsize=7,
                 ha="center", color="#38761d")
    G2 = np.array([0.95, 0.45])
    S2 = np.array([[1.35, 0.05], [1.75, 0.75]])
    ax3.plot(G2[0], G2[1], "*", ms=11, color="k")
    ax3.plot(S2[:, 0], S2[:, 1], "o", ms=5, mfc="white", mec="k")
    a = math.radians(210)
    ax3.arrow(G2[0], G2[1], 1.0 * math.cos(a), 1.0 * math.sin(a),
              head_width=0.09, color="#cc0000", lw=1.3)
    ax3.annotate("$G\\notin\\mathrm{conv}(S_G)$", (1.15, -0.35), fontsize=7,
                 ha="center", color="#cc0000")
    ax3.text(0.0, -1.35,
             "red arrow: an orientation $d$ for which\nno point of $S_G$ is illuminated",
             fontsize=5.8, ha="center", color="#555")
    ax3.set_title("(c) Detectability condition (Thm. 3)", fontsize=7.5)
    ax3.set_xlim(-2.2, 2.3)
    ax3.set_ylim(-1.6, 1.6)
    ax3.tick_params(labelsize=6)
    fig.tight_layout()
    save(fig, "en_net.png")


if __name__ == "__main__":
    main()
