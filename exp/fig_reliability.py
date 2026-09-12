# -*- coding: utf-8 -*-
"""
证书可靠性图（论文级矢量图）
  figs/reliability_map.pdf
  (a) 误证率热图：孔径 × Rician K
  (b) 同一数据的认证率热图
  (c) SNR 切片
  (d) 占空比切片
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib                                          # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tab")
FIG = os.path.join(ROOT, "figs")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.7,
})


def load(fn):
    return list(csv.DictReader(open(os.path.join(TAB, fn), encoding="utf-8-sig")))


def gridify(rows, field):
    aps = sorted({round(float(r["aperture"]), 2) for r in rows})
    ks = sorted({float(r["k_db"]) for r in rows})
    Z = np.full((len(ks), len(aps)), np.nan)
    for r in rows:
        i = ks.index(float(r["k_db"]))
        j = aps.index(round(float(r["aperture"]), 2))
        Z[i, j] = float(r[field])
    return aps, ks, Z


def heat(ax, aps, ks, Z, title, cmap, vmax=None, fmt="%.2f", thresh=None):
    im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap,
                   vmin=0.0, vmax=vmax if vmax else max(1e-6, np.nanmax(Z)))
    ax.set_xticks(range(len(aps)))
    ax.set_xticklabels(["%.0f" % a for a in aps])
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(["%.0f" % k for k in ks])
    ax.set_xlabel("Array aperture $M\\,R_\\lambda$")
    ax.set_ylabel("$\\Gamma_{\\mathrm{mp}}=20\\log_{10}(1/\\rho)$ (dB)")
    ax.set_title(title)
    for i in range(len(ks)):
        for j in range(len(aps)):
            v = Z[i, j]
            if np.isnan(v):
                continue
            shade = "white" if v > 0.5 * (vmax or np.nanmax(Z)) else "black"
            ax.text(j, i, fmt % v, ha="center", va="center", fontsize=6.5,
                    color=shade)
    if thresh is not None:
        ax.contour(range(len(aps)), range(len(ks)), Z, levels=[thresh],
                   colors="k", linewidths=1.2, linestyles="--")
    return im


def main():
    d = json.load(open(os.path.join(TAB, "reliability_all.json"), encoding="utf-8"))
    fig = plt.figure(figsize=(7.4, 4.9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.45, wspace=0.32)

    ax = fig.add_subplot(gs[0, 0])
    aps, ks, Z = gridify(d["map"], "false_absence_rate")
    im = heat(ax, aps, ks, Z, "(a) False-certificate rate", "YlOrRd",
              vmax=0.6, thresh=0.01)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = fig.add_subplot(gs[0, 1])
    aps2, ks2, Z2 = gridify(d["map"], "certified_rate")
    im = heat(ax, aps2, ks2, Z2, "(b) Certified-completion rate", "YlGn",
              vmax=1.0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)

    ax = fig.add_subplot(gs[1, 0])
    for a in sorted({round(float(r["aperture"]), 2) for r in d["snr"]}):
        rs = sorted([r for r in d["snr"] if round(float(r["aperture"]), 2) == a],
                    key=lambda r: float(r["snr_db"]))
        ax.plot([float(r["snr_db"]) for r in rs],
                100 * np.array([float(r["false_absence_rate"]) for r in rs]),
                "o-", ms=4, label="$M R_\\lambda=%.1f$" % a)
    ax.set_yscale("symlog", linthresh=0.1)
    ax.set_xlabel("SNR per element (dB)")
    ax.set_ylabel("False-certificate rate (%)")
    ax.set_title("(c) Dependence on SNR ($\\Gamma_{\\mathrm{mp}}=7$ dB)")
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    duty = d["duty"]
    ax.plot([float(r["duty"]) for r in duty],
            100 * np.array([float(r["false_absence_rate"]) for r in duty]),
            "s-", color="#c0392b", ms=4.5, label="False certificate")
    ax.plot([float(r["duty"]) for r in duty],
            100 * np.array([float(r["certified_rate"]) for r in duty]),
            "o-", color="#2e8b57", ms=4.5, label="Certified completion")
    ax.set_xlabel("Duty cycle $p_{\\mathrm{on}}$")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("(d) Intermittent transmission")
    ax.legend(fontsize=6.5)
    ax.grid(alpha=0.3)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, "reliability_map.%s" % ext),
                    format=ext, bbox_inches="tight", dpi=300)
    print("图已保存 figs/reliability_map.pdf")


if __name__ == "__main__":
    main()
