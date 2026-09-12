# -*- coding: utf-8 -*-
"""
问题2 出图
  figs/q2_F1_candidate.png    (a) 源可行域 F1  (b) 候选区域（ψ,d 空间）
  figs/q2_worst_diam_map.png  最坏情况定位区域直径热力图 + 可行域边界 + 最优解
  figs/q2_pareto.png          移动时间 - 定位精度 帕累托前沿
并输出 tab/q2_*.csv
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.plot_style import C, TABDIR, draw_arena, draw_poly, fill_poly, save, setup  # noqa: E402
from robot import geometry as G      # noqa: E402
from robot import selector as SEL    # noqa: E402
import matplotlib.pyplot as plt      # noqa: E402

setup()
S1 = (0.0, 0.0)
TH1 = 0.0
F1 = SEL.F1Set(S1, TH1)


def _csv(name, header, rows):
    p = os.path.join(TABDIR, name)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("  表已保存:", p)


_CACHE = {}


def wd(psi, d, R_lo=1000.0):
    """带缓存的最坏直径（不可行返回 nan）"""
    key = (round(psi, 4), round(d, 4))
    if key in _CACHE:
        return _CACHE[key]
    S2 = (S1[0] + d * math.cos((TH1 + psi) * G.D2R),
          S1[1] + d * math.sin((TH1 + psi) * G.D2R))
    if F1.max_dist(S2) > R_lo + 1e-9:
        _CACHE[key] = float("nan")
        return float("nan")
    v = SEL.worst_diameter(S1, TH1, S2, F1)
    _CACHE[key] = v
    return v


# ================================================================ 图5
def fig_F1_candidate():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))

    # ---- (a) 空间中的 F1 与候选区域 ----
    ax = axes[0]
    draw_arena(ax)
    poly = F1.polygon()
    fill_poly(ax, poly, color=C["region"], alpha=0.45, zorder=3, label="源可行域 $F_1$")
    draw_poly(ax, poly, color=C["region"], lw=1.3, zorder=4)
    # 楔形边界
    for a in (TH1 - 1, TH1 + 1):
        ax.plot([S1[0], S1[0] + 1600 * math.cos(a * G.D2R)],
                [S1[1], S1[1] + 1600 * math.sin(a * G.D2R)],
                color=C["s1"], lw=1.0, ls=":", zorder=2)
    tt = np.linspace(0, 2 * math.pi, 400)
    ax.plot(S1[0] + 1500 * np.cos(tt), S1[1] + 1500 * np.sin(tt),
            color="#999", lw=1.0, ls="-.", label="$|S_1G|\\leq R_{\\max}=1500$ m")
    ax.plot(*S1, "o", color=C["s1"], ms=8, zorder=6)
    ax.annotate("$S_1$", (S1[0] + 25, S1[1] - 80), color=C["s1"], fontsize=13)
    for R_lo, col, ls in ((1000.0, C["s2"], "-"), (1250.0, C["hl"], "--")):
        reg = SEL.candidate_region(S1, TH1, R_lo)
        if len(reg["polygon"]) >= 3:
            ax.plot([p[0] for p in reg["polygon"]] + [reg["polygon"][0][0]],
                    [p[1] for p in reg["polygon"]] + [reg["polygon"][0][1]],
                    color=col, lw=2.0, ls=ls, zorder=5,
                    label="候选区域 $\\mathcal{C}$ ($R_{lo}$=%.0f m)" % R_lo)
        b = SEL.best_by(SEL.scan(S1, TH1, R_lo, np.arange(-60, 60.1, 2.5),
                                 np.arange(400, 1401, 25.0), F1))
        if b:
            ax.plot(*b["S2"], "*", color=col, ms=15, zorder=7,
                    label="$S_2^*$ ($R_{lo}$=%.0f: $\\psi$=%.0f°, $d$=%.0f m)" %
                          (R_lo, b["psi_deg"], b["d"]))
    ax.set_aspect("equal")
    ax.set_xlim(-2200, 2200)
    ax.set_ylim(-2000, 2000)
    ax.set_xlabel("x (m, 正东)")
    ax.set_ylabel("y (m, 正北)")
    ax.set_title("(a) 源可行域 $F_1$ 与第二检测点候选区域（空间视图）")
    ax.legend(loc="lower left", fontsize=7.5)

    # ---- (b) (ψ,d) 空间 ----
    ax = axes[1]
    for R_lo, col, ls in ((1000.0, C["s2"], "-"), (1250.0, C["hl"], "--"),
                          (1500.0, C["s1"], ":")):
        psis = np.arange(-89, 89.01, 0.5)
        d1s, d2s = [], []
        for p in psis:
            d1, d2 = SEL.analytic_d_feasible_range(p, R_lo)
            d1s.append(d1 if not math.isnan(d1) else np.nan)
            d2s.append(d2 if not math.isnan(d2) else np.nan)
        ax.plot(psis, d1s, color=col, lw=1.8, ls=ls)
        ax.plot(psis, d2s, color=col, lw=1.8, ls=ls,
                label="$R_{lo}$=%.0f m（精确 $\\psi_{\\max}$=%.2f°，闭式上界 %.2f°）"
                      % (R_lo, SEL.analytic_psi_max(R_lo), SEL.psi_max_upper(R_lo)))
        pm = SEL.analytic_psi_max(R_lo)
        ax.axvline(pm, color=col, lw=0.8, ls=":", alpha=0.7)
        ax.axvline(-pm, color=col, lw=0.8, ls=":", alpha=0.7)
    ax.set_xlabel("偏转角 $\\psi$ (°，相对示向度方向)")
    ax.set_ylabel("第二检测点距离 $d$ (m)")
    ax.set_ylim(300, 1600)
    ax.set_xlim(-70, 70)
    ax.set_title("(b) 候选区域在 $(\\psi,d)$ 空间是**曲边矩形**\n"
                 "上/下边界 = 近端/远端仍能测到信号的临界线")
    ax.legend(fontsize=7.5, loc="upper center")
    fig.tight_layout()
    save(fig, "q2_F1_candidate.png")


# ================================================================ 图6
def fig_worst_map():
    R_lo = 1000.0
    psis = np.arange(-45, 45.01, 2.5)
    ds = np.arange(450, 1150.1, 25.0)
    Z = np.full((len(psis), len(ds)), np.nan)
    for i, p in enumerate(psis):
        for j, d in enumerate(ds):
            Z[i, j] = wd(p, d, R_lo)

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    ax = axes[0]
    from matplotlib.colors import LogNorm
    im = ax.pcolormesh(ds, psis, Z, shading="auto", cmap="viridis_r",
                       norm=LogNorm(vmin=40, vmax=2e4))
    ax.contour(ds, psis, Z, levels=[40, 60, 100, 150, 300, 1000, 5000],
               colors="w", linewidths=0.7, alpha=0.85)
    cb = plt.colorbar(im, ax=ax, label="最坏情况定位区域直径 (m，对数色标)")
    # 可行域边界
    pgrid = np.arange(-60, 60.01, 0.5)
    d1s = [SEL.analytic_d_feasible_range(p, R_lo)[0] for p in pgrid]
    d2s = [SEL.analytic_d_feasible_range(p, R_lo)[1] for p in pgrid]
    ax.plot(d1s, pgrid, color="r", lw=2.0, label="可检测性边界（远支 $d_1$）")
    ax.plot(d2s, pgrid, color="m", lw=2.0, label="可检测性边界（近支 $d_2$）")
    best = SEL.best_by(SEL.scan(S1, TH1, R_lo, np.arange(-45, 45.01, 2.5), ds, F1))
    if best:
        ax.plot(best["d"], best["psi_deg"], "k*", ms=18,
                label="极小极大最优 $\\psi$=%.0f°, $d$=%.0f m, $D_{max}$=%.0f m"
                      % (best["psi_deg"], best["d"], best["worst_diam_m"]))
    ax.set_xlabel("第二检测点距离 $d$ (m)")
    ax.set_ylabel("偏转角 $\\psi$ (°)")
    ax.set_title("(a) 最坏情况定位区域直径 $\\max_{G,\\varepsilon}\\mathrm{diam}(\\mathcal{L})$ (m)\n"
                 "$R_{lo}=1000$ m；灰色区不可行；等值线 40/60/100/150/300/1000/5000 m")
    ax.set_facecolor("#dddddd")
    ax.legend(fontsize=7.5, loc="lower right")

    # ---- (b) 若干 psi 下的剖面 ----
    ax = axes[1]
    for psi in (-10, 0, 20, 40):
        ys = [wd(psi, d, R_lo) for d in ds]
        ax.plot(ds, ys, lw=1.8, marker=".", ms=3, label="$\\psi=%d°$" % psi)
    ax.axhline(40.0, color=C["bad"], ls="--", lw=1.2, label="40 m 精度阈值")
    ax.set_yscale("log")
    ax.set_ylim(30, 1e5)
    ax.set_xlabel("第二检测点距离 $d$ (m)")
    ax.set_ylabel("最坏定位区域直径 (m，对数刻度)")
    ax.set_title("(b) 固定 $\\psi$ 时最坏直径随 $d$ 的剖面\n"
                 "$\\psi$ 接近 0 时，远源从两站看方向近平行 → 区域退化成"
                 "数万米长条")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "q2_worst_diam_map.png")

    rows = []
    for i, p in enumerate(psis):
        for j, d in enumerate(ds):
            if not math.isnan(Z[i, j]):
                rows.append(["%.1f" % p, "%.0f" % d, "%.2f" % Z[i, j]])
    _csv("q2_worst_diam_map.csv", ["psi_deg", "d_m", "worst_diam_m"], rows)


# ================================================================ 图7
def fig_pareto():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    ax = axes[0]
    for R_lo, col in ((1000.0, C["s2"]), (1250.0, C["hl"]), (1500.0, C["s1"])):
        rows = SEL.scan(S1, TH1, R_lo, np.arange(-85, 85.1, 5.0),
                        np.arange(200, 2400.1, 100.0), F1)
        ok = [r for r in rows if r["feasible"] and r["worst_diam_m"] < SEL.INF]
        ax.scatter([r["move_time_s"] for r in ok], [r["worst_diam_m"] for r in ok],
                   s=9, alpha=0.35, color=col, label="$R_{lo}$=%.0f m 全部可行解" % R_lo)
        front = SEL.pareto_front(rows)
        if front:
            ax.plot([r["move_time_s"] for r in front], [r["worst_diam_m"] for r in front],
                    "-o", color=col, ms=4, lw=1.6, label="$R_{lo}$=%.0f m 帕累托前沿" % R_lo)
        b = SEL.best_by(rows)
        if b:
            ax.plot(b["move_time_s"], b["worst_diam_m"], "*", color=col, ms=17,
                    markeredgecolor="k", markeredgewidth=0.6)
    ax.axhline(40.0, color=C["bad"], ls="--", lw=1.2, label="40 m 精度阈值")
    ax.set_xlabel("移动到第二点的耗时 (s)")
    ax.set_ylabel("最坏情况定位区域直径 (m)")
    ax.set_ylim(0, 400)
    ax.set_title("(a) 移动代价 - 定位精度权衡（星号 = 极小极大最优）")
    ax.legend(fontsize=7.5)

    # ---- (b) 推荐解在空间中的位置对比 ----
    ax = axes[1]
    draw_arena(ax)
    for a in (TH1 - 1, TH1 + 1):
        ax.plot([0, 1700 * math.cos(a * G.D2R)], [0, 1700 * math.sin(a * G.D2R)],
                color=C["s1"], lw=1.0, ls=":")
    ax.plot(0, 0, "o", color=C["s1"], ms=8, zorder=6)
    ax.annotate("$S_1$", (20, -80), color=C["s1"], fontsize=12)
    marks = {"正侧方 90°, $d$=1000 m（常见错误答案）": (90.0, 1000.0, C["bad"], "X"),
             "同向 ψ=0°, $d$=1000 m（定位区域无界！）": (0.0, 1000.0, "#888888", "s"),
             "推荐解": (None, None, C["s2"], "*")}
    for lab, (psi, d, col, mk) in marks.items():
        if psi is None:
            b = SEL.best_by(SEL.scan(S1, TH1, 1000.0, np.arange(-45, 45.01, 2.5),
                                     np.arange(400, 1200, 25.0), F1))
            psi, d = b["psi_deg"], b["d"]
            lab = "推荐解 ψ=%.0f°, $d$=%.0f m" % (psi, d)
        P = (d * math.cos((TH1 + psi) * G.D2R), d * math.sin((TH1 + psi) * G.D2R))
        feas = F1.max_dist(P) <= 1000.0 + 1e-9
        if feas:
            w = SEL.worst_diameter(S1, TH1, P, F1)
            extra = "  ← 直径无界！" if w >= SEL.INF else "（最坏直径 %.0f m）" % w
        else:
            extra = "  ← 不可行（测不到信号）！"
        ax.plot(*P, mk, color=col, ms=13, zorder=7, label="%s%s" % (lab, extra))
        ax.annotate("", xy=P, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2, alpha=0.6))
    ax.set_aspect("equal")
    ax.set_xlim(-2200, 2200)
    ax.set_ylim(-1800, 1800)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("(b) 常见错误答案 vs 推荐解（$R_{lo}=1000$ m）")
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, "q2_pareto.png")

    rows = []
    for R_lo in (1000.0, 1250.0, 1500.0):
        b = SEL.best_by(SEL.scan(S1, TH1, R_lo, np.arange(-85, 85.1, 5.0),
                                 np.arange(200, 2400.1, 100.0), F1))
        d1, d2 = SEL.analytic_d_feasible_range(b["psi_deg"], R_lo)
        rows.append([R_lo, "%.2f" % SEL.analytic_psi_max(R_lo),
                     "%.2f" % SEL.psi_max_upper(R_lo),
                     "%.1f" % b["psi_deg"], "%.0f" % b["d"],
                     "%.1f" % b["worst_diam_m"], "%.1f" % d1, "%.1f" % d2,
                     "%.0f" % b["move_time_s"]])
    _csv("q2_recommend.csv",
         ["R_lo_m", "psi_max_exact_deg", "psi_max_closed_form_deg", "psi_star_deg",
          "d_star_m", "worst_diam_m", "d_min_m", "d_max_m", "move_time_s"], rows)
    for r in rows:
        print("    R_lo=%4.0f m  ψ_max=%.2f°(闭式 %.2f°)  ψ*=%.0f°  d*=%.0f m  "
              "最坏直径=%.1f m  d∈[%.0f, %.0f]  移动 %.0f s"
              % (r[0], float(r[1]), float(r[2]), float(r[3]), float(r[4]),
                 float(r[5]), float(r[6]), float(r[7]), float(r[8])))


if __name__ == "__main__":
    print("问题2 出图 ...")
    fig_F1_candidate()
    fig_worst_map()
    fig_pareto()
    print("完成")
