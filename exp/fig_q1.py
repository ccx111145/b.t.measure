# -*- coding: utf-8 -*-
"""
问题1 出图
  figs/q1_wedge_model.png       楔形半平面表示 + 两楔形交会定位区域
  figs/q1_region_vs_gamma.png   定位区域随交会角的变化（6 个子图）
  figs/q1_diameter_map.png      D(r,γ) 热力图与 D/r 等值线
  figs/q1_cover.png             覆盖判据：成功/反例 + R*/(D/2) 直方图 + 成功率曲线
并输出 tab/q1_*.csv
"""
import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.plot_style import C, TABDIR, draw_poly, fill_poly, save, setup  # noqa: E402
from robot import geometry as G                                          # noqa: E402
import matplotlib.pyplot as plt                                          # noqa: E402

setup()
RNG = np.random.default_rng(20260913)


def _save_csv(name, header, rows):
    p = os.path.join(TABDIR, name)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print("  表已保存:", p)


# ================================================================ 图1
def fig_wedge_model():
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0))
    S = np.array([0.0, 0.0])
    th = 55.0
    d = 1.0

    # ---- (a) 楔形与半平面 ----
    ax = axes[0]
    R = 1000.0
    for a, ls, lab in ((th - d, "-", "边界射线 $\\theta-\\delta$"),
                       (th + d, "-", "边界射线 $\\theta+\\delta$")):
        ax.plot([0, R * math.cos(a * G.D2R)], [0, R * math.sin(a * G.D2R)],
                color=C["s1"], lw=1.6, ls=ls, label=lab)
    span = np.linspace(th - d, th + d, 60)
    ax.fill([0] + [R * math.cos(a * G.D2R) for a in span] + [0],
            [0] + [R * math.sin(a * G.D2R) for a in span] + [0],
            color=C["s1"], alpha=0.22, label="楔形 $W(S,\\theta,1°)$")
    # 半平面法线示意
    for a in (th - d, th + d):
        n = np.array([math.sin(a * G.D2R), -math.cos(a * G.D2R)])
        ax.annotate("", xy=420 * n, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.0, ls=":"))
    ax.plot(*S, "o", color=C["s1"], ms=7, zorder=5)
    ax.annotate("$S$", S + np.array([18, -55]), color=C["s1"], fontsize=12)
    Gt = np.array([900 * math.cos((th + 0.4) * G.D2R), 900 * math.sin((th + 0.4) * G.D2R)])
    ax.plot(*Gt, "*", color=C["src"], ms=14, zorder=6, label="干扰源 $G$")
    ax.annotate("$G$", Gt + np.array([20, 20]), color=C["src"], fontsize=12)
    ax.annotate("", xy=520 * np.array([math.cos(th * G.D2R), math.sin(th * G.D2R)]),
                xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=C["s1"], lw=1.2, alpha=0.6))
    ax.text(300 * math.cos(th * G.D2R) + 40, 300 * math.sin(th * G.D2R) - 40,
            r"$\theta$", color=C["s1"], fontsize=13)
    ax.set_title("(a) 示向度楔形的凸锥/半平面表示\n"
                 r"$W=\{X: f(X;\theta-\delta)\leq0\}\cap\{X: f(X;\theta+\delta)\geq0\}$")
    ax.set_xlim(-200, 1250)
    ax.set_ylim(-250, 1250)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, 正东)")
    ax.set_ylabel("y (m, 正北)")
    ax.legend(loc="lower right", fontsize=8)

    # ---- (b) 两楔形交会 ----
    ax = axes[1]
    S1 = np.array([-900.0, -500.0])
    S2 = np.array([900.0, -400.0])
    Gt = np.array([120.0, 900.0])
    m = [(S1[0], S1[1], G.bearing(S1, Gt)), (S2[0], S2[1], G.bearing(S2, Gt))]
    for S_, col, lab in ((S1, C["s1"], "$S_1$"), (S2, C["s2"], "$S_2$")):
        th_ = G.bearing(S_, Gt)
        span = np.linspace(th_ - 1, th_ + 1, 40)
        L = 3200.0
        ax.fill([S_[0]] + [S_[0] + L * math.cos(a * G.D2R) for a in span] + [S_[0]],
                [S_[1]] + [S_[1] + L * math.sin(a * G.D2R) for a in span] + [S_[1]],
                color=col, alpha=0.16)
        for a in (th_ - 1, th_ + 1):
            ax.plot([S_[0], S_[0] + L * math.cos(a * G.D2R)],
                    [S_[1], S_[1] + L * math.sin(a * G.D2R)],
                    color=col, lw=0.9, ls="--", alpha=0.8)
        ax.plot([S_[0], Gt[0]], [S_[1], Gt[1]], color=col, lw=1.4)
        ax.plot(*S_, "o", color=col, ms=7, zorder=5)
        ax.annotate(lab, S_ + np.array([-10, -70]), color=col, fontsize=12, ha="center")
    poly = G.locate_region(m, arena=True)
    fill_poly(ax, poly, color=C["region"], alpha=0.65, zorder=4,
              label="定位区域 $\\mathcal{L}$ (|V|=%d)" % len(poly))
    draw_poly(ax, poly, color="k", lw=1.2, zorder=5)
    D, pair = G.polygon_diameter_pair(poly)
    ax.plot([pair[0][0], pair[1][0]], [pair[0][1], pair[1][1]], color="k", lw=1.4,
            ls="-.", zorder=6, label="直径 $D$=%.1f m" % D)
    ax.plot(*Gt, "*", color=C["src"], ms=15, zorder=7, label="真值 $G$")
    ax.set_title("(b) 交会定位区域（两楔形之交）与直径")
    ax.set_xlim(-1900, 1900)
    ax.set_ylim(-1600, 2100)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, 正东)")
    ax.set_ylabel("y (m, 正北)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    save(fig, "q1_wedge_model.png")


# ================================================================ 图2
def fig_region_vs_gamma():
    gammas = [15, 30, 45, 60, 75, 90]
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 7.6))
    r = 1200.0
    Gt = np.array([r, 0.0])
    S1 = np.array([0.0, 0.0])
    rows = []
    for ax, gam in zip(axes.ravel(), gammas):
        # 让 S2 与 G 的连线相对 GS1 方向夹角为 gam
        S2 = Gt + r * np.array([math.cos((180 - gam) * G.D2R), math.sin((180 - gam) * G.D2R)])
        m = [(S1[0], S1[1], G.bearing(S1, Gt)), (S2[0], S2[1], G.bearing(S2, Gt))]
        poly = G.locate_region(m, arena=False, box_half=20000.0)
        D, pair = G.polygon_diameter_pair(poly)
        A = G.polygon_area(poly)
        rows.append({"gamma_deg": gam, "diam_m": D, "area_m2": A,
                     "diam_over_r": D / r,
                     "diam_analytic_m": SEL_wa(r, r, gam),
                     "r_max_for_40m": 20.0 * math.sin(gam * G.D2R) / math.tan(G.SV_DELTA * G.D2R)})
        fill_poly(ax, poly, color=C["region"], alpha=0.75, zorder=3)
        draw_poly(ax, poly, color="k", lw=1.2, zorder=4)
        # 两条视线（自动被坐标轴裁剪）
        ax.plot([S1[0], Gt[0]], [S1[1], Gt[1]], color=C["s1"], lw=1.1, ls=":", zorder=2)
        ax.plot([S2[0], Gt[0]], [S2[1], Gt[1]], color=C["s2"], lw=1.1, ls=":", zorder=2)
        ax.plot(*Gt, "*", color=C["src"], ms=13, zorder=6)
        # 放大到定位区域附近
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * 1.9 + 4
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.annotate("指向 $S_1$", xy=(cx - half * 0.72, cy), color=C["s1"], fontsize=8.5,
                    ha="right", va="bottom")
        ax.annotate("指向 $S_2$", xy=(cx + half * 0.2, cy + half * 0.72), color=C["s2"],
                    fontsize=8.5, va="bottom")
        ax.set_title("交会角 $\\gamma=%d°$  $D$=%.1f m  ($D/r$=%.3f)\n"
                     "面积 %.0f m$^2$，直径≤40 m 需 $r\\leq$%.0f m"
                     % (gam, D, D / r, A, rows[-1]["r_max_for_40m"]))
        ax.set_aspect("equal")
    for ax in axes[1]:
        ax.set_xlabel("x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)")
    fig.suptitle("定位区域随交会角的变化（$r_1=r_2=1200$ m，各子图已放大到定位区域附近）",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "q1_region_vs_gamma.png")
    _save_csv("q1_region_vs_gamma.csv",
              ["gamma_deg", "diam_m", "area_m2", "diam_over_r", "diam_analytic_m",
               "r_max_for_40m"],
              [[x["gamma_deg"], "%.4f" % x["diam_m"], "%.2f" % x["area_m2"],
                "%.5f" % x["diam_over_r"], "%.4f" % x["diam_analytic_m"],
                "%.1f" % x["r_max_for_40m"]] for x in rows])


def SEL_wa(r1, r2, gam):
    from robot import selector as S
    return S.worst_diam_analytic(r1, r2, gam)


# ================================================================ 图3
def fig_diameter_map():
    rs = np.arange(200, 1501, 50.0)
    gs = np.arange(10, 90.1, 2.5)
    Dm = np.zeros((len(gs), len(rs)))
    for i, gam in enumerate(gs):
        for j, r in enumerate(rs):
            Gt = np.array([r, 0.0])
            S1 = np.array([0.0, 0.0])
            S2 = Gt + r * np.array([math.cos((180 - gam) * G.D2R),
                                    math.sin((180 - gam) * G.D2R)])
            poly = G.locate_region([(S1[0], S1[1], 0.0),
                                    (S2[0], S2[1], G.bearing(S2, Gt))],
                                   arena=False, box_half=20000.0)
            Dm[i, j] = G.polygon_diameter(poly) if len(poly) >= 3 else np.nan

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    ext = [rs[0], rs[-1], gs[0], gs[-1]]
    im = axes[0].imshow(Dm, origin="lower", aspect="auto", extent=ext, cmap="viridis")
    axes[0].set_xlabel("真值距离 $r$ (m)")
    axes[0].set_ylabel("交会角 $\\gamma$ (°)")
    axes[0].set_title("(a) 定位区域直径 $D$ (m)")
    plt.colorbar(im, ax=axes[0], fraction=0.046)

    im = axes[1].imshow(Dm / rs[None, :], origin="lower", aspect="auto", extent=ext,
                        cmap="magma")
    axes[1].set_xlabel("真值距离 $r$ (m)")
    axes[1].set_title("(b) 归一化 $D/r$")
    plt.colorbar(im, ax=axes[1], fraction=0.046)
    cs = axes[1].contour(rs, gs, Dm / rs[None, :], levels=np.arange(0.04, 1.2, 0.04),
                         colors="w", linewidths=0.5, alpha=0.6)
    try:
        axes[1].clabel(cs, fmt="%.2f", fontsize=6)
    except Exception as e:      # 渲染器内存不足时跳过标签，不影响主图
        print("   (跳过等值线标签:", e, ")")

    ax = axes[2]
    for gam in (30, 45, 60, 75, 90):
        i = int(np.argmin(np.abs(gs - gam)))
        ax.plot(rs, Dm[i], lw=1.8, label="$\\gamma=%d°$" % gam)
        rm = 20.0 * math.sin(gam * G.D2R) / math.tan(G.SV_DELTA * G.D2R)
        if rs[0] <= rm <= rs[-1]:
            ax.plot([rm], [40.0], "o", color="k", ms=5, zorder=6)
            ax.annotate("%.0f m" % rm, (rm, 40.0), textcoords="offset points",
                        xytext=(4, -12), fontsize=8)
    ax.axhline(40.0, color=C["bad"], ls="--", lw=1.2, label="40 m 精度阈值")
    ax.set_xlabel("真值距离 $r$ (m)")
    ax.set_ylabel("定位区域直径 $D$ (m)")
    ax.set_yscale("log")
    ax.set_title("(c) $D$ 随 $r$ 的变化与 40 m 阈值交点（黑点 = $r_{\\max}$）")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "q1_diameter_map.png")

    rows = []
    for i, gam in enumerate(gs):
        for j, r in enumerate(rs):
            rows.append([gam, r, "%.4f" % Dm[i, j], "%.5f" % (Dm[i, j] / r)])
    _save_csv("q1_diameter_map.csv", ["gamma_deg", "r_m", "diam_m", "diam_over_r"], rows)


# ================================================================ 图4
def fig_cover():
    rng = np.random.default_rng(7)
    stats = {}
    examples = {"ok": None, "bad": None}
    for n_st in (2, 3, 4):
        ratios, succ = [], 0
        tot = 0
        for _ in range(3000):
            S = [tuple(rng.uniform(-1400, 1400, 2)) for _ in range(n_st)]
            Gt = rng.uniform(-1500, 1500, 2)
            m = [(s[0], s[1], G.bearing(s, Gt)) for s in S]
            poly = G.locate_region(m, arena=False, box_half=30000.0)
            if len(poly) < 3:
                continue
            D = G.polygon_diameter(poly)
            if D <= 1e-9 or D > 12000:
                continue
            res = G.covers_with_diameter_circle(poly)
            tot += 1
            succ += 1 if res["covered"] else 0
            ratios.append(res["ratio"])
            if examples["bad"] is None and not res["covered"]:
                examples["bad"] = (poly, res)
            if examples["ok"] is None and res["covered"] and 60 < D < 160:
                examples["ok"] = (poly, res)
        stats[n_st] = {"n": tot, "success_rate": succ / max(1, tot),
                       "ratios": np.array(ratios)}

    fig = plt.figure(figsize=(13.2, 4.4))
    for k, key, title in ((1, "ok", "(a) 覆盖成功：直径圆覆盖定位区域"),
                          (2, "bad", "(b) 反例：直径圆**不能**覆盖")):
        ax = fig.add_subplot(1, 3, k)
        if examples[key] is None:
            ax.set_title(title + "（未找到样例）")
            continue
        poly, res = examples[key]
        fill_poly(ax, poly, color=C["region"], alpha=0.6, zorder=3)
        draw_poly(ax, poly, color="k", lw=1.2, zorder=4)
        A, B = np.array(res["A"]), np.array(res["B"])
        ctr = (A + B) / 2
        R = res["diameter"] / 2
        tt = np.linspace(0, 2 * math.pi, 200)
        ax.plot(ctr[0] + R * np.cos(tt), ctr[1] + R * np.sin(tt), color=C["hl"], lw=1.5,
                label="以 $D$ 为直径的圆")
        ax.plot([A[0], B[0]], [A[1], B[1]], color="k", lw=1.4, ls="-.", label="直径 $AB$")
        for (x, y, dot) in res["violators"]:
            ax.plot(x, y, "v", color=C["bad"], ms=9, zorder=6,
                    label="违反 Thales 判据的顶点")
            ax.annotate("$\\angle APB<90°$", (x, y), textcoords="offset points",
                        xytext=(10, 10), color=C["bad"], fontsize=8)
        ax.set_title(title + "\n$D$=%.1f m, $R^*/ (D/2)$=%.5f" % (res["diameter"], res["ratio"]))
        ax.set_aspect("equal")
        ax.legend(fontsize=7.5, loc="best")

    ax = fig.add_subplot(1, 3, 3)
    bins = np.linspace(0.9995, 1.004, 90)
    for n_st, col in ((2, C["s1"]), (3, C["s2"]), (4, C["region"])):
        rr = stats[n_st]["ratios"]
        ax.hist(rr, bins=bins, histtype="step", lw=1.6, color=col,
                label="$n=%d$，覆盖成功 %.2f%%（%d 组）"
                      % (n_st, 100 * stats[n_st]["success_rate"], stats[n_st]["n"]))
    ax.axvline(1.0, color="k", lw=1.0, ls="--", label="$R^*=D/2$（恰好覆盖）")
    ax.axvline(2 / math.sqrt(3), color=C["bad"], lw=1.0, ls=":",
               label="Jung 上界 $2/\\sqrt{3}$=%.4f" % (2 / math.sqrt(3)))
    ax.set_xlabel("$R^*/(D/2)$")
    ax.set_ylabel("频数")
    ax.set_title("(c) 最小包围圆半径 / ($D$/2) 分布")
    ax.set_xlim(0.9995, 1.006)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "q1_cover.png")

    _save_csv("q1_cover_stats.csv",
              ["n_stations", "n_samples", "success_rate", "ratio_mean", "ratio_max"],
              [[n, stats[n]["n"], "%.4f" % stats[n]["success_rate"],
                "%.6f" % stats[n]["ratios"].mean(),
                "%.6f" % stats[n]["ratios"].max()] for n in (2, 3, 4)])
    for n in (2, 3, 4):
        print("     n=%d 覆盖率=%.2f%%  R*/(D/2) 均值=%.6f 最大=%.6f"
              % (n, 100 * stats[n]["success_rate"], stats[n]["ratios"].mean(),
                 stats[n]["ratios"].max()))


if __name__ == "__main__":
    print("问题1 出图 ...")
    fig_wedge_model()
    fig_region_vs_gamma()
    fig_diameter_map()
    fig_cover()
    print("完成")
