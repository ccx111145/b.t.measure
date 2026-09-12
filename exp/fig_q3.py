# -*- coding: utf-8 -*-
"""
问题3 出图
  figs/q3_trajectory.png     单局轨迹（巡测路线 / 逼近腿 / 清除点 / 真值源）
  figs/q3_metrics.png        指标分布（平均定位清除时间、总时间、动作数、清除比例）
  figs/q3_sensitivity.png    参数灵敏度（由 tab/q3_sweep_*.csv 绘制）
  figs/q3_feasible_evolution.png  某频道可行域随证据的收缩过程
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.plot_style import C, TABDIR, draw_arena, fill_poly, save, setup   # noqa: E402
from robot import geometry as G                                           # noqa: E402
from robot.client import LocalRobotClient                                  # noqa: E402
from robot.policy import Q3Config, Q3Policy                               # noqa: E402
from sim.arena import generate_scenario                                   # noqa: E402
from sim.engine import ArenaEngine                                        # noqa: E402
import matplotlib.pyplot as plt                                           # noqa: E402

setup()


# ================================================================ 轨迹
def run_traced(seed: int, watch_ch: int = None, snap_max: int = 6):
    """跑一局并记录轨迹与（可选）某频道可行域快照"""
    sc = generate_scenario(seed=seed)
    eng = ArenaEngine(sc, "TEAM-TRACE")
    eng.arm()
    cl = LocalRobotClient(eng, "TEAM-TRACE")
    pol = Q3Policy(cl, Q3Config())

    snaps = []
    orig_measure = pol._measure

    def hooked(x, y, ch):
        r = orig_measure(x, y, ch)
        if watch_ch is not None and ch == watch_ch and r.accepted:
            snaps.append({"pos": (x, y), "result": r.measure_result,
                          "svd": r.svd_deg, "mask": pol.tr.mask(ch).copy(),
                          "spread": pol.tr.spread(ch),
                          "n_dir": pol.tr.st[ch].n_dir})
        return r

    pol._measure = hooked
    res = pol.run()
    return sc, eng, cl, pol, res, snaps


def fig_trajectory(seed: int = 3):
    sc, eng, cl, pol, res, _ = run_traced(seed)
    path = []
    for rec in cl.log:
        resp = rec.get("response") or {}
        if not resp.get("accepted"):
            continue
        pos = (rec.get("request") or {}).get("position")
        if pos is not None:
            path.append((float(pos["x"]), float(pos["y"]), rec["path"],
                         resp.get("clear_result")))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.6))

    ax = axes[0]
    draw_arena(ax)
    wp = [(0.0, 0.0)] + [(1200 * math.cos(2 * math.pi * k / 6),
                          1200 * math.sin(2 * math.pi * k / 6)) for k in range(6)]
    ax.plot([p[0] for p in wp], [p[1] for p in wp], "s--", color=C["hl"], ms=5,
            lw=1.0, alpha=0.7, label="巡测网（中心 + 6 环 ρ=1200）")
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    ax.plot(xs, ys, "-", color="#444", lw=0.8, alpha=0.55, zorder=2, label="机器狗轨迹")
    clears = [(p[0], p[1]) for p in path if p[3] == "success"]
    clear_fail = [(p[0], p[1]) for p in path if p[3] == "no_target_in_range"]
    ax.plot([c[0] for c in clears], [c[1] for c in clears], "P", color=C["ok"],
            ms=10, zorder=6, label="清除成功点 (%d)" % len(clears))
    if clear_fail:
        ax.plot([c[0] for c in clear_fail], [c[1] for c in clear_fail], "x",
                color=C["bad"], ms=8, zorder=6, label="清除未发现点 (%d)" % len(clear_fail))
    for s in sc.sources:
        ax.plot(s.x, s.y, "*", color=C["src"], ms=14, zorder=5,
                label="干扰源（真值）" if s is sc.sources[0] else None)
    ax.set_aspect("equal")
    ax.set_xlim(-2050, 2050)
    ax.set_ylim(-2050, 2050)
    ax.set_xlabel("x (m, 正东)")
    ax.set_ylabel("y (m, 正北)")
    ax.set_title("(a) 第 %d 局轨迹：清除 %d/%d，平均 %.0f s/源，动作 %d 次"
                 % (seed, res.n_cleared, res.n_sources, res.avg_locate_clear_time_s,
                    res.n_actions))
    ax.legend(fontsize=7, loc="lower left", ncol=2)

    # 虚拟时刻-距离累计
    ax = axes[1]
    vt = []
    for rec in cl.log:
        resp = rec.get("response") or {}
        if resp.get("accepted"):
            vt.append(resp.get("virtual_time_s", 0.0))
    cum = np.zeros(len(xs))
    for i in range(1, len(xs)):
        cum[i] = cum[i - 1] + math.dist(path[i - 1][:2], path[i][:2])
    ax.plot(np.arange(len(vt)), vt, "-", color=C["s1"], lw=1.6, label="虚拟时刻 $t_V$")
    ax.set_xlabel("指令序号")
    ax.set_ylabel("虚拟时刻 (s)")
    ax.set_title("(b) 虚拟时间推进（总 %.0f s）" % res.virtual_time_s)
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(cum)), cum, "--", color=C["region"], lw=1.4,
             label="累计行进距离")
    ax2.set_ylabel("累计行进距离 (m)", color=C["region"])
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.tight_layout()
    save(fig, "q3_trajectory.png")
    print("    轨迹：清除 %d/%d，虚拟总时间 %.0f s，行进 %.0f m，动作 %d"
          % (res.n_cleared, res.n_sources, res.virtual_time_s, cum[-1], res.n_actions))


# ================================================================ 指标分布
def fig_metrics(tag: str = "final200"):
    p = os.path.join(TABDIR, "q3_runs_%s.csv" % tag)
    rows = []
    with open(p, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    avg = np.array([float(r["avg_locate_clear_time_s"]) for r in rows
                    if r["avg_locate_clear_time_s"] not in ("", "nan")])
    vt = np.array([float(r["virtual_time_s"]) for r in rows])
    acts = np.array([int(r["n_actions"]) for r in rows])
    reqs = np.array([int(r["n_requests"]) for r in rows])
    ratio = np.array([float(r["clear_ratio"]) for r in rows])
    nsrc = np.array([int(r["n_sources"]) for r in rows])

    fig, axes = plt.subplots(1, 4, figsize=(15.0, 3.9))
    ax = axes[0]
    ax.hist(avg, bins=22, color=C["s1"], alpha=0.8, edgecolor="w")
    ax.axvline(avg.mean(), color=C["bad"], lw=1.6,
               label="均值 %.1f s" % avg.mean())
    ax.axvline(np.percentile(avg, 90), color=C["region"], lw=1.4, ls="--",
               label="P90 %.1f s" % np.percentile(avg, 90))
    ax.set_xlabel("平均定位清除时间 (s)")
    ax.set_ylabel("频数")
    ax.set_title("(a) 平均定位清除时间分布（%d 局）" % len(avg))
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(vt, bins=22, color=C["s2"], alpha=0.8, edgecolor="w")
    ax.axvline(vt.mean(), color=C["bad"], lw=1.6, label="均值 %.0f s" % vt.mean())
    ax.set_xlabel("定位清除总时间 (s)")
    ax.set_title("(b) 定位清除总时间分布")
    ax.legend(fontsize=8)

    ax = axes[2]
    ax.hist(acts, bins=22, color=C["region"], alpha=0.8, edgecolor="w",
            label="/measure + /clear 次数")
    ax.hist(reqs, bins=22, histtype="step", color=C["s1"], lw=1.5, label="总请求数")
    ax.set_xlabel("次数")
    ax.set_title("(c) 动作数与请求数（真实日志上限远大于此）")
    ax.legend(fontsize=8)

    ax = axes[3]
    cnt = {}
    for k in nsrc:
        cnt[k] = cnt.get(k, 0) + 1
    ks = sorted(cnt)
    ax.bar(ks, [cnt[k] for k in ks], color=C["hl"], alpha=0.85)
    ax.set_xlabel("干扰源个数 $k$")
    ax.set_ylabel("局数")
    ax.set_title("(d) 场景源数分布；满清除率 %.1f%%"
                 % (100 * float((ratio == 1.0).mean())))
    fig.tight_layout()
    save(fig, "q3_metrics.png")
    print("    指标：平均 %.1f±%.1f s，P90 %.1f，满清除率 %.3f"
          % (avg.mean(), avg.std(), np.percentile(avg, 90), (ratio == 1.0).mean()))


# ================================================================ 灵敏度
def fig_sensitivity():
    files = {"census_radius": ("环半径 ρ (m)", "巡测网环半径"),
             "census_ring_n": ("环上点数 k", "巡测环点数"),
             "grid_cell": ("栅格边长 (m)", "可行域栅格分辨率"),
             "homing_step_max": ("单步上限 (m)", "逼近单步上限"),
             "census_remeasure_spread": ("跳测阈值 (m)", "巡测跳测阈值"),
             "fuse_clear_radius": ("顺路清除门限 (m)", "顺路清除半径门限")}
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 7.4))
    for ax, (name, (xl, title)) in zip(axes.ravel(), files.items()):
        p = os.path.join(TABDIR, "q3_sweep_%s.csv" % name)
        if not os.path.exists(p):
            ax.set_title(title + "（无数据）")
            continue
        xs, ys, es, acts = [], [], [], []
        with open(p, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                xs.append(float(r[name]))
                ys.append(float(r["avg_time_mean"]))
                es.append(float(r["avg_time_std"]))
                acts.append(float(r["actions_mean"]))
        xs = np.array(xs)
        ys = np.array(ys)
        es = np.array(es)
        ax.errorbar(xs, ys, yerr=es, marker="o", lw=1.6, capsize=3, color=C["s1"],
                    label="平均定位清除时间")
        ax.set_xlabel(xl)
        ax.set_ylabel("平均定位清除时间 (s)", color=C["s1"])
        ax.set_title(title)
        ax2 = ax.twinx()
        ax2.plot(xs, acts, "s--", color=C["region"], ms=4, lw=1.2, label="动作数")
        ax2.set_ylabel("动作数", color=C["region"])
        ax2.grid(False)
        # 标注覆盖率
        for x, y in zip(xs, ys):
            pass
        if name == "census_radius":
            ax.axvspan(min(xs), 1150, color=C["bad"], alpha=0.10)
            ax.annotate("欠覆盖区\n（性能劣化 3 倍）", (1000, ys[0] * 0.55),
                        color=C["bad"], fontsize=8, ha="center")
    fig.suptitle("问题3 参数灵敏度（每点 30~40 局演练，误差棒为 1 倍标准差）", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "q3_sensitivity.png")


# ================================================================ 可行域演化
def fig_feasible_evolution(seed: int = 3):
    sc = generate_scenario(seed=seed)
    # 选一个离区域中心较远、证据较多的频道
    ch = max(sc.sources, key=lambda s: math.hypot(s.x, s.y)).channel
    _, _, cl, pol, res, snaps = run_traced(seed, watch_ch=ch)
    if not snaps:
        print("    该局没有频道 %d 的快照" % ch)
        return
    k = min(len(snaps), 6)
    idx = np.linspace(0, len(snaps) - 1, k).astype(int)
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 7.0))
    src = sc.by_channel(ch)
    for ax, i in zip(axes.ravel(), idx):
        s = snaps[i]
        m = s["mask"]
        x, y = pol.tr.r.x[m], pol.tr.r.y[m]
        if len(x):
            step = max(1, len(x) // 3000)           # 降采样，避免大图内存峰值
            ax.scatter(x[::step], y[::step], s=1.8, color=C["region"], alpha=0.5,
                       label="可行域单元")
        ax.plot(*s["pos"], "o", color=C["s1"], ms=6, label="检测点")
        if s["svd"] is not None:
            a = s["svd"] * G.D2R
            L = 1700
            for dd in (-1, 1):
                aa = a + dd * G.D2R
                ax.plot([s["pos"][0], s["pos"][0] + L * math.cos(aa)],
                        [s["pos"][1], s["pos"][1] + L * math.sin(aa)],
                        color=C["s1"], lw=0.8, ls=":")
        ax.plot(src.x, src.y, "*", color=C["src"], ms=14, label="真值")
        try:
            ok, c, md = pol.tr.clear_point(ch, 20.0)
            if ok and c is not None:
                ax.plot(*c, "+", color=C["ok"], ms=12, label="可清除点")
        except Exception:
            pass
        sp = s["spread"]
        bb = 900
        if len(x):
            cx, cy = float(x.mean()), float(y.mean())
            half = max(60.0, 1.4 * max(float(x.max() - x.min()), float(y.max() - y.min())))
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
        else:
            ax.set_xlim(-1850, 1850)
            ax.set_ylim(-1850, 1850)
        ax.set_aspect("equal")
        ax.set_title("第 %d 次证据后：%s\n可行域格数 %d，外接盒对角线 %.1f m"
                     % (i + 1, s["result"] or "-", int(m.sum()), sp), fontsize=9)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("x (m)")
    for ax in axes[:, 0]:
        ax.set_ylabel("y (m)")
    ok, c, md = pol.tr.clear_candidate(ch, 20.0)[:3]
    rep = pol.tr.near_point(ch)
    dev = math.dist(rep, (src.x, src.y)) if rep else float("nan")
    fig.suptitle("频道 %d 的可行域随证据收缩（本局最终：%s，可行域代表点偏离真值 %.1f m）"
                 % (ch, "已清除" if pol.tr.is_cleared(ch) else "未清除", dev),
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "q3_feasible_evolution.png")


if __name__ == "__main__":
    print("问题3 出图 ...")
    fig_trajectory(3)
    fig_metrics("final200")
    fig_sensitivity()
    fig_feasible_evolution(3)
    print("完成")
