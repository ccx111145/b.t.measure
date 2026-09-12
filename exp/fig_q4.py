# -*- coding: utf-8 -*-
"""
问题4 出图
  figs/q4_detectability.png   定向源可检出性：环状网 vs 六角点阵；区域外圈的必要性
  figs/q4_metrics.png         问题4 指标分布（全定向 / 半定向 / 用环状网的失败对照）
  figs/q4_trajectory.png      单局轨迹（31 点巡测网 + 清除点）
  figs/q4_phases.png          阶段耗时构成（问题3 vs 问题4）
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp.plot_style import C, TABDIR, draw_arena, save, setup     # noqa: E402
from robot.policy import Q3Config, Q3Policy                       # noqa: E402
from robot.client import LocalRobotClient                         # noqa: E402
from sim.arena import generate_scenario                           # noqa: E402
from sim.engine import ArenaEngine                                # noqa: E402
import matplotlib.pyplot as plt                                   # noqa: E402

setup()
R_LO = 1000.0
ARENA_R = 1800.0


def _hex_pts(step, extent):
    pts = []
    dy = step * math.sqrt(3) / 2
    ny = int(2 * extent / dy) + 2
    nx = int(2 * extent / step) + 2
    for j in range(-ny, ny + 1):
        y = j * dy
        off = (step / 2) if (j % 2) else 0.0
        for i in range(-nx, nx + 1):
            x = i * step + off
            if math.hypot(x, y) <= extent + 1e-9:
                pts.append((x, y))
    return np.array(pts)


def _in_hull(p, pts, tol=1e-6):
    if len(pts) == 0:
        return False
    d = pts - p
    r = np.hypot(d[:, 0], d[:, 1])
    if (r <= tol).any():
        return True
    a = np.sort(np.arctan2(d[:, 1], d[:, 0]))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * math.pi]]))
    return bool(gaps.max() <= math.pi + 1e-9)


def detect_ok(P, G):
    d = np.hypot(P[:, 0] - G[0], P[:, 1] - G[1])
    return _in_hull(np.asarray(G, float), P[d <= R_LO + 1e-9])


# ================================================================ 图1
def fig_detectability():
    ring = np.array([[0.0, 0.0]] + [(1200 * math.cos(2 * math.pi * k / 6),
                                     1200 * math.sin(2 * math.pi * k / 6)) for k in range(6)])
    hex_out = _hex_pts(1000.0, 2800.0)
    hex_in = _hex_pts(1000.0, 1800.0)

    rng = np.random.default_rng(3)
    n = 2500
    r = 1800 * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * math.pi
    G = np.stack([r * np.cos(a), r * np.sin(a)], 1)

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.8))
    for ax, P, title in ((axes[0], ring, "(a) 中心+6环 ρ=1200（问题3 用）\n%d 点" % len(ring)),
                         (axes[1], hex_in, "(b) 六角点阵 step=1000，裁剪到区域内\n%d 点" % len(hex_in)),
                         (axes[2], hex_out, "(c) 六角点阵 step=1000，含区域外\n%d 点" % len(hex_out))):
        draw_arena(ax, radius=ARENA_R)
        bad, good = [], []
        for g in G:
            (good if detect_ok(P, g) else bad).append(g)
        if bad:
            bad = np.array(bad)
            ax.scatter(bad[:, 0], bad[:, 1], s=3.5, color=C["bad"], alpha=0.75,
                       label="存在不可检出朝向（%d/%d = %.1f%%）"
                             % (len(bad), n, 100 * len(bad) / n))
        if good:
            good = np.array(good)
            ax.scatter(good[:, 0], good[:, 1], s=2.0, color=C["ok"], alpha=0.35,
                       label="任意朝向均可检出（%.1f%%）" % (100 * len(good) / n))
        ax.plot(P[:, 0], P[:, 1], "s", color=C["hl"], ms=4.5, zorder=5,
                label="检测点")
        ax.set_aspect("equal")
        ax.set_xlim(-3000, 3000)
        ax.set_ylim(-3000, 3000)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
    fig.suptitle("定向源可检出性判据 $G\\in\\mathrm{conv}(S_G)$，"
                 "$S_G$=网中到 $G$ 距离 ≤1000 m 的点（红=存在至少一个朝向无法检出）",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "q4_detectability.png")


# ================================================================ 图2
def fig_metrics():
    tags = [("q4_all", "问题4 全定向（hex 网）", C["s1"]),
            ("q4_half", "问题4 半定向（hex 网）", C["s2"]),
            ("final200", "问题3 全向（环状网）", C["hl"]),
            ("q4_ring", "问题4 半定向（环状网）", C["bad"])]
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.2))

    ax = axes[0]
    for tag, lab, col in tags:
        p = os.path.join(TABDIR, "q3_runs_%s.csv" % tag)
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        ratio = np.array([float(x["clear_ratio"]) for x in rows])
        ax.bar(lab, 100 * float((ratio == 1.0).mean()), color=col, alpha=0.85)
        ax.annotate("%.0f%%" % (100 * float((ratio == 1.0).mean())),
                    (lab, 100 * float((ratio == 1.0).mean())), ha="center",
                    va="bottom", fontsize=9)
    ax.set_ylabel("满清除率 (%)")
    ax.set_ylim(0, 115)
    ax.set_title("(a) 满清除率对比")
    ax.tick_params(axis="x", labelrotation=12, labelsize=8)

    ax = axes[1]
    for tag, lab, col in tags:
        p = os.path.join(TABDIR, "q3_runs_%s.csv" % tag)
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        v = np.array([float(x["avg_locate_clear_time_s"]) for x in rows
                      if x["avg_locate_clear_time_s"] not in ("", "nan")])
        if len(v) == 0:
            continue
        ax.hist(v, bins=22, histtype="step", lw=1.6, color=col,
                label="%s（均值 %.0f s）" % (lab, v.mean()))
    ax.set_xlabel("平均定位清除时间 (s)")
    ax.set_ylabel("频数")
    ax.set_title("(b) 平均定位清除时间分布")
    ax.legend(fontsize=7.5)

    ax = axes[2]
    labs, acts = [], []
    for tag, lab, col in tags:
        p = os.path.join(TABDIR, "q3_runs_%s.csv" % tag)
        if not os.path.exists(p):
            continue
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
        labs.append(lab.split("（")[0] + "\n" + lab.split("（")[1].rstrip("）"))
        acts.append(np.mean([float(x["n_actions"]) for x in rows]))
    ax.bar(range(len(labs)), acts, color=[t[2] for t in tags if
                                          os.path.exists(os.path.join(TABDIR, "q3_runs_%s.csv" % t[0]))],
           alpha=0.85)
    ax.set_xticks(range(len(labs)))
    ax.set_xticklabels(labs, fontsize=7.5)
    ax.set_ylabel("平均动作数")
    ax.set_title("(c) 平均动作数（真实加密日志 2 MB 上限的代理指标）")
    fig.tight_layout()
    save(fig, "q4_metrics.png")


# ================================================================ 图3
def fig_trajectory(seed: int = 5):
    sc = generate_scenario(seed=seed, directional_ratio=1.0)
    eng = ArenaEngine(sc, "T")
    eng.arm()
    cl = LocalRobotClient(eng, "T")
    pol = Q3Policy(cl, Q3Config(census_layout="hex_ring", directional=True))
    res = pol.run()
    wp = pol.census_waypoints()

    path = []
    for rec in cl.log:
        resp = rec.get("response") or {}
        if not resp.get("accepted"):
            continue
        pos = (rec.get("request") or {}).get("position")
        if pos is not None:
            path.append((float(pos["x"]), float(pos["y"]),
                         resp.get("clear_result")))

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.0))
    ax = axes[0]
    draw_arena(ax)
    ax.plot([p[0] for p in wp], [p[1] for p in wp], "-", color=C["hl"], lw=0.8,
            alpha=0.6, label="巡测路线（%d 点，%.1f km）"
                             % (len(wp), sum(math.dist(wp[i - 1][:2], wp[i][:2])
                                             for i in range(1, len(wp))) / 1000))
    ax.plot([p[0] for p in path], [p[1] for p in path], "-", color="#555", lw=0.7,
            alpha=0.5, label="机器狗轨迹")
    clears = [(p[0], p[1]) for p in path if p[2] == "success"]
    ax.plot([c[0] for c in clears], [c[1] for c in clears], "P", color=C["ok"],
            ms=9, zorder=6, label="清除成功点 (%d)" % len(clears))
    for s in sc.sources:
        mk = "*" if s.kind == "omni" else "D"
        ax.plot(s.x, s.y, mk, color=C["src"], ms=13 if s.kind == "omni" else 8,
                zorder=5, label=("全向源" if s.kind == "omni" else "定向源")
                if (s is sc.sources[0] or
                    (s.kind == "omni" and not any(t.kind == "omni" for t in sc.sources[:sc.sources.index(s)]))
                    or (s.kind == "dir" and not any(t.kind == "dir" for t in sc.sources[:sc.sources.index(s)])))
                else None)
        if s.kind == "dir" and s.dir_deg is not None:
            a = s.dir_deg * math.pi / 180
            ax.annotate("", xy=(s.x + 120 * math.cos(a), s.y + 120 * math.sin(a)),
                        xytext=(s.x, s.y),
                        arrowprops=dict(arrowstyle="->", color=C["src"], lw=1.0, alpha=0.8))
    ax.set_aspect("equal")
    ax.set_xlim(-2950, 2950)
    ax.set_ylim(-2950, 2950)
    ax.legend(fontsize=7.5, loc="lower left", ncol=2)
    ax.set_title("(a) 第 %d 局（全定向）：清除 %d/%d，平均 %.0f s/源，动作 %d"
                 % (seed, res.n_cleared, res.n_sources,
                    res.avg_locate_clear_time_s, res.n_actions))

    ax = axes[1]
    vt = []
    for rec in cl.log:
        resp = rec.get("response") or {}
        if resp.get("accepted"):
            vt.append(resp.get("virtual_time_s", 0.0))
    ax.plot(vt, "-", color=C["s1"], lw=1.5)
    n_census = None
    ax.set_xlabel("指令序号")
    ax.set_ylabel("虚拟时刻 (s)")
    ax.set_title("(b) 虚拟时间推进（总 %.0f s）\n"
                 "巡测段近似线性 = 主要在走路；清除段为折线" % res.virtual_time_s)
    fig.tight_layout()
    save(fig, "q4_trajectory.png")
    print("    Q4 轨迹：清除 %d/%d，平均 %.0f s/源，动作 %d，虚拟总时间 %.0f s"
          % (res.n_cleared, res.n_sources, res.avg_locate_clear_time_s,
             res.n_actions, res.virtual_time_s))


# ================================================================ 图4
def fig_phases():
    """问题3 vs 问题4 的固定成本对比（解析估算 + 实测）"""

    def route_len(pts, start=(0.0, 0.0)):
        if not pts:
            return 0.0
        P = [np.asarray(p, float) for p in pts]
        s = np.asarray(start, float)
        k = int(np.argmin([np.linalg.norm(p - s) for p in P]))
        order = [k]
        left = set(range(len(P))) - {k}
        cur = P[k]
        while left:
            j = min(left, key=lambda i: float(np.linalg.norm(P[i] - cur)))
            order.append(j)
            left.discard(j)
            cur = P[j]
        return float(sum(np.linalg.norm(P[order[i + 1]] - P[order[i]])
                         for i in range(len(order) - 1))) + \
            float(np.linalg.norm(P[order[0]] - s))

    p3 = Q3Policy.__new__(Q3Policy)
    p3.cfg = Q3Config()
    w3 = p3.census_waypoints()
    p4 = Q3Policy.__new__(Q3Policy)
    p4.cfg = Q3Config(census_layout="hex_ring")
    w4 = p4.census_waypoints()
    L3, L4 = route_len(w3), route_len(w4)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    labels = ["巡测移动", "巡测检测", "清除移动", "清除动作"]
    q3 = [L3 / 5, 140 * 6, 9600 / 5, 14 * 5]
    q4 = [L4 / 5, 620 * 6, 3000 / 5, 14 * 5]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, q3, 0.4, color=C["s1"], alpha=0.85,
           label="问题3（环状网 7 点，实测均值 4586 s）")
    ax.bar(x + 0.2, q4, 0.4, color=C["region"], alpha=0.85,
           label="问题4（六角网 31 点，实测均值 ~11800 s）")
    for i, (a, b) in enumerate(zip(q3, q4)):
        ax.annotate("%.0f" % a, (i - 0.2, a), ha="center", va="bottom", fontsize=8)
        ax.annotate("%.0f" % b, (i + 0.2, b), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("虚拟时间 (s)")
    ax.set_title("固定成本构成对比：问题4 的代价主要来自"
                 "「定向源检测保证」要求的加密巡测网\n"
                 "巡测路线 %.1f km → %.1f km；巡测检测 %d → %d 次"
                 % (L3 / 1000, L4 / 1000, 140, 620))
    ax.legend(fontsize=8)
    fig.tight_layout()
    save(fig, "q4_phases.png")


if __name__ == "__main__":
    print("问题4 出图 ...")
    fig_detectability()
    fig_metrics()
    fig_trajectory(5)
    fig_phases()
    print("完成")
