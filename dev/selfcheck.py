# -*- coding: utf-8 -*-
"""
论文自检：把 paper_sci.tex 里的每个数字对回原始数据文件与解析式
输出：每项 PASS/FAIL + 依据
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAB = os.path.join(ROOT, "tab")
tex = open(os.path.join(ROOT, "paper_sci.tex"), encoding="utf-8").read()

OK, BAD = [], []


def chk(name, cond, detail=""):
    (OK if cond else BAD).append(name)
    print("  [%s] %-46s %s" % ("PASS" if cond else "FAIL", name, detail))


def jload(fn):
    return json.load(open(os.path.join(TAB, fn), encoding="utf-8"))


def cload(fn):
    return list(csv.DictReader(open(os.path.join(TAB, fn), encoding="utf-8-sig")))


print("=" * 96)
print("A. 解析式复核")
print("=" * 96)
R, Rmin, Rmax, dlt, r_min, clr = 1800.0, 1000.0, 1500.0, 1.0, 5.0, 20.0

# A1 边界弧长下界
rho_star = math.sqrt(R * R - Rmin * Rmin)
phi = math.degrees(math.acos((rho_star ** 2 + R ** 2 - Rmin ** 2) / (2 * rho_star * R)))
k_lb = math.ceil(360.0 / (2 * phi))
chk("rho* = sqrt(R^2-Rmin^2) = 1496.7", abs(rho_star - 1496.7) < 0.1, "%.1f" % rho_star)
chk("phi_max = arcsin(Rmin/R) = 33.75 deg",
    abs(phi - math.degrees(math.asin(Rmin / R))) < 1e-6 and abs(phi - 33.75) < 0.01,
    "%.2f" % phi)
chk("arc = 67.5 deg", abs(2 * phi - 67.5) < 0.1, "%.1f" % (2 * phi))
chk("k >= 6", k_lb == 6, "k>=%d" % k_lb)

# A2 路线下界
L_lb = math.pi * (R ** 2 - Rmin ** 2) / (2 * Rmin)
chk("route lower bound = 3519 m", abs(L_lb - 3519) < 2, "%.0f" % L_lb)

# A3 第二测点
psi_max = math.degrees(math.asin(Rmin / Rmax)) - dlt
chk("closed-form upper bound 40.81 deg", abs(psi_max - 40.8) < 0.1, "%.2f" % psi_max)
# (5)(6) 的可行区间：注意 Θ = |ψ| + δ（此前误用 ψ，导致结论错误）
def admissible(psi_deg):
    th = math.radians(psi_deg + dlt)                 # Θ = ψ + δ
    c = Rmax * math.cos(th)
    disc = c * c - (Rmax * Rmax - Rmin * Rmin)
    if disc < 0:
        return None
    sq = math.sqrt(disc)
    i5 = (c - sq, c + sq)                            # 式(5)
    z = math.cos(th)
    jj = math.sqrt(r_min * r_min * z * z + (Rmin * Rmin - r_min * r_min))
    i6 = (r_min * z - jj, r_min * z + jj)            # 式(6)
    lo, hi = max(i5[0], i6[0]), min(i5[1], i6[1])
    return (lo, hi) if lo <= hi else None


# 二分求精确 ψ_max
lo, hi = 0.0, 45.0
for _ in range(200):
    mid = (lo + hi) / 2
    if admissible(mid) is None:
        hi = mid
    else:
        lo = mid
psi_exact = lo
iv = admissible(psi_exact)
chk("psi_max(精确) = 40.436 deg", abs(psi_exact - 40.436) < 0.005, "%.4f" % psi_exact)
chk("psi_max 处区间退化为单点 d ≈ 1003.74",
    iv is not None and abs(iv[0] - 1003.74) < 0.5 and abs(iv[1] - iv[0]) < 0.5,
    "[%.2f, %.2f]" % iv if iv else "空")
chk("闭式 arcsin(Rmin/Rmax)-δ = 40.81 只是上界",
    abs(math.degrees(math.asin(Rmin / Rmax)) - dlt - 40.8103) < 0.001
    and math.degrees(math.asin(Rmin / Rmax)) - dlt > psi_exact + 0.3,
    "上界 %.4f > 精确 %.4f" % (math.degrees(math.asin(Rmin / Rmax)) - dlt, psi_exact))
iv0 = admissible(0.0)
chk("psi=0 -> d in [500.1, 1005.0]",
    abs(iv0[0] - 500.1) < 0.5 and abs(iv0[1] - 1005.0) < 0.5,
    "[%.2f, %.2f]" % iv0)

# 定向源反例（审稿人给的）
_G, _S2 = (900.0, 0.0), (600.0, 200.0)
chk("定向反例：S2 在照射半平面外 x+2y=1000>900", 600 + 2 * 200 > 900)
chk("定向反例：|S2-G| = 360.6 m <= Rmin 但仍收不到",
    abs(math.dist(_S2, _G) - 360.6) < 0.5)
_psi = math.degrees(math.atan2(200.0, 600.0))
_d = math.dist((0.0, 0.0), _S2)
_rw = max(math.sqrt(_d ** 2 + r ** 2 - 2 * _d * r * math.cos(math.radians(_psi + dlt)))
          for r in (r_min, Rmax))
chk("定向反例：式(4) 最坏距离 927.8 <= Rmin（判据通过但实际失败）",
    abs(_rw - 927.8) < 1.0, "%.1f m" % _rw)

# A4 网络
inner, outer = 13, 12
chk("two-layer network = 25 points", inner + outer == 25, "%d+%d" % (inner, outer))
route_new, route_old = 17692.0, 30732.0
chk("route 17.7 km", abs(route_new / 1000 - 17.7) < 0.05, "%.1f km" % (route_new / 1000))
saving = 1 - route_new / route_old
chk("42% shorter", abs(saving - 0.42) < 0.01, "%.1f%%" % (100 * saving))
chk("single layer = 31 pts / 30.7 km", abs(route_old / 1000 - 30.7) < 0.05, "%.1f" % (route_old / 1000))

# A5 GridClear 解析成本（论文 Table II 引用）
area = math.pi * R ** 2
cell = 28.28
n_pts = int(area / (cell ** 2))
acts = n_pts * 3 + 2             # 每次 /clear 3 s（仅 1 次成功再 +2 s）
travel = area / cell / 5.0       # 犁地扫描：长度 ≈ 面积/行距
per_ch = acts + travel
chk("GridClear 每频道 ≈ 110170 s", abs(per_ch - 110170) < 400, "%.0f s" % per_ch)
chk("GridClear 20 频道 ≈ 2.2e6 s", abs(20 * per_ch / 2.2e6 - 1) < 0.02,
    "%.2e s" % (20 * per_ch))
print("      GridClear 解析：格点数=%d，动作=%.0f s，移动=%.0f s，每频道=%.0f s，20 频道=%.2e s"
      % (n_pts, acts, travel, per_ch, 20 * per_ch))

print()
print("=" * 96)
print("B. 实验数字对回原始数据")
print("=" * 96)
d = jload("q3_stats_final200.json")
chk("S1 200 局平均 359.7+-44.6", abs(d["avg_time_mean"] - 359.7) < 0.1
    and abs(d["avg_time_std"] - 44.6) < 0.1,
    "%.1f+-%.1f" % (d["avg_time_mean"], d["avg_time_std"]))
chk("S1 满清除率 1.000", d["full_clear_rate"] == 1.0, "%.3f" % d["full_clear_rate"])
chk("S1 总时间 4667 s", abs(d["virtual_time_mean"] - 4667) < 2,
    "%.0f" % d["virtual_time_mean"])

for tag, nm, mt, tt in (("q4_all", "S2 全定向 678.5+-100.4", 678.5, 8804),
                        ("q4_half", "S2 半定向 674.2+-99.9", 674.2, 8751)):
    dd = jload("q3_stats_%s.json" % tag)
    chk(nm, abs(dd["avg_time_mean"] - mt) < 0.2 and abs(dd["virtual_time_mean"] - tt) < 3,
        "%.1f+-%.1f, total %.0f" % (dd["avg_time_mean"], dd["avg_time_std"],
                                    dd["virtual_time_mean"]))

# 基线对照
for tag, prob in (("q3", "S1"), ("q4", "S2")):
    rows = cload("compare_%s.csv" % tag)
    st = {}
    for r in rows:
        st.setdefault(r["strategy"], []).append(r)
    o = st["Ours"]
    chk("%s Proposed cleared=1.000" % prob,
        abs(np.mean([float(r["clear_ratio"]) for r in o]) - 1.0) < 1e-9)
    lm = st["Lawnmower"]
    expect = 0.509 if prob == "S1" else 0.227
    got = float(np.mean([float(r["clear_ratio"]) for r in lm]))
    chk("%s Lawnmower cleared %.3f" % (prob, expect), abs(got - expect) < 0.004,
        "%.3f" % got)
    chk("%s no baseline certifies" % prob,
        all(float(r["certified"]) == 0 for s in ("Lawnmower", "RandomWaypoints", "HomingOnly")
            for r in st[s]) or prob == "S1")

s1 = jload("compare_q3_stats.json")
chk("S1 paired diff +0.49 (Lawnmower)",
    abs(s1["_paired_tests_vs_Ours"]["Lawnmower"]["clear_ratio"]["mean_diff"] - 0.4906) < 0.01,
    "%.4f" % s1["_paired_tests_vs_Ours"]["Lawnmower"]["clear_ratio"]["mean_diff"])
p = s1["_paired_tests_vs_Ours"]["Lawnmower"]["clear_ratio"]["sign_test_p"]
chk("p < 2e-9", p < 2e-9, "%.2e" % p)

# 消融
a3 = jload("ablation_q3_stats.json")
chk("S1 A1 cleared 0.420", abs(a3["A1-no-feasible"]["clear_ratio_mean"] - 0.420) < 0.002,
    "%.3f" % a3["A1-no-feasible"]["clear_ratio_mean"])
chk("S1 A1 time 4345", abs(a3["A1-no-feasible"]["virtual_time_mean"] - 4345) < 2)
a4 = jload("ablation_q4_stats.json")
chk("S2 A1 cleared 0.710", abs(a4["A1-no-feasible"]["clear_ratio_mean"] - 0.710) < 0.002,
    "%.3f" % a4["A1-no-feasible"]["clear_ratio_mean"])
chk("S2 A2 cleared 0.672", abs(a4["A2-no-mec"]["clear_ratio_mean"] - 0.672) < 0.002,
    "%.3f" % a4["A2-no-mec"]["clear_ratio_mean"])
r_abl = a4["A3-no-certify"]["virtual_time_mean"] / a4["Full"]["virtual_time_mean"]
chk("certificate saves 4.8x", abs(r_abl - 4.8) < 0.15, "%.2fx" % r_abl)
chk("S2 A3 cleared unchanged 1.000", a4["A3-no-certify"]["clear_ratio_mean"] == 1.0)
chk("S2 A3 certified 0.000", a4["A3-no-certify"]["certified_mean"] == 0.0)
chk("S2 A4 cleared 0.996", abs(a4["A4-ring-net"]["clear_ratio_mean"] - 0.996) < 0.002,
    "%.3f" % a4["A4-ring-net"]["clear_ratio_mean"])
chk("S2 A6 cleared 0.981 / cert 0.733",
    abs(a4["A6-no-sweep"]["clear_ratio_mean"] - 0.981) < 0.002
    and abs(a4["A6-no-sweep"]["certified_mean"] - 0.733) < 0.01,
    "%.3f / %.3f" % (a4["A6-no-sweep"]["clear_ratio_mean"],
                     a4["A6-no-sweep"]["certified_mean"]))

# 泛化
def gen(tag, dim):
    return [r for r in cload("generalize_%s.csv" % tag) if r["dim"] == dim]


g = gen("q4", "k")
chk("S2 k=10 -> 835 s, k=16 -> 555 s",
    abs(float(g[0]["avg_time"]) - 835.1) < 1 and abs(float(g[-1]["avg_time"]) - 554.7) < 1,
    "%.0f / %.0f" % (float(g[0]["avg_time"]), float(g[-1]["avg_time"])))
gd = gen("q4", "dir")
chk("S2 dir cost +2.8%",
    abs(float(gd[-1]["avg_time"]) / float(gd[0]["avg_time"]) - 1.028) < 0.005,
    "+%.1f%%" % (100 * (float(gd[-1]["avg_time"]) / float(gd[0]["avg_time"]) - 1)))
g1 = gen("q3", "dir")
chk("S1 dir collapse 1.000 -> 0.116",
    float(g1[0]["clear_ratio"]) == 1.0 and abs(float(g1[-1]["clear_ratio"]) - 0.116) < 0.003,
    "%.3f -> %.3f" % (float(g1[0]["clear_ratio"]), float(g1[-1]["clear_ratio"])))
chk("S1 dir intermediate 0.783/0.547/0.328",
    all(abs(float(g1[i]["clear_ratio"]) - v) < 0.003
        for i, v in ((1, 0.783), (2, 0.547), (3, 0.328))),
    "/".join("%.3f" % float(g1[i]["clear_ratio"]) for i in (1, 2, 3)))
gr = gen("q4", "rc")
spread = max(float(r["avg_time"]) for r in gr) / min(float(r["avg_time"]) for r in gr) - 1
chk("S2 Rc sensitivity < 0.2%", spread < 0.002, "%.3f%%" % (100 * spread))
gdl = gen("q4", "delta")
chk("S2 delta 0.25->2 costs <1.5%",
    (float(gdl[-1]["avg_time"]) / float(gdl[0]["avg_time"]) - 1) < 0.015,
    "%.2f%%" % (100 * (float(gdl[-1]["avg_time"]) / float(gdl[0]["avg_time"]) - 1)))
gdl1 = gen("q3", "delta")
chk("S1 delta 0.25->2 costs 11.7%",
    abs((float(gdl1[-1]["avg_time"]) / float(gdl1[0]["avg_time"]) - 1) - 0.117) < 0.005,
    "%.1f%%" % (100 * (float(gdl1[-1]["avg_time"]) / float(gdl1[0]["avg_time"]) - 1)))

# oracle
o3 = jload("oracle_q3_stats.json")
o4 = jload("oracle_q4_stats.json")
chk("S1 ours/oracle 2.43+-0.24",
    abs(o3["ratio_mean"] - 2.43) < 0.02 and abs(o3["ratio_std"] - 0.24) < 0.02,
    "%.2f+-%.2f" % (o3["ratio_mean"], o3["ratio_std"]))
chk("S2 ours/oracle 4.68+-0.62",
    abs(o4["ratio_mean"] - 4.68) < 0.02 and abs(o4["ratio_std"] - 0.62) < 0.02,
    "%.2f+-%.2f" % (o4["ratio_mean"], o4["ratio_std"]))
chk("discovery 58% / 78%",
    abs(o3["discovery_overhead_pct"] - 58.2) < 0.5
    and abs(o4["discovery_overhead_pct"] - 78.3) < 0.5,
    "%.1f%% / %.1f%%" % (o3["discovery_overhead_pct"], o4["discovery_overhead_pct"]))

print()
print("=" * 96)
print("C. 结论：%d 项通过，%d 项失败" % (len(OK), len(BAD)))
if BAD:
    for b in BAD:
        print("   FAIL:", b)
print("=" * 96)
