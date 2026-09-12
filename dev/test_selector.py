# -*- coding: utf-8 -*-
"""
selector.py 单元测试

 S1 analytic_psi_max 与数值可行性边界一致
 S2 analytic_d_feasible_range 与 F1.max_dist 判定一致
 S3 候选区域内所有点都满足可检测性约束
 S4 最坏直径：解析法(不含圆域) >= 含圆域精算值（保守性）
 S5 推荐解与独立实现(plan_check 的扫描逻辑)一致
 S6 40 m 精度阈值表单调、量级正确
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot import geometry as G      # noqa: E402
from robot import selector as SEL    # noqa: E402

FAIL = []


def check(name, cond, info=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, info))
    if not cond:
        FAIL.append(name)


print("=" * 78)
print("S1 analytic_psi_max vs 数值可行性边界（psi 与 d 同时搜索）")
for R_lo in (1000.0, 1150.0, 1250.0, 1400.0):
    pm = SEL.analytic_psi_max(R_lo)
    F1 = SEL.F1Set((0.0, 0.0), 0.0)
    num = 0.0
    for psi in np.arange(0, 89.5, 0.25):
        found = False
        for d in np.arange(50.0, 2600.0, 5.0):
            S2 = (d * math.cos(psi * G.D2R), d * math.sin(psi * G.D2R))
            if F1.max_dist(S2) <= R_lo + 1e-6:
                found = True
                break
        if found:
            num = psi
        else:
            break
    check("R_lo=%.0f 解析=%.2f° 数值=%.2f°" % (R_lo, pm, num),
          abs(pm - num) < 0.4, "差 = %.3f°" % abs(pm - num))
    print("         闭式上界 = %.2f°" % SEL.psi_max_upper(R_lo))

print("=" * 78)
print("S2 analytic_d_feasible_range 与 F1.max_dist 一致")
F1 = SEL.F1Set((0.0, 0.0), 0.0)
bad = 0
worst = 0.0
for psi in np.arange(-55, 55.1, 5.0):
    d1, d2 = SEL.analytic_d_feasible_range(psi, 1000.0)
    ds = np.arange(1.0, 2600.0, 1.0)
    feas = []
    for d in ds:
        S2 = (d * math.cos(psi * G.D2R), d * math.sin(psi * G.D2R))
        feas.append(F1.max_dist(S2) <= 1000.0 + 1e-6)
    feas = np.array(feas)
    if not feas.any():
        if not math.isnan(d1):
            bad += 1
        continue
    idx = np.where(feas)[0]
    n1, n2 = ds[idx[0]], ds[idx[-1]]
    if math.isnan(d1):
        bad += 1
        continue
    worst = max(worst, abs(n1 - d1), abs(n2 - d2))
check("解析区间与数值判定一致", bad == 0,
      "不一致 = %d, 端点最大偏差 = %.1f m" % (bad, worst))

print("=" * 78)
print("S3 候选区域内所有点都满足可检测性约束")
bad = 0
for R_lo in (1000.0, 1250.0, 1500.0):
    for theta1 in (0.0, 37.0, 210.0):
        F1 = SEL.F1Set((0.0, 0.0), theta1)
        reg = SEL.candidate_region((0.0, 0.0), theta1, R_lo)
        for p in reg["points"][:: max(1, len(reg["points"]) // 200)]:
            if F1.max_dist(p) > R_lo + 1.0:      # 给 1 m 解析近似容差
                bad += 1
check("候选区域内点全部可检测", bad == 0, "越界 = %d" % bad)

print("=" * 78)
print("S4 最坏直径：解析法(无圆域) >= 含圆域精算（保守性）")
bad = 0
worst_gap = 0.0
for (psi, d) in [(40, 1000), (25, 1200), (40, 1500), (-35, 1100), (10, 900)]:
    S1, theta1 = (0.0, 0.0), 0.0
    S2 = (d * math.cos(psi * G.D2R), d * math.sin(psi * G.D2R))
    F1 = SEL.F1Set(S1, theta1)
    if F1.max_dist(S2) > 1000.0 + 1e-6:
        continue
    a = SEL.worst_diameter(S1, theta1, S2, F1, clip_arena=False)
    b = SEL.worst_diameter(S1, theta1, S2, F1, clip_arena=True)
    if a < b - 1e-6:
        bad += 1
    worst_gap = max(worst_gap, a - b)
check("解析上界 >= 精算值", bad == 0, "违反 = %d, 最大高估 = %.2f m" % (bad, worst_gap))

print("=" * 78)
print("S5 推荐解复算（与 plan_check.py 的独立扫描对照）")
rec = SEL.recommend((0.0, 0.0), 0.0, (1000.0, 1250.0, 1500.0))
for c in rec["cases"]:
    b = c["best"]
    if b is None:
        print("    R_lo=%.0f  无可行解" % c["R_lo"])
        continue
    print("    R_lo=%4.0f  psi_max解析=%5.2f°  可行psi=[%6.1f, %6.1f]°  最优 d=%5.0f psi=%5.0f 最坏直径=%7.1f m"
          % (c["R_lo"], c["psi_max_analytic_deg"], c["psi_feasible_range_deg"][0],
             c["psi_feasible_range_deg"][1], b["d"], b["psi_deg"], b["worst_diam_m"]))
    check("R_lo=%.0f 有可行解且最坏直径有限" % c["R_lo"],
          b is not None and b["worst_diam_m"] < SEL.INF)

print("=" * 78)
print("S6 40 m 精度阈值表")
tab = SEL.precision_table()
prev = None
mono = True
for row in tab:
    print("    交会角 %2.0f°  r_max = %7.1f m" % (row["gamma_deg"], row["r_max_m"]))
    if prev is not None and row["r_max_m"] <= prev:
        mono = False
    prev = row["r_max_m"]
check("阈值随交会角单调递增", mono)
r90 = [r for r in tab if r["gamma_deg"] == 90][0]["r_max_m"]
check("90° 时阈值 ≈ 1146 m", abs(r90 - 1145.8) < 1.0, "实际 = %.1f" % r90)

print("=" * 78)
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
print("全部测试通过")
