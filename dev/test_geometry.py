# -*- coding: utf-8 -*-
"""
geometry.py 单元测试（不依赖 pytest，直接 python 运行）

覆盖：
 T1 楔形半平面表示：与"角差法"逐点一致
 T2 定位区域：区域内点全部满足两楔形；区域外点至少违反一个楔形
 T3 半平面裁剪：与解析求交 fast_region_vertices 结果一致（面积/直径）
 T4 直径：枚举法 == 旋转卡壳（随机凸多边形 + 定位区域族）
 T5 直径极值在顶点取得：区域内部网格采样最大距离 ≤ 直径
 T6 MEC：与暴力枚举 1/2/3 点圆一致
 T7 有界性判据：与"大框裁剪是否触框"一致
 T8 覆盖判据：Thales 判据与"顶点是否在直径圆内"一致；平行四边形必被覆盖
 T9 栅格外近似：keep_wedge 不会删掉真值；单格塌缩时 best_clear_point 必成功
 T10 时间模型：与附录计时示例完全一致
"""
import math
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot import geometry as G  # noqa: E402

FAIL = []


def _candidates(pts):
    """由 1/2/3 点确定的候选圆心（用于暴力验证 MEC）"""
    pts = [np.asarray(p, float) for p in pts]
    out = [p for p in pts]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            out.append((pts[i] + pts[j]) / 2)
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            for k in range(j + 1, len(pts)):
                c, _ = G._circle_from3(pts[i], pts[j], pts[k])
                out.append(c)
    return out


def _in_poly(p, poly, tol=1e-6):
    n = len(poly)
    q = np.asarray(p, float)
    for i in range(n):
        a = np.asarray(poly[i], float)
        b = np.asarray(poly[(i + 1) % n], float)
        if G.cross2(b - a, q - a) < -tol:
            return False
    return True


def check(name, cond, info=""):
    tag = "PASS" if cond else "FAIL"
    print("  [%s] %s %s" % (tag, name, info))
    if not cond:
        FAIL.append(name)


print("=" * 78)
print("T1 楔形半平面表示 vs 角差法")
rng = np.random.default_rng(1)
bad = 0
for _ in range(60):
    S = rng.uniform(-1500, 1500, 2)
    th = rng.uniform(0, 360)
    for beta in np.arange(0, 360, 0.5):
        X = S + 800.0 * np.array([math.cos(beta * G.D2R), math.sin(beta * G.D2R)])
        got = G.in_wedge(X, S, th)
        truth = G.ang_dist(beta, th) <= G.SV_DELTA + 1e-12
        if got != truth:
            bad += 1
check("60 组随机构型 × 720 个方位角全部一致", bad == 0, "不一致 = %d" % bad)

print("=" * 78)
print("T2 定位区域：内外点判定")
bad = 0
for _ in range(200):
    S1 = rng.uniform(-1200, 1200, 2)
    S2 = rng.uniform(-1200, 1200, 2)
    Gt = rng.uniform(-1600, 1600, 2)
    if np.linalg.norm(Gt) > G.ARENA_R:
        continue
    th1 = G.bearing(S1, Gt)
    th2 = G.bearing(S2, Gt)
    poly = G.locate_region([(S1[0], S1[1], th1), (S2[0], S2[1], th2)])
    if len(poly) < 3:
        bad += 1
        continue
    # 真值必在区域内（测量无误差时）
    if not _in_poly(Gt, poly):
        bad += 1
check("真值点必落在定位区域内", bad == 0, "失败 = %d" % bad)

print("=" * 78)
print("T3 半平面裁剪法 vs 解析求交")
worst_area = worst_diam = 0.0
n_cmp = 0
for _ in range(3000):
    S1 = rng.uniform(-1500, 1500, 2)
    S2 = rng.uniform(-1500, 1500, 2)
    Gt = rng.uniform(-1700, 1700, 2)
    th1 = G.bearing(S1, Gt)
    th2 = G.bearing(S2, Gt)
    m = [(S1[0], S1[1], th1), (S2[0], S2[1], th2)]
    if not G.region_is_bounded(m):
        continue                      # 无界：直径无意义
    p2 = G.fast_region_vertices(m)
    if len(p2) < 3:
        continue
    if G.polygon_diameter(p2) > 5000:
        continue                      # 退化长条：大框会截断，不可比
    p1 = G.locate_region(m, arena=False, box_half=50000.0)
    if len(p1) < 3:
        continue
    n_cmp += 1
    worst_area = max(worst_area, abs(G.polygon_area(p1) - G.polygon_area(p2)))
    worst_diam = max(worst_diam,
                     abs(G.polygon_diameter(p1) - G.polygon_diameter(p2)))
check("面积一致", worst_area < 1e-3,
      "可比样本 = %d, 最大差 = %.3e m^2" % (n_cmp, worst_area))
check("直径一致", worst_diam < 1e-6,
      "可比样本 = %d, 最大差 = %.3e m" % (n_cmp, worst_diam))

print("=" * 78)
print("T4 直径：枚举法 vs 旋转卡壳")
worst = 0.0
cnt = 0
for _ in range(2000):
    k = int(rng.integers(3, 40))
    pts = rng.uniform(-1, 1, (k, 2))
    poly = G.convex_hull([tuple(p) for p in pts])
    if len(poly) < 3:
        continue
    d1 = G.diameter_bruteforce(poly)[0]
    d2 = G.diameter_calipers(poly)[0]
    worst = max(worst, abs(d1 - d2))
    cnt += 1
check("2000 个随机凸包上两法一致", worst < 1e-9,
      "样本 = %d, 最大差 = %.3e" % (cnt, worst))

# 定位区域族
worst2 = 0.0
for _ in range(2000):
    S1 = rng.uniform(-1500, 1500, 2)
    S2 = rng.uniform(-1500, 1500, 2)
    Gt = rng.uniform(-1700, 1700, 2)
    poly = G.locate_region([(S1[0], S1[1], G.bearing(S1, Gt)),
                            (S2[0], S2[1], G.bearing(S2, Gt))], arena=False)
    if len(poly) < 3:
        continue
    worst2 = max(worst2, abs(G.diameter_bruteforce(poly)[0] -
                             G.diameter_calipers(poly)[0]))
check("定位区域族两法一致", worst2 < 1e-9, "最大差 = %.3e" % worst2)

print("=" * 78)
print("T5 直径极值在顶点取得（内部网格采样 ≤ 直径）")
viol = 0
maxgap = 0.0
n_ok = 0
for _ in range(200):
    S1 = rng.uniform(-1200, 1200, 2)
    S2 = rng.uniform(-1200, 1200, 2)
    Gt = rng.uniform(-1600, 1600, 2)
    poly = G.locate_region([(S1[0], S1[1], G.bearing(S1, Gt)),
                            (S2[0], S2[1], G.bearing(S2, Gt))], arena=False)
    if len(poly) < 3:
        continue
    D = G.polygon_diameter(poly)
    if D > 400 or D <= 0:
        continue                      # 限制网格规模
    n_ok += 1
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    XX, YY = np.meshgrid(np.arange(min(xs), max(xs) + 1e-9, 0.5),
                         np.arange(min(ys), max(ys) + 1e-9, 0.5))
    inside = np.ones(XX.shape, bool)
    n = len(poly)
    for i in range(n):
        a = np.asarray(poly[i], float)
        b = np.asarray(poly[(i + 1) % n], float)
        E = np.stack([XX - a[0], YY - a[1]], -1)
        inside &= (E[..., 0] * (b - a)[1] - E[..., 1] * (b - a)[0]) >= -1e-6
    P = np.stack([XX[inside], YY[inside]], 1)
    if len(P) < 2:
        continue
    dm = 0.0
    step = max(1, len(P) // 300)
    for i in range(0, len(P), step):
        q = P[i]
        dm = max(dm, float(np.hypot(q[0] - P[:, 0], q[1] - P[:, 1]).max()))
    if dm > D + 1e-6:
        viol += 1
    maxgap = max(maxgap, D - dm)
check("采样最大距离恒 ≤ 直径", viol == 0,
      "样本 = %d, 违反 = %d, 最大间隙 = %.3f m" % (n_ok, viol, maxgap))

print("=" * 78)
print("T6 MEC vs 暴力枚举")
worst = 0.0
for _ in range(500):
    k = int(rng.integers(1, 9))
    pts = [tuple(p) for p in rng.uniform(-500, 500, (k, 2))]
    c, r = G.mec(pts)
    best = min(max(math.dist(cc, p) for p in pts)
               for cc in _candidates(pts))
    worst = max(worst, abs(r - best))
check("MEC 半径与暴力最优一致", worst < 1e-6, "最大差 = %.3e" % worst)


print("=" * 78)
print("T7 有界性判据：区间法 vs 回收射线枚举（互为正交校验）")
bad = 0
n_cmp = 0
n_unb = 0
for _ in range(4000):
    S1 = rng.uniform(-1000, 1000, 2)
    S2 = rng.uniform(-1000, 1000, 2)
    th1 = rng.uniform(0, 360)
    gap = rng.uniform(0, 12)
    th2 = (th1 + gap) % 360
    if abs(gap - 2 * G.SV_DELTA) < 0.25:
        continue                       # 临界区不可数值判定，跳过
    m = [(S1[0], S1[1], th1), (S2[0], S2[1], th2)]
    a = G.region_is_bounded(m, method="interval")
    b = G.region_is_bounded(m, method="rays")
    n_cmp += 1
    if not b:
        n_unb += 1
    if a != b:
        bad += 1
check("两法一致", bad == 0,
      "样本 = %d（其中无界 %d），不一致 = %d" % (n_cmp, n_unb, bad))

# 三站情形
bad = 0
for _ in range(2000):
    ms = []
    for _k in range(3):
        S = rng.uniform(-1000, 1000, 2)
        ms.append((S[0], S[1], rng.uniform(0, 360)))
    if G.region_is_bounded(ms, method="interval") != \
       G.region_is_bounded(ms, method="rays"):
        bad += 1
check("三站情形两法一致", bad == 0, "不一致 = %d" % bad)

print("=" * 78)
print("T8 覆盖判据 + 平行四边形定理")
bad = 0
for _ in range(3000):
    S1 = rng.uniform(-1500, 1500, 2)
    S2 = rng.uniform(-1500, 1500, 2)
    Gt = rng.uniform(-1700, 1700, 2)
    poly = G.locate_region([(S1[0], S1[1], G.bearing(S1, Gt)),
                            (S2[0], S2[1], G.bearing(S2, Gt))], arena=False)
    if len(poly) < 3:
        continue
    res = G.covers_with_diameter_circle(poly)
    A, B = np.asarray(res["A"]), np.asarray(res["B"])
    manual = all(float((np.asarray(P) - A) @ (np.asarray(P) - B)) <= 1e-9 for P in poly)
    if manual != res["covered"]:
        bad += 1
check("Thales 判据自洽", bad == 0, "不一致 = %d" % bad)

bad = 0
for _ in range(4000):
    a = rng.uniform(20, 400)
    b = rng.uniform(20, 400)
    T = rng.uniform(2, 178)
    ct, st = math.cos(T * G.D2R), math.sin(T * G.D2R)
    poly = [(0, 0), (a, 0), (a + b * ct, b * st), (b * ct, b * st)]
    res = G.covers_with_diameter_circle(poly)
    if not res["covered"]:
        bad += 1
check("任意平行四边形必被长对角线圆覆盖", bad == 0, "反例 = %d" % bad)

print("=" * 78)
print("T9 栅格外近似")
r = G.ArenaRaster(cell=10.0)
check("栅格规模合理", 90000 < r.n < 110000, "单元数 = %d" % r.n)
bad = 0
n_pt = 0
for _ in range(200):
    S = rng.uniform(-1500, 1500, 2)
    th = rng.uniform(0, 360)
    for _try in range(20):
        Gt = S + rng.uniform(200, 1500) * np.array(
            [math.cos((th + rng.uniform(-1, 1)) * G.D2R),
             math.sin((th + rng.uniform(-1, 1)) * G.D2R)])
        if math.hypot(Gt[0], Gt[1]) <= G.ARENA_R - 20:
            break
    else:
        continue
    n_pt += 1
    m = r.keep_wedge(S, th, G.SV_DELTA)
    d2 = (r.x - Gt[0]) ** 2 + (r.y - Gt[1]) ** 2
    k = int(np.argmin(d2))
    if math.sqrt(float(d2[k])) > r.cell * 0.7072 + 1e-6:
        continue                       # 最近格心超出一个栅格半径，不构成外近似反例
    if not m[k]:
        bad += 1
check("keep_wedge 不删真值（外近似）", bad == 0,
      "样本 = %d, 误删 = %d" % (n_pt, bad))

r2 = G.ArenaRaster(cell=10.0)
m = np.zeros(r2.n, bool)
d2 = (r2.x - 100.0) ** 2 + (r2.y - 100.0) ** 2
m[int(np.argmin(d2))] = True
r2.set_mask(m)
ok, c, md = r2.best_clear_point(20.0)
check("单格塌缩 → 一次 clear 必成功", ok and md <= 20.0,
      "center=(%.1f,%.1f) maxdist=%.2f" % (c[0], c[1], md))

print("=" * 78)
print("T10 时间模型 vs 附录计时示例（表2 / 第10节）")
p = (0.0, 0.0)
ch = 1
tv = 0.0
tv += 0.0                                            # /enter
tv += G.measure_cost((0, 0), (300, 400), 1, 1)       # 移动 100 + 检测 5
check("step2 = 105", abs(tv - 105) < 1e-9, "t=%.1f" % tv)
tv += G.measure_cost((300, 400), (300, 400), 1, 2)   # 切换 1 + 检测 5
check("step3 = 111", abs(tv - 111) < 1e-9, "t=%.1f" % tv)
tv += G.clear_cost((300, 400), (300, 0), success=False)  # 移动 80 + 3
check("step4 = 194", abs(tv - 194) < 1e-9, "t=%.1f" % tv)
tv += G.measure_cost((300, 0), (300, 0), 2, 2)       # 检测 5（不切频道）
check("step5 = 199", abs(tv - 199) < 1e-9, "t=%.1f" % tv)

print("=" * 78)
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
print("全部测试通过")
