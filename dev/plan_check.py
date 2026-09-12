# -*- coding: utf-8 -*-
"""
B题 建模方案 数值校验脚本
A. 楔形(±1度)半平面表示校验
B. 定位区域直径：顶点枚举 vs 区域内部网格采样
C. 问题2：第二检测点选择（含"第二点仍能测到信号"的硬可行性约束）
D. 覆盖几何：用半径R的检测圆盘覆盖半径1800圆域所需最少点数
E. 关键阈值
F. 定位区域直径与真值距离的经验关系
"""
import math
import numpy as np

D2R = math.pi / 180.0
R_ARENA = 1800.0
R_MIN, R_MAX = 1000.0, 1500.0
DELTA = 1.0


# ---------- 基础几何 ----------
def wedge_halfplanes(S, theta_deg, delta=DELTA):
    """{A·p >= b} 形式的两个半平面，其交 = 顶点S、中心方位theta、半张角delta 的闭凸锥

    f(P;a) = (P-S) x u(a) = (x-Sx)sin(a) - (y-Sy)cos(a)，对 P=S+t*u(beta) 有 f=t*sin(a-beta)。
    beta in [theta-d, theta+d]  <=>  f(P;theta-d) <= 0  且  f(P;theta+d) >= 0
    """
    Sx, Sy = S
    hs = []
    for a, keep_le in ((theta_deg - delta, True), (theta_deg + delta, False)):
        ar = a * D2R
        s = +1.0 if keep_le else -1.0
        A = np.array([-s * math.sin(ar), s * math.cos(ar)])
        b = float(A[0] * Sx + A[1] * Sy)
        hs.append((A, b))
    return hs


def clip_halfplane(poly, A, b, tol=1e-9):
    if len(poly) < 3:
        return []
    out = []
    n = len(poly)
    for i in range(n):
        P = np.array(poly[i], float)
        Q = np.array(poly[(i + 1) % n], float)
        fp = float(A @ P - b)
        fq = float(A @ Q - b)
        if fp >= -tol:
            out.append((P[0], P[1]))
        if (fp > tol and fq < -tol) or (fp < -tol and fq > tol):
            t = fp / (fp - fq)
            R = P + t * (Q - P)
            out.append((R[0], R[1]))
    return out


def circle_poly(R, n=90):
    return [(R * math.cos(2 * math.pi * k / n), R * math.sin(2 * math.pi * k / n)) for k in range(n)]


def locate_region(measures, arena_circle=True, box=None):
    """凸多边形表示（半平面裁剪法）。arena_circle=True 时先用目标圆域作先验约束。"""
    if arena_circle:
        poly = circle_poly(R_ARENA)
    else:
        B = box or 20000.0
        poly = [(-B, -B), (B, -B), (B, B), (-B, B)]
    for (Sx, Sy, th) in measures:
        for A, b in wedge_halfplanes((Sx, Sy), th):
            poly = clip_halfplane(poly, A, b)
            if len(poly) < 3:
                return []
    return poly


def diam_poly(poly):
    n = len(poly)
    if n < 2:
        return 0.0
    best = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = math.dist(poly[i], poly[j])
            if d > best:
                best = d
    return best


def area_poly(poly):
    n = len(poly)
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def region_quad_fast(measures):
    """仅由半平面(不含圆域)定义的定位区域，解析求交：O(1)，用于大规模扫描。
    返回凸多边形顶点(可能0~4个)。"""
    hs = []
    for (Sx, Sy, th) in measures:
        hs += wedge_halfplanes((Sx, Sy), th)
    n = len(hs)
    pts = []
    for i in range(n):
        for j in range(i + 1, n):
            A1, b1 = hs[i]
            A2, b2 = hs[j]
            M = np.array([A1, A2])
            det = float(np.linalg.det(M))
            if abs(det) < 1e-12:
                continue
            p = np.linalg.solve(M, np.array([b1, b2]))
            if all(float(A @ p - b) >= -1e-7 for A, b in hs):
                pts.append((float(p[0]), float(p[1])))
    # 去重
    out = []
    for p in pts:
        if not any(math.dist(p, q) < 1e-6 for q in out):
            out.append(p)
    return out


def diam_fast(measures, arena=True):
    poly = region_quad_fast(measures)
    if len(poly) < 2:
        return float('inf') if len(poly) < 2 else 0.0
    if arena:
        # 与目标圆域求交（近似：仅对顶点做裁剪，用于筛选阶段已足够）
        poly2 = poly
        for A, b in clip_halfplanes_circle():
            pass
    return diam_poly(poly)


def clip_halfplanes_circle():
    return []


# ---------- A ----------
print("=" * 78)
print("A. 楔形半平面表示校验：P=S+t*u(beta) 是否落在 [theta-1, theta+1]")
S = (137.0, -256.0)
th = 123.0
hs = wedge_halfplanes(S, th)
ok = True
for beta in np.arange(0, 360, 0.25):
    P = np.array(S) + 800.0 * np.array([math.cos(beta * D2R), math.sin(beta * D2R)])
    inside = all(float(A @ P - b) >= -1e-9 for A, b in hs)
    truth = abs(((beta - th + 180) % 360) - 180) <= 1.0
    if inside != truth:
        ok = False
        print("   不一致 beta=%.2f inside=%s truth=%s" % (beta, inside, truth))
print("   全部 1440 个角度一致:", ok)

# ---------- B ----------
print()
print("=" * 78)
print("B. 定位区域直径：顶点枚举 vs 区域内部网格采样（验证'极值在顶点取得'）")
rng = np.random.default_rng(20260913)
for trial in range(4):
    S1 = tuple(rng.uniform(-900, 900, 2))
    S2 = tuple(rng.uniform(-900, 900, 2))
    G = tuple(rng.uniform(-1200, 1200, 2))
    th1 = math.degrees(math.atan2(G[1] - S1[1], G[0] - S1[0])) % 360
    th2 = math.degrees(math.atan2(G[1] - S2[1], G[0] - S2[0])) % 360
    poly = locate_region([(S1[0], S1[1], th1), (S2[0], S2[1], th2)], arena_circle=False, box=6000)
    if len(poly) < 3:
        print("   trial%d 交集为空/退化" % trial)
        continue
    D = diam_poly(poly)
    xs0 = [p[0] for p in poly]
    ys0 = [p[1] for p in poly]
    XX, YY = np.meshgrid(np.arange(min(xs0), max(xs0) + 1e-9, 1.0),
                         np.arange(min(ys0), max(ys0) + 1e-9, 1.0))
    mask = np.ones(XX.shape, bool)
    for (Sx, Sy, t) in [(S1[0], S1[1], th1), (S2[0], S2[1], th2)]:
        for A, b in wedge_halfplanes((Sx, Sy), t):
            mask &= (A[0] * XX + A[1] * YY) >= b - 1e-6
    P = np.stack([XX[mask], YY[mask]], 1)
    dmax = 0.0
    for i in range(len(P)):
        dmax = max(dmax, np.hypot(P[i, 0] - P[:, 0], P[i, 1] - P[:, 1]).max())
    print("   trial%d 顶点直径=%9.3f m  网格最大距离=%9.3f m (差=%+.3f)  顶点数=%d  面积=%9.1f m^2"
          % (trial, D, dmax, dmax - D, len(poly), area_poly(poly)))

# ---------- C ----------
print()
print("=" * 78)
print("C. 问题2：第二检测点选择（S1=(0,0)，测得示向度 theta1=0 度）")
print("   源可行域 F1 = 楔形(theta1±1deg) ∩ 目标圆(1800) ∩ 圆盘(S1,1500)")
print("   硬约束：第二点必须仍能测到信号 => max_{G in F1} |S2-G| <= R_lo")

TH1 = 0.0
err_grid = [-1.0, 0.0, 1.0]

# 精细 F1 采样（仅用于最远距离可行性判定，向量化）
_r = np.arange(10.0, R_MAX + 1e-9, 5.0)
_b = np.arange(-DELTA, DELTA + 1e-9, 0.25)
RR, BB = np.meshgrid(_r, _b, indexing='ij')
F1x = (RR * np.cos((TH1 + BB) * D2R)).ravel()
F1y = (RR * np.sin((TH1 + BB) * D2R)).ravel()
m = F1x ** 2 + F1y ** 2 <= R_ARENA ** 2
F1x, F1y = F1x[m], F1y[m]
print("   F1 精细采样点数 = %d，最远半径 = %.1f m" % (len(F1x), np.hypot(F1x, F1y).max()))

# 粗 F1 采样（用于最坏直径扫描）
_r2 = np.arange(100.0, R_MAX + 1e-9, 100.0)
_b2 = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
R2, B2 = np.meshgrid(_r2, _b2, indexing='ij')
F1cx = (R2 * np.cos((TH1 + B2) * D2R)).ravel()
F1cy = (R2 * np.sin((TH1 + B2) * D2R)).ravel()
m2 = F1cx ** 2 + F1cy ** 2 <= R_ARENA ** 2
F1c = np.stack([F1cx[m2], F1cy[m2]], 1)
print("   F1 粗采样点数 = %d" % len(F1c))


def worst_diam_fast(S2):
    """最坏(面向上限, 不含圆域约束)直径"""
    worst = 0.0
    for G in F1c:
        base2 = math.degrees(math.atan2(G[1] - S2[1], G[0] - S2[0])) % 360
        for d1 in err_grid:
            for d2 in err_grid:
                poly = region_quad_fast([(0.0, 0.0, TH1 + d1), (S2[0], S2[1], base2 + d2)])
                if len(poly) < 3:
                    return float('inf')
                worst = max(worst, diam_poly(poly))
    return worst


def max_dist(S2):
    return float(np.hypot(F1x - S2[0], F1y - S2[1]).max())


summary = {}
for R_lo in [1000.0, 1250.0, 1500.0]:
    print("   --- 保守可检测半径 R_lo = %.0f m ---" % R_lo)
    rows = []
    for psi_deg in np.arange(-85, 85.1, 5.0):
        for d in np.arange(200, 2600, 100.0):
            S2 = (d * math.cos(psi_deg * D2R), d * math.sin(psi_deg * D2R))
            if max_dist(S2) > R_lo:
                continue
            rows.append((worst_diam_fast(S2), d, psi_deg))
    rows.sort()
    if not rows:
        print("      无可行解")
        continue
    for val, d, psi_deg in rows[:6]:
        print("      最坏直径*=%8.1f m   d=%5.0f m   psi=%5.0f 度" % (val, d, psi_deg))
    feas_psi = sorted(set(p for _, _, p in rows))
    print("      可行 psi 范围: %.0f ~ %.0f 度（共 %d 个可行候选）"
          % (min(feas_psi), max(feas_psi), len(rows)))
    summary[R_lo] = rows[0]

print()
print("   解析结论：F1 最远点约 (%.0f, 0)，要求 |S2-G|<=R_lo 且覆盖整个 F1" % R_MAX)
print("   => 令 S2 = d*u(psi)，需 d^2 + 1500^2 - 3000 d cos(psi) <= R_lo^2")
for R_lo in [1000.0, 1250.0]:
    lim = math.degrees(math.acos(min(1.0, (1500.0 ** 2 + 1500.0 ** 2 - R_lo ** 2) / (2 * 1500.0 * 1500.0))))
    print("      R_lo=%.0f m 时 |psi| <= %.1f 度 (d=1500 时)" % (R_lo, lim))

print()
print("   纯几何（不含可行性约束）：真值 G=(1200,0)，第二点取 S1 正北 d 米")
for d in [400, 600, 800, 1000, 1200, 1500, 2000, 2500]:
    S2 = (0.0, float(d))
    G = np.array([1200.0, 0.0])
    base2 = math.degrees(math.atan2(G[1] - S2[1], G[0] - S2[0])) % 360
    poly = locate_region([(0.0, 0.0, 0.0), (S2[0], S2[1], base2)], arena_circle=True)
    print("      d=%5d  直径=%7.2f m  面积=%9.0f m^2" % (d, diam_poly(poly), area_poly(poly)))

# ---------- D ----------
print()
print("=" * 78)
print("D. 覆盖几何：半径 R 的检测圆盘覆盖半径 1800 圆域所需点数")
samples = []
for x in np.arange(-R_ARENA, R_ARENA + 1, 20.0):
    for y in np.arange(-R_ARENA, R_ARENA + 1, 20.0):
        if x * x + y * y <= R_ARENA * R_ARENA:
            samples.append((x, y))
samples = np.array(samples)
print("   采样点数(20m 网格) =", len(samples))


def cover_radius(centers):
    d = np.full(len(samples), 1e18)
    for c in centers:
        d = np.minimum(d, np.hypot(samples[:, 0] - c[0], samples[:, 1] - c[1]))
    return d.max()


for R in [1000.0, 1250.0, 1500.0]:
    print("   --- 检测半径 R = %.0f ---" % R)
    for n_ring in range(1, 9):
        for with_center in (True, False):
            best_rho, best_val = None, 1e18
            for rho in np.arange(25.0, 2600.0, 25.0):
                cs = ([(0.0, 0.0)] if with_center else []) + \
                     [(rho * math.cos(2 * math.pi * k / n_ring), rho * math.sin(2 * math.pi * k / n_ring))
                      for k in range(n_ring)]
                v = cover_radius(cs)
                if v < best_val:
                    best_val, best_rho = v, rho
            tag = ("1中心+" if with_center else "") + "%d环" % n_ring
            print("      %-10s rho*=%6.0f m  最大未覆盖距离=%7.1f m  %s"
                  % (tag, best_rho, best_val, "覆盖OK" if best_val <= R else ""))

# ---------- E ----------
print()
print("=" * 78)
print("E. 关键阈值")
for gdeg in [30, 45, 60, 90]:
    r_lim = 20.0 * math.sin(gdeg * D2R) / math.tan(DELTA * D2R)
    print("   交会角 %2d 度：两点交会区域直径<=40m 的最大作用距离 r <= %7.1f m" % (gdeg, r_lim))
print("   归航误差界：移动 L 米后横向偏差 <= L*tan(1deg) = %.5f*L；<=20m 需 L<=%.0f m"
      % (math.tan(DELTA * D2R), 20.0 / math.tan(DELTA * D2R)))
print("   20m 清除半径的方格扫描：格距 <= 2*20/sqrt(2) = %.2f m" % (40 / math.sqrt(2)))
A_arena = math.pi * R_ARENA ** 2
sp = 28.28
n_cells = A_arena / (sp * sp)
print("   全区域方格 /clear 扫描：%d 点，纯动作 %.0f s + 移动 %.0f s = %.0f s（单频道）"
      % (n_cells, n_cells * 3, A_arena / (2 * 20) / 5, n_cells * 3 + A_arena / (2 * 20) / 5))
print("   单点全频道扫描耗时 = 20*5 + 19*1 = %d s" % (20 * 5 + 19))
print("   目标区域面积 = %.4e m^2，直径 3600 m，横穿耗时 = %.0f s" % (A_arena, 3600 / 5))

# ---------- F ----------
print()
print("=" * 78)
print("F. 定位区域直径与真值距离的经验关系（S1 在原点，第二点取垂直方向，基线 = r）")
for r in [200, 400, 600, 800, 1000, 1200, 1500]:
    S2 = (0.0, float(r))
    G = np.array([float(r), 0.0])
    base2 = math.degrees(math.atan2(G[1] - S2[1], G[0] - S2[0])) % 360
    poly = locate_region([(0.0, 0.0, 0.0), (S2[0], S2[1], base2)], arena_circle=True)
    D = diam_poly(poly)
    print("   r=%5d m  直径=%7.2f m  = %.4f*r   面积=%9.0f m^2" % (r, D, D / r, area_poly(poly)))
