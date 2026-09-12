# -*- coding: utf-8 -*-
"""
问题1 第二问：以定位区域直径为直径的圆，能否覆盖定位区域？
方法：随机生成交会构型 -> 求凸多边形定位区域 -> 求直径端点(A,B)
      判据 (Thales)：点 P 在以 AB 为直径的闭圆内  <=>  (P-A)·(P-B) <= 0
      同时求最小包围圆(MEC)半径 R*，与 D/2 比较（Jung 定理给出 R* <= D/sqrt(3)）
"""
import math
import numpy as np

D2R = math.pi / 180.0
DELTA = 1.0


def wedge_halfplanes(S, theta_deg, delta=DELTA):
    Sx, Sy = S
    hs = []
    for a, keep_le in ((theta_deg - delta, True), (theta_deg + delta, False)):
        ar = a * D2R
        s = +1.0 if keep_le else -1.0
        A = np.array([-s * math.sin(ar), s * math.cos(ar)])
        b = float(A[0] * Sx + A[1] * Sy)
        hs.append((A, b))
    return hs


def region_poly(measures, box=60000.0):
    """半平面交（不含圆域），用大框保证有界"""
    poly = [(-box, -box), (box, -box), (box, box), (-box, box)]
    for (Sx, Sy, th) in measures:
        for A, b in wedge_halfplanes((Sx, Sy), th):
            out = []
            n = len(poly)
            for i in range(n):
                P = np.array(poly[i], float)
                Q = np.array(poly[(i + 1) % n], float)
                fp = float(A @ P - b)
                fq = float(A @ Q - b)
                if fp >= -1e-9:
                    out.append((P[0], P[1]))
                if (fp > 1e-9 and fq < -1e-9) or (fp < -1e-9 and fq > 1e-9):
                    t = fp / (fp - fq)
                    R = P + t * (Q - P)
                    out.append((R[0], R[1]))
            poly = out
            if len(poly) < 3:
                return []
    return poly


def diam_pair(poly):
    best, A, B = -1.0, None, None
    n = len(poly)
    for i in range(n):
        for j in range(i + 1, n):
            d = math.dist(poly[i], poly[j])
            if d > best:
                best, A, B = d, np.array(poly[i]), np.array(poly[j])
    return best, A, B


def mec(poly):
    """最小包围圆（n<=8，暴力枚举 1/2/3 点确定的圆）"""
    P = [np.array(p, float) for p in poly]
    n = len(P)
    best = (1e18, None, None)
    # 1 点
    for i in range(n):
        r = max(np.linalg.norm(P[i] - Q) for Q in P)
        if r < best[0]:
            best = (r, P[i], 0.0)
    # 2 点
    for i in range(n):
        for j in range(i + 1, n):
            c = (P[i] + P[j]) / 2
            r = max(np.linalg.norm(c - Q) for Q in P)
            if r < best[0]:
                best = (r, c, 0.0)
    # 3 点外接圆
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = P[i]
                bx, by = P[j]
                cx, cy = P[k]
                d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
                if abs(d) < 1e-12:
                    continue
                ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) +
                      (cx ** 2 + cy ** 2) * (ay - by)) / d
                uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) +
                      (cx ** 2 + cy ** 2) * (bx - ax)) / d
                c = np.array([ux, uy])
                r = max(np.linalg.norm(c - Q) for Q in P)
                if r < best[0]:
                    best = (r, c, 0.0)
    return best[0], best[1]


rng = np.random.default_rng(20260913)
print("=" * 84)
print("随机交会构型（不含圆域约束）下, '以直径为直径的圆' 的覆盖情况")
print("=" * 84)
print("%6s %10s %10s %8s %10s %10s %8s" %
      ("n_st", "D(m)", "2*(D/2)", "覆盖?", "R*/D", "kink", "顶点数"))
tot = {2: 0, 3: 0, 4: 0}
bad = {2: 0, 3: 0, 4: 0}
ratio = {2: [], 3: [], 4: []}
examples = []
for n_st in (2, 3, 4):
    for t in range(4000):
        # 检测点散布在半径 1500 的圆内
        S = [tuple(rng.uniform(-1500, 1500, 2)) for _ in range(n_st)]
        G = np.array(rng.uniform(-1600, 1600, 2))
        meas = []
        for (sx, sy) in S:
            th = math.degrees(math.atan2(G[1] - sy, G[0] - sx)) % 360
            meas.append((sx, sy, th))
        poly = region_poly(meas)
        if len(poly) < 3:
            continue
        D, A, B = diam_pair(poly)
        if D <= 1e-6 or D > 1e5:
            continue
        covered = all(float((np.array(P) - A) @ (np.array(P) - B)) <= 1e-6 for P in poly)
        Rstar, _ = mec(poly)
        tot[n_st] += 1
        ratio[n_st].append(Rstar / (D / 2))
        if not covered:
            bad[n_st] += 1
            if len(examples) < 6:
                examples.append((n_st, D, Rstar, poly))

for n_st in (2, 3, 4):
    if tot[n_st] == 0:
        continue
    rr = np.array(ratio[n_st])
    print("%6d %10s %10s %7.2f%% %10s %10s %8s" %
          (n_st, "-", "-", 100.0 * (tot[n_st] - bad[n_st]) / tot[n_st],
           "均值%.4f" % rr.mean(), "最大%.4f" % rr.max(), "-"))
    print("       覆盖失败率 = %d/%d = %.2f%%   R*/(D/2): 均值 %.4f  最大 %.4f  最小 %.4f"
          % (bad[n_st], tot[n_st], 100.0 * bad[n_st] / tot[n_st], rr.mean(), rr.max(), rr.min()))

print()
print("Jung 定理平面情形: R* <= D/sqrt(3) => R*/(D/2) <= 2/sqrt(3) = %.4f" % (2 / math.sqrt(3)))
print("R*/(D/2) == 1  <=>  以直径为直径的圆恰好覆盖")
print()
print("失败样例（可用作反例作图）：")
for (n_st, D, Rstar, poly) in examples:
    print("   n_st=%d  D=%.1f m  R*=%.1f m  R*/(D/2)=%.4f  顶点=%s"
          % (n_st, D, Rstar, Rstar / (D / 2),
             " ".join("(%.0f,%.0f)" % p for p in poly)))

print()
print("=" * 84)
print("平行四边形形状的解析结论：以对角线为直径的圆何时覆盖平行四边形")
print("=" * 84)
print("顶点 (0,0),(a,0),(a+b*cosT,b*sinT),(b*cosT,b*sinT)")
for (a, b) in [(100.0, 100.0), (100.0, 50.0), (200.0, 30.0)]:
    for T in [90, 60, 30, 10]:
        ct, st = math.cos(T * D2R), math.sin(T * D2R)
        V = [np.array(p) for p in [(0, 0), (a, 0), (a + b * ct, b * st), (b * ct, b * st)]]
        d1 = np.linalg.norm(V[0] - V[2])
        d2 = np.linalg.norm(V[1] - V[3])
        if d1 >= d2:
            A, B = V[0], V[2]
        else:
            A, B = V[1], V[3]
        ok = all(float((P - A) @ (P - B)) <= 1e-9 for P in V)
        print("   a=%5.0f b=%5.0f theta=%2d  长对角=%.1f  %s"
              % (a, b, T, max(d1, d2), "覆盖" if ok else "**不覆盖**"))
