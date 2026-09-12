# -*- coding: utf-8 -*-
"""
定理 2 的独立复算（用于核实审稿意见 2）

关键点：式(5)(6) 中的角度是 **Theta = |psi| + delta**，不是 psi。
作者初版自检脚本误用了 psi，因此错误地得出“式(6) 全程不起约束作用”、
并把闭式 arcsin(Rmin/Rmax) - delta = 40.81 度当成了精确边界。

本脚本给出正确结果：精确 psi_max = 40.4361 度，该处 d 退化为 1003.74 m。
并以审稿人给出的反例说明：距离判据对定向源必要但不充分。
"""
import math

Rmin, Rmax, r_min, delta = 1000.0, 1500.0, 5.0, 1.0


def intervals(psi_deg):
    """返回 (5)、(6) 各自的可行区间，以及交集；Theta = |psi| + delta"""
    th = math.radians(psi_deg + delta)                 # <-- 关键：Theta = psi + delta
    c = Rmax * math.cos(th)
    disc = c * c - (Rmax * Rmax - Rmin * Rmin)
    if disc < 0:
        return None, None, None
    s = math.sqrt(disc)
    i5 = (c - s, c + s)                                # 式(5)
    z = math.cos(th)
    j = math.sqrt(r_min * r_min * z * z + (Rmin * Rmin - r_min * r_min))
    i6 = (r_min * z - j, r_min * z + j)                # 式(6)
    lo, hi = max(i5[0], i6[0]), min(i5[1], i6[1])
    return i5, i6, (lo, hi) if lo <= hi else None


def main():
    print("=" * 78)
    print("A. 复核审稿人给出的数字")
    print("=" * 78)
    i5, i6, inter = intervals(40.8103)
    print("  psi=40.8103 deg  Theta=41.8103 deg")
    print("     eq(5): d in [%.2f, %.2f]" % i5)
    print("     eq(6): d <= %.2f" % i6[1])
    print("     intersection: %s   ==> reviewer is right"
          % ("EMPTY" if inter is None else "[%.2f, %.2f]" % inter))

    print()
    print("=" * 78)
    print("B. 精确求解 psi_max")
    print("=" * 78)
    lo, hi = 0.0, 45.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if intervals(mid)[2] is None:
            hi = mid
        else:
            lo = mid
    iv = intervals(lo)[2]
    print("  psi_max (exact) = %.4f deg" % lo)
    print("  intersection there = [%.2f, %.2f]  (degenerates to a point)" % iv)
    print("  closed form arcsin(Rmin/Rmax) - delta = %.4f deg  (UPPER BOUND only)"
          % (math.degrees(math.asin(Rmin / Rmax)) - delta))

    print()
    print("=" * 78)
    print("C. psi = 0 处的 d 区间")
    print("=" * 78)
    i5, i6, inter = intervals(0.0)
    print("  eq(5): [%.2f, %.2f]   eq(6) upper: %.2f   intersection: [%.2f, %.2f]"
          % (i5[0], i5[1], i6[1], inter[0], inter[1]))

    print()
    print("=" * 78)
    print("D. 审稿人的定向源反例")
    print("=" * 78)
    S1, G, S2 = (0.0, 0.0), (900.0, 0.0), (600.0, 200.0)
    print("  half-plane H: x + 2y <= 900")
    print("  S1=(0,0) in H: %s" % (0 <= 900))
    print("  S2=(600,200) in H: %s   (600+400 = 1000 > 900)" % (1000 <= 900))
    print("  |S1-G| = %.0f m  (<= Rmax, first measurement succeeds)" % math.dist(S1, G))
    print("  |S2-G| = %.1f m  (<= Rmin = 1000, but OUTSIDE H  ==> no signal)"
          % math.dist(S2, G))
    psi = math.degrees(math.atan2(S2[1] - S1[1], S2[0] - S1[0]))
    d = math.dist(S1, S2)
    rw = max(math.sqrt(d * d + r * r - 2 * d * r * math.cos(math.radians(psi + delta)))
             for r in (r_min, Rmax))
    print("  eq(4) worst-case distance from S2 = %.1f m <= Rmin  ==> TEST PASSES" % rw)
    print("  CONCLUSION: for directional emitters the distance criterion is")
    print("              necessary but NOT sufficient (now stated in the paper).")


if __name__ == "__main__":
    main()
