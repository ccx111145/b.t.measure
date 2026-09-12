# -*- coding: utf-8 -*-
"""
1a 真实天线方向图（替代理想半平面）

理想半平面 = 前向增益为 1、后向严格为 0 的极端模型。真实定向天线
（八木、贴片阵、喇叭）有：有限的主瓣波束宽度、副瓣、后瓣电平与零陷。

模型
----
用归一化电压方向图 g(psi)（psi 为偏离主轴的角度）描述，
接收功率与 g^2 成正比。设参考距离 R_ref 处的轴向功率恰好等于接收灵敏度，
则**从偏离主轴 psi 的方向、距离 r 处能收到信号**当且仅当

        r <= R_ref * g(psi)                                  (1)

于是给定信标朝向 d、平台位置 P：

        可检出  <=>  |P - G| <= R_ref * g( angle(u(d), P-G) )   (2)

论文采用的理想半平面即 g(psi)=1 (psi<=90°)、0 (psi>90°)，且 R_ref = R_max。

**推广后的检测完备性判据**
对固定朝向 d，可检区域 A(d) = { P : (2) 成立 }。
网络 {P_i} 能保证"任意朝向都能检出 G" 当且仅当

        for all d :  {P_i} ∩ A(d) != empty                   (3)

理想半平面下 (3) 退化为 G ∈ conv(S_G)（论文定理 3）✓
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

R_MIN, R_MAX = 1000.0, 1500.0


# ------------------------------------------------------------------ 方向图
def pattern_ideal_halfplane() -> Callable[[np.ndarray], np.ndarray]:
    """理想半平面：前向 1、后向 0（论文基线模型）"""
    def g(psi):
        p = np.abs(np.asarray(psi, dtype=float))
        return np.where(p <= math.pi / 2 + 1e-12, 1.0, 0.0)
    return g


def pattern_cardioid(back_db: float = -20.0) -> Callable[[np.ndarray], np.ndarray]:
    """心形近似：无零陷，后瓣有限"""
    b = 10 ** (back_db / 20.0)

    def g(psi):
        c = np.cos(np.asarray(psi, dtype=float))
        return np.maximum((1 - b) / 2 * (1 + c) + b, 0.0)
    return g


def pattern_yagi(front_back_db: float = 16.0, hpbw_deg: float = 55.0,
                 sidelobe_db: float = -13.0) -> Callable[[np.ndarray], np.ndarray]:
    """八木天线近似：主瓣 cos^n + 副瓣 + 有限后瓣"""
    # 由 3 dB 波束宽度定 n：cos^n(hpbw/2) = 1/sqrt(2)
    n = math.log(1 / math.sqrt(2)) / math.log(math.cos(math.radians(hpbw_deg / 2)))
    fb = 10 ** (-front_back_db / 20.0)
    sl = 10 ** (sidelobe_db / 20.0)

    def g(psi):
        p = np.abs(np.asarray(psi, dtype=float))
        p = np.minimum(p, math.pi)
        main = np.maximum(np.cos(p), 0.0) ** n
        back = fb * np.maximum(np.cos(p), 0.0) ** 4 * (p > math.pi / 2)
        # 一个可解析的副瓣模型：在 ±(hpbw*1.6) 附近抬起
        sl_term = sl * np.exp(-((p - math.radians(hpbw_deg) * 1.6) ** 2)
                              / (2 * math.radians(25.0) ** 2))
        return np.clip(main + back + sl_term, 0.0, 1.0)
    return g


def pattern_sector(hpbw_deg: float = 65.0, front_back_db: float = 25.0) -> Callable:
    """3GPP 型扇区方向图：主瓣高斯 + 前后比"""
    fb = 10 ** (-front_back_db / 20.0)

    def g(psi):
        p = np.abs(np.asarray(psi, dtype=float))
        p = np.minimum(p, math.pi)
        main = 10 ** (-3.0 * (p / math.radians(hpbw_deg)) ** 2 / 2.0)
        fwd = p <= math.pi / 2
        return np.where(fwd, main, fb)
    return g


PATTERNS = {
    "Ideal half-plane (paper baseline)": pattern_ideal_halfplane(),
    "Cardioid, back lobe -20 dB": pattern_cardioid(-20.0),
    "Patch array, HPBW 65 deg, F/B 25 dB": pattern_sector(65.0, 25.0),
    "Yagi, HPBW 55 deg, F/B 16 dB, SL -13 dB": pattern_yagi(16.0, 55.0, -13.0),
    "Yagi, HPBW 40 deg, F/B 20 dB, SL -15 dB": pattern_yagi(20.0, 40.0, -15.0),
}


# ------------------------------------------------------------------ 判据
def detectable_for_all_orientations(P: np.ndarray, g, pattern: Callable, r_ref: float,
                                    n_orient: int = 180) -> bool:
    """(3) 式的数值实现：对每个朝向 d，检查是否有网络点落在 A(d) 内"""
    P = np.asarray(P, dtype=float)
    v = P - np.asarray(g, dtype=float)              # G→P
    r = np.hypot(v[:, 0], v[:, 1])
    if (r <= 1e-9).any():
        return True                                  # 与网络点重合
    phi = np.arctan2(v[:, 1], v[:, 0])               # G→P 的绝对方向
    # 对每个朝向 d：psi = 夹角(u(d), G→P) = |phi - d| (mod 2pi)
    d = np.arange(n_orient) * (2 * math.pi / n_orient)
    psi = np.abs((phi[None, :] - d[:, None] + math.pi) % (2 * math.pi) - math.pi)
    ok = r[None, :] <= r_ref * pattern(psi)
    return bool(ok.any(axis=1).all())


def max_guaranteed_radius(pattern: Callable, r_ref: float,
                          n: int = 720) -> float:
    """任意朝向下都能收到信号的最大距离（即方向图最小值处的等效半径）"""
    psi = np.arange(n) * (math.pi / n)
    return float(r_ref * np.min(pattern(psi)))


if __name__ == "__main__":
    print("Minimum normalised gain of each pattern (determines the worst-orientation detection radius):")
    for name, g in PATTERNS.items():
        psi = np.linspace(0, math.pi, 721)
        vals = np.asarray(g(psi))
        print("  %-42s min=%.4f  最坏可检半径=%.0f m  (R_ref=1500 m)"
              % (name, vals.min(), 1500 * vals.min()))
