# -*- coding: utf-8 -*-
"""
用 Cramér-Rao 下界校验测向链

UCA（M 元、半径 R 波长、N 快拍、每阵元 SNR）单源测向的确定性 CRLB：

    J = N · SNR · M · (2πR)²
    σ_θ >= 1/sqrt(J)

（因为 Σ_m sin²(θ-φ_m) = M/2 且 Σ_m sin(θ-φ_m) = 0）

若实测 MUSIC 误差**低于**该下界，说明仿真或公式有误——必须先查清。
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.rf import ChannelConfig, DirectionFinder, UniformCircularArray   # noqa: E402


def crlb_deg(m, radius_wl, n_snap, snr_db, n_ang=20000):
    """数值计算 CRLB（不依赖闭式）"""
    snr = 10 ** (snr_db / 10.0)
    phi = 2 * math.pi * np.arange(m) / m
    # 对多个方位取平均（UCA 本应与方向无关）
    vals = []
    for th in np.linspace(0, 2 * math.pi, 24, endpoint=False):
        a = np.exp(1j * 2 * math.pi * radius_wl * np.cos(th - phi))
        d = (-1j * 2 * math.pi * radius_wl * np.sin(th - phi)) * a     # ∂a/∂θ
        P = np.eye(m) - np.outer(a, a.conj()) / (a.conj() @ a)
        # 确定性（Slepian-Bangs）CRLB：J = (2N/σ²)·Re{d^H Π⊥ d}，
        # 单源时 d^H Π⊥ d = (2πR)²·M/2，故 J = N·SNR·M·(2πR)²
        J = (2 * n_snap / (1.0 / snr)) * np.real(d.conj() @ P @ d)
        vals.append(1.0 / math.sqrt(J))
    return math.degrees(float(np.mean(vals)))


def measure(m, radius_wl, n_snap, snr_db, n_trial=4000, seed=5):
    rng = np.random.default_rng(seed)
    df = DirectionFinder(UniformCircularArray(m=m, radius_wl=radius_wl),
                         ChannelConfig(n_multipath=0, snr_db=snr_db,
                                       n_snapshots=n_snap))
    errs = []
    for _ in range(n_trial):
        th = rng.uniform(0, 360.0)
        est, _ = df.estimate(math.radians(th), rng)
        errs.append(abs((est - th + 180.0) % 360.0 - 180.0))
    e = np.array(errs)
    return float(np.sqrt((e ** 2).mean())), float(np.median(e)), float(np.percentile(e, 90))


if __name__ == "__main__":
    print("=" * 88)
    print("测向链 vs 解析 CRLB（LOS 单源）")
    print("=" * 88)
    print("%-34s %10s %10s %10s %10s" % ("配置", "CRLB(deg)", "实测RMS", "实测p50", "实测p90"))
    for (m, r, n, snr) in ((8, 0.35, 48, 12.0), (8, 0.35, 48, 20.0),
                           (8, 0.35, 256, 12.0), (16, 1.0, 48, 12.0),
                           (8, 1.0, 48, 12.0)):
        c = crlb_deg(m, r, n, snr)
        rms, p50, p90 = measure(m, r, n, snr)
        flag = ""
        if rms < c * 0.98:
            flag = "   <-- RMS 低于 CRLB，异常!"
        print("%-34s %10.4f %10.4f %10.4f %10.4f%s"
              % ("M=%d R=%.2fw N=%d SNR=%.0fdB" % (m, r, n, snr), c, rms, p50, p90, flag))
