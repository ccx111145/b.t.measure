# -*- coding: utf-8 -*-
"""
② 软件无线电测向链路（样本级仿真）

替换掉抽象化的"示向度 = 真值 + 有界误差"模型，改为一条**完整的阵列处理链路**：

    发射源（可能多径）→ 信道（LOS + K 条镜面反射）→ 均匀圆阵快照
        → 采样协方差估计 → MUSIC 谱 → 峰值搜索 → 示向度估计

这样做的意义
------------
1. 测向误差不再是人造的"有界确定性误差"，而是**由物理链路产生**的：
   低 SNR 下谱峰变宽、多径下会出现**虚假谱峰**（对应论文 §4.5 的伪装向）；
2. 可以量化真实的"离群率"，而不是假设一个 q；
3. 这等价于一台**样本级 SDR 台架**：若把 array/carrier/snr 换成实测参数，
   同一套代码即可处理真实采集数据。

模型约定
--------
* 窄带远场模型，载波波长 lambda，阵元为半径 R 的 M 元均匀圆阵（UCA）；
* 第 k 条路径的到达角 theta_k、复增益 alpha_k；LOS 增益为 1，多径增益由
  反射系数 rho 与随机相位构成（Rician 型）；
* 快拍 x(t) = sum_k alpha_k a(theta_k) s(t) + n(t)，s 为恒模导频；
* 接收端不知道路径数，用 MUSIC 在角度栅格上找**最强谱峰**作为测向输出
  —— 这正是多径导致假示向角的机制。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class UniformCircularArray:
    """M 元均匀圆阵（半径以波长为单位）"""
    m: int = 8
    radius_wl: float = 0.35          # ≈ 半波长间距：2πR/M = 0.5λ ⟹ R = 0.4λ

    def steering(self, theta: np.ndarray) -> np.ndarray:
        """返回形状 (len(theta), m) 的导向矢量（相位参考在圆心）"""
        th = np.atleast_1d(theta)
        phi = 2 * math.pi * np.arange(self.m) / self.m
        return np.exp(1j * 2 * math.pi * self.radius_wl
                      * np.cos(th[:, None] - phi[None, :]))


@dataclass
class ChannelConfig:
    n_multipath: int = 2             # 镜面反射条数（0 = 纯 LOS）
    rho: float = 0.6                 # 多径幅度相对 LOS
    angular_spread_deg: float = 40.0 # 多径来波相对真值的角度散布
    snr_db: float = 12.0
    n_snapshots: int = 48


class DirectionFinder:
    """完整测向链路：快拍 → 协方差 → MUSIC → 角度估计"""

    def __init__(self, array: Optional[UniformCircularArray] = None,
                 cfg: Optional[ChannelConfig] = None,
                 grid_deg: float = 0.1):
        self.array = array or UniformCircularArray()
        self.cfg = cfg or ChannelConfig()
        g = np.arange(0.0, 360.0, grid_deg)
        self.grid = np.radians(g)
        self.A = self.array.steering(self.grid)          # (G, m)

    # ---------------------------------------------------------------- 信道
    def draw_paths(self, true_theta: float, rng: np.random.Generator
                   ) -> Tuple[np.ndarray, np.ndarray]:
        """返回 (到达角数组, 复增益数组)，第一条为 LOS"""
        cfg = self.cfg
        n = int(cfg.n_multipath)
        ang = [true_theta]
        gain = [1.0 + 0j]
        for _ in range(n):
            d = math.radians(cfg.angular_spread_deg) * rng.uniform(-1.0, 1.0)
            ang.append(true_theta + d)
            amp = cfg.rho * rng.uniform(0.5, 1.2)
            gain.append(amp * np.exp(1j * rng.uniform(0, 2 * math.pi)))
        return np.array(ang), np.array(gain)

    # ---------------------------------------------------------------- 估计
    def estimate(self, true_theta: float, rng: np.random.Generator):
        """执行一次测向，返回 (估计角(度), 是否落在栅格外, 主峰功率占比)"""
        cfg = self.cfg
        ang, gain = self.draw_paths(true_theta, rng)
        A = self.array.steering(ang)                     # (K, m)
        m = self.array.m
        snr = 10 ** (cfg.snr_db / 10.0)
        sig_pow = 1.0
        noise_pow = sig_pow / snr
        # 快拍
        X = np.zeros((m, cfg.n_snapshots), dtype=complex)
        for t in range(cfg.n_snapshots):
            s = np.exp(1j * rng.uniform(0, 2 * math.pi))  # 恒模导频
            x = A.T @ (gain * s)
            n = (rng.normal(0, math.sqrt(noise_pow / 2), m)
                 + 1j * rng.normal(0, math.sqrt(noise_pow / 2), m))
            X[:, t] = x + n
        R = (X @ X.conj().T) / cfg.n_snapshots
        # MUSIC（假设 1 个源）
        w, V = np.linalg.eigh(R)
        En = V[:, :-1]                                   # 噪声子空间
        proj = En @ En.conj().T
        denom = np.einsum("gm,mn,gn->g", self.A.conj(), proj, self.A).real
        spec = 1.0 / np.maximum(denom, 1e-12)
        k = int(np.argmax(spec))
        est = float(np.degrees(self.grid[k]))
        # 主峰占总谱的比例（用于置信度）
        share = float(spec[k] / spec.sum())
        return est % 360.0, share


class RFRobotSensor:
    """把测向链路接到机器人上的适配器

    * 每个 (位置量化, 频道) 用一个**固定**的随机种子 —— 对应论文中
      "同一地点的误差是确定的、只有移动才获得独立信息" 这一物理事实；
    * 返回估计角与置信度，供策略使用。
    """

    def __init__(self, df: DirectionFinder, seed: int = 20260913,
                 quant: float = 1.0):
        self.df = df
        self.seed = int(seed)
        self.quant = float(quant)
        self.n_calls = 0
        self.last_share = 1.0

    def bearing(self, pos: Tuple[float, float], channel: int,
                true_bearing_deg: float) -> float:
        qx = int(round(pos[0] / self.quant))
        qy = int(round(pos[1] / self.quant))
        rng = np.random.default_rng(
            (hash((qx, qy, int(channel), self.seed)) & 0xFFFFFFFF))
        est, share = self.df.estimate(math.radians(true_bearing_deg), rng)
        self.n_calls += 1
        self.last_share = share
        return est
