# -*- coding: utf-8 -*-
"""
计时与语义引擎（附件1 表1、附件2 第 4/6/7/8/9 节的忠实实现）

虚拟时间规则
------------
* ``/enter``、``/exit`` 不推进时钟；
* ``/measure`` 总耗时 = 移动耗时 + 切频道耗时(0/1) + 5；
* ``/clear``  总耗时 = 移动耗时 + (成功 5 / 未发现 3)，且**不切频道**；
* 移动耗时 = 直线距离 / 5 m/s；
* 只有 ``accepted=true`` 才推进时钟；``accepted=false`` 时响应中 ``virtual_time_s=0``。

示向度误差
----------
* 误差 ∈ [-1°, 1°]，**同一地点固定**：实现为 ``(位置量化到 1 m, 频道)`` 的确定性哈希，
  因此重复检测得到完全相同的读数，而"抖一下再平均"无法降噪（与题面一致）。
  量化尺度可用 ``svd_quant`` 配置。
"""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import protocol as P
from .arena import CLEAR_R, NEAR_R, Scenario, Source

DOG_SPEED = 5.0
T_MEASURE = 5.0
T_SWITCH = 1.0
T_CLEAR_OK = 5.0
T_CLEAR_FAIL = 3.0

MASK64 = (1 << 64) - 1


def _splitmix64(x: int) -> int:
    x = (x + 0x9E3779B97F4A7C15) & MASK64
    z = x
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


def _stable_hash(a: int, b: int, c: int, seed: int) -> int:
    h = _splitmix64(a & MASK64)
    h = _splitmix64(h ^ (b & MASK64))
    h = _splitmix64(h ^ (c & MASK64))
    h = _splitmix64(h ^ (seed & MASK64))
    return h


@dataclass
class IdemRecord:
    fingerprint: Tuple
    status: int
    response: dict


class ArenaEngine:
    """一局测试的模拟器内核（不涉及 HTTP）"""

    def __init__(self, scenario: Scenario, robot_id: str,
                 arena_id: str = P.ARENA_ID,
                 max_virtual: float = 360_000.0,
                 max_real: float = 1200.0,
                 window_s: float = 1500.0,
                 clock: Callable[[], float] = time.monotonic,
                 wall_ms: Callable[[], float] = lambda: time.time() * 1000.0,
                 svd_quant: float = 1.0,
                 svd_delta: float = 1.0,
                 p_outlier: float = 0.0,
                 p_miss: float = 0.0,
                 rf_chain=None,
                 ghost=None,
                 duty_cycle: float = 1.0,
                 duty_slot_s: float = 30.0,
                 svd_seed: int = 20260913,
                 idem_limit: int = 200_000):
        self.scenario = scenario
        self.robot_id = robot_id
        self.arena_id = arena_id
        self.max_virtual = float(max_virtual)
        self.max_real = float(max_real)
        self.window_s = float(window_s)
        self._clock = clock
        self._wall_ms = wall_ms
        self.svd_quant = float(svd_quant)
        self.svd_delta = float(svd_delta)
        self.p_outlier = float(p_outlier)
        self.p_miss = float(p_miss)
        # rf_chain: dict(n_multipath=, rho=, snr_db=, n_snapshots=, seed=)
        # 给定时，示向度由**完整阵列处理链路**产生（样本级 SDR 仿真），
        # 而不再使用加性有界误差模型。
        self.rf = None
        # ghost=(x, y, prob)：固定的镜面反射体造成的**相干鬼源**
        # （空间相关多径：平台移动时虚假方位连续变化且始终指向同一位置）
        self.ghost = tuple(ghost) if ghost else None
        # duty_cycle：信标占空比。1.0 = 常发；<1 表示间歇突发发射。
        self.duty_cycle = float(min(max(duty_cycle, 0.0), 1.0))
        self.duty_slot_s = float(duty_slot_s)
        self.n_duty_off = 0
        if rf_chain:
            from .rf import (ChannelConfig, DirectionFinder, RFRobotSensor,
                             UniformCircularArray)
            kw = dict(rf_chain)
            seed = int(kw.pop("seed", 20260913))
            arr_m = int(kw.pop("array_m", 8))
            arr_r = float(kw.pop("array_radius_wl", 0.35))
            df = DirectionFinder(UniformCircularArray(m=arr_m, radius_wl=arr_r),
                                 cfg=ChannelConfig(**kw))
            self.rf = RFRobotSensor(df, seed=seed, quant=self.svd_quant)
        self.svd_seed = int(svd_seed)
        self.idem_limit = int(idem_limit)

        self.pos: Tuple[float, float] = (0.0, 0.0)
        self.channel: int = 1
        self.virtual_time: float = 0.0
        self.entered = False
        self.exited = False
        self.ended = False
        self.end_reason: Optional[str] = None
        self.window_opened_at: Optional[float] = None
        self.enter_at: Optional[float] = None
        self.records: Dict[str, IdemRecord] = {}

        self.n_measure = 0
        self.n_clear_ok = 0
        self.n_clear_fail = 0
        self.n_switch = 0
        self.n_outlier = 0
        self.n_miss = 0
        self.n_ghost = 0
        self.action_log: List[dict] = []
        self.truth_leak_guard = True     # 只为文档提醒：真值不通过任何接口返回

    # ============================================================ 生命周期
    def arm(self, at: Optional[float] = None):
        """打开 25 分钟测试窗口并开放机器狗接口"""
        self.window_opened_at = self._clock() if at is None else float(at)
        self.entered = self.exited = self.ended = False
        self.end_reason = None

    def _now(self) -> float:
        return self._clock()

    def deadline(self) -> Optional[float]:
        if self.enter_at is None or self.window_opened_at is None:
            return None
        return min(self.enter_at + self.max_real,
                   self.window_opened_at + self.window_s)

    def remaining_real(self) -> float:
        d = self.deadline()
        if d is None:
            return 0.0
        return max(0.0, d - self._now())

    def elapsed_real(self) -> float:
        return 0.0 if self.enter_at is None else max(0.0, self._now() - self.enter_at)

    def _check_timeouts(self):
        if self.ended:
            return
        if self.virtual_time >= self.max_virtual:
            self.ended, self.end_reason = True, "virtual_timeout"
            return
        if self.entered:
            d = self.deadline()
            if d is not None and self._now() > d:
                self.ended, self.end_reason = True, "real_timeout"

    # ============================================================ 响应构造
    def _env(self, accepted: bool, **extra) -> dict:
        r = {"accepted": bool(accepted),
             "real_timestamp_ms": int(self._wall_ms()),
             "virtual_time_s": (self.virtual_time if accepted else 0)}
        r.update(extra)
        return r

    @staticmethod
    def _norm360(a: float) -> float:
        return a % 360.0

    # ============================================================ 示向度误差
    def _chan_event(self, kind: str, channel: int) -> float:
        """非理想信道事件：由位置(量化)、频道与事件类型决定的确定性值 ∈ [0,1)"""
        qx = int(round(self.pos[0] / self.svd_quant))
        qy = int(round(self.pos[1] / self.svd_quant))
        h = hashlib.sha256(("%s|%d|%d|%d|%d"
                            % (kind, self.svd_seed, qx, qy, int(channel))).encode())
        return int.from_bytes(h.digest()[:6], "big") / float(1 << 48)

    def svd_error(self, p: Tuple[float, float], channel: int) -> float:
        """位置(量化后)+频道 决定的固定误差 ∈ (-δ, δ)，δ 由 svd_delta 给定（默认 1°）"""
        qx = int(round(p[0] / self.svd_quant))
        qy = int(round(p[1] / self.svd_quant))
        h = _stable_hash(qx, qy, int(channel), self.svd_seed)
        u = (h & 0xFFFFFF) / float(0x1000000)          # [0,1)
        return (u * 2.0 - 1.0) * self.svd_delta

    # ============================================================ 业务动作
    def _move_time(self, new_pos: Tuple[float, float]) -> float:
        return math.dist(self.pos, new_pos) / DOG_SPEED

    def _do_enter(self, payload) -> dict:
        if self.entered:
            return self._env(False)
        self.entered = True
        self.enter_at = self._now()
        rem = int(min(self.max_real, max(0.0, self.deadline() - self._now())))
        # /enter 的 remaining 也受 1200 上限约束
        rem = int(max(0, min(int(self.max_real), rem)))
        return self._env(True,
                         max_virtual_duration_s=self.max_virtual,
                         max_real_duration_s=self.max_real,
                         remaining_real_duration_s=rem)

    def _do_exit(self) -> dict:
        if not self.entered or self.exited:
            return self._env(False)
        self.exited = True
        self.ended = True
        self.end_reason = "user_exit"
        return self._env(True, exit_reason="user_exit")

    def _do_measure(self, pos, channel: int) -> dict:
        if not self.entered or self.exited or self.ended:
            return self._env(False)
        target = (float(pos["x"]), float(pos["y"]))
        t = self._move_time(target) + T_MEASURE
        switched = (channel != self.channel)
        if switched:
            t += T_SWITCH
            self.n_switch += 1
        self.pos = target
        self.channel = int(channel)
        self.virtual_time += t
        self.n_measure += 1

        src = self.scenario.by_channel(int(channel))
        # 占空比：本时隙是否发射（由 时隙 + 位置量化 + 频道 决定，可复现）
        duty_on = True
        if src is not None and not src.cleared and self.duty_cycle < 1.0:
            slot = int(self.virtual_time // self.duty_slot_s)
            qx = int(round(self.pos[0] / self.svd_quant))
            qy = int(round(self.pos[1] / self.svd_quant))
            h = hashlib.sha256(("duty|%d|%d|%d|%d"
                                % (slot, qx, qy, int(channel))).encode())
            u = int.from_bytes(h.digest()[:6], "big") / float(1 << 48)
            duty_on = u < self.duty_cycle
            if not duty_on:
                self.n_duty_off += 1
        if src is None or src.cleared or not duty_on:
            out = self._env(True, measure_result="no_signal")
        else:
            d = src.dist_to(self.pos)
            if d <= NEAR_R + 1e-9 and src.in_sector(self.pos):
                out = self._env(True, measure_result="near")
            elif src.detectable(self.pos):
                true_b = self._norm360(
                    math.degrees(math.atan2(src.y - self.pos[1], src.x - self.pos[0])))
                if self.rf is not None:
                    svd = self._norm360(self.rf.bearing(self.pos, channel, true_b))
                else:
                    svd = self._norm360(true_b + self.svd_error(self.pos, channel))
                if self.ghost is not None:
                    gx, gy, gp = self.ghost
                    if self._chan_event("ghost", channel) < gp:
                        svd = self._norm360(math.degrees(
                            math.atan2(gy - self.pos[1], gx - self.pos[0])))
                        self.n_ghost += 1
                out = self._env(True, measure_result="direction",
                                svd_deg=round(svd, 2))
            else:
                out = self._env(True, measure_result="no_signal")

        # ---- 非理想信道：阴影/衰落导致的漏检 ----
        if (self.p_miss > 0.0
                and out.get("measure_result") in ("direction", "near")
                and self._chan_event("miss", channel) < self.p_miss):
            out = self._env(True, measure_result="no_signal")
            self.n_miss += 1

        # ---- 非理想信道：多径/旁瓣导致的伪装示向度（虚假测向角）----
        if (self.p_outlier > 0.0
                and out.get("measure_result") == "no_signal"
                and self._chan_event("outlier", channel) < self.p_outlier):
            fake = self._norm360(360.0 * self._chan_event("angle", channel))
            out = self._env(True, measure_result="direction", svd_deg=round(fake, 2))
            self.n_outlier += 1

        self._log("measure", self.pos, channel, out)
        return out

    def _do_clear(self, pos, channel: int) -> dict:
        if not self.entered or self.exited or self.ended:
            return self._env(False)
        target = (float(pos["x"]), float(pos["y"]))
        t = self._move_time(target)
        self.pos = target
        src = self.scenario.by_channel(int(channel))
        hit = (src is not None and not src.cleared
               and src.dist_to(self.pos) <= CLEAR_R + 1e-9)
        if hit:
            src.cleared = True
            t += T_CLEAR_OK
            self.n_clear_ok += 1
            result = "success"
        else:
            t += T_CLEAR_FAIL
            self.n_clear_fail += 1
            result = "no_target_in_range"
        self.virtual_time += t                  # 先推进时钟，响应里报的是推进后的时刻
        out = self._env(True, clear_result=result)
        self._log("clear", self.pos, channel, out)
        return out

    def _log(self, kind, pos, channel, resp):
        self.action_log.append({
            "kind": kind, "pos": (float(pos[0]), float(pos[1])), "channel": int(channel),
            "virtual_time_s": self.virtual_time,
            "accepted": resp.get("accepted"),
            "result": resp.get("measure_result") or resp.get("clear_result"),
        })

    # ============================================================ 入口
    def handle(self, path: str, payload) -> Tuple[int, dict]:
        """结构校验 + 业务处理。抛 ``ProtocolError`` 表示需要非 200 状态码。"""
        path = P.normalize_path(path)
        unknown = P.check_structure(path, payload)

        # ---- 未声明字段 / arena_id / robot_id：200 + accepted=false，不占用 request_id
        if unknown:
            return 200, self._env(False)
        if payload["arena_id"] != self.arena_id:
            return 200, self._env(False)
        if payload["robot_id"] != self.robot_id:
            return 200, self._env(False)

        # ---- 幂等
        rid = payload["request_id"]
        fp = self._fingerprint(path, payload)
        if rid in self.records:
            rec = self.records[rid]
            if rec.fingerprint != fp:
                raise P.ProtocolError(409, "同一 request_id 对应了不同动作")
            return rec.status, dict(rec.response)
        if len(self.records) >= self.idem_limit:
            raise P.ProtocolError(429, "本局幂等记录达到上限")

        # ---- 时间约束
        self._check_timeouts()
        if self.window_opened_at is None:
            return 200, self._env(False)     # 接口尚未开放
        if not self.entered and self._now() > self.window_opened_at + self.window_s:
            self.ended, self.end_reason = True, "window_timeout"
            return 200, self._env(False)

        if path == "/enter":
            resp = self._do_enter(payload)
        elif path == "/exit":
            resp = self._do_exit()
        elif path == "/measure":
            resp = self._do_measure(payload["position"], payload["channel"])
        else:
            resp = self._do_clear(payload["position"], payload["channel"])

        self.records[rid] = IdemRecord(fp, 200, dict(resp))
        return 200, resp

    @staticmethod
    def _fingerprint(path: str, payload) -> Tuple:
        pos = payload.get("position")
        return (path,
                None if pos is None else (float(pos["x"]), float(pos["y"])),
                None if "channel" not in payload else int(payload["channel"]))

    # ============================================================ 报告
    def report(self) -> dict:
        return {
            "end_reason": self.end_reason,
            "entered": self.entered,
            "exited": self.exited,
            "virtual_time_s": self.virtual_time,
            "real_elapsed_s": self.elapsed_real(),
            "n_measure": self.n_measure,
            "n_switch": self.n_switch,
            "n_clear_ok": self.n_clear_ok,
            "n_clear_fail": self.n_clear_fail,
            "n_actions_total": self.n_measure + self.n_clear_ok + self.n_clear_fail,
            "n_requests_recorded": len(self.records),
            "n_sources": self.scenario.n_sources,
            "n_cleared": self.scenario.n_sources - self.scenario.remaining(),
            "n_remaining": self.scenario.remaining(),
            "avg_locate_clear_time_s": (
                self.virtual_time / max(1, self.scenario.n_sources - self.scenario.remaining())
                if self.scenario.remaining() < self.scenario.n_sources else float("nan")),
            "clear_ratio": (
                (self.scenario.n_sources - self.scenario.remaining())
                / max(1, self.scenario.n_sources)),
        }
