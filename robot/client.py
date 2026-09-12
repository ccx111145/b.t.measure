# -*- coding: utf-8 -*-
"""
机器狗侧通信客户端

严格实现附件2 第 12 节的编程注意事项：
* 使用当前登录参赛队号作为 ``robot_id``；
* 每个新动作用新的 ``request_id``；**只在重试完全相同的动作时复用原 request_id 与原请求内容**；
* 逐次等待响应，不并发发送不同动作；
* 同时检查 HTTP 状态与 ``accepted``；
* 使用 ``/enter`` 返回的 ``remaining_real_duration_s`` 控制现实运行时间。

提供两个实现，接口一致：
* :class:`HttpRobotClient`  —— 走 HTTP+JSON，用于真实模拟器与协议一致性测试；
* :class:`LocalRobotClient` —— 进程内直调引擎，用于批量演练（无 HTTP 开销）。
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

DEFAULT_BASE_URL = "http://127.0.0.1:2026"
ARENA_ID = "default"


@dataclass
class ActionResult:
    """一次动作的统一结果"""

    kind: str
    http_status: int
    body: Dict[str, Any] = field(default_factory=dict)
    transport_ok: bool = True
    error: Optional[str] = None
    request_id: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    attempts: int = 1

    @property
    def accepted(self) -> bool:
        return self.transport_ok and self.body.get("accepted") is True

    @property
    def measure_result(self) -> Optional[str]:
        return self.body.get("measure_result")

    @property
    def svd_deg(self) -> Optional[float]:
        if self.measure_result != "direction":
            return None                     # 只有 direction 时才读 svd_deg
        return self.body.get("svd_deg")

    @property
    def clear_result(self) -> Optional[str]:
        return self.body.get("clear_result")

    @property
    def virtual_time_s(self) -> float:
        return float(self.body.get("virtual_time_s", 0.0))

    def __repr__(self):
        return ("ActionResult(%s, http=%s, accepted=%s, result=%s)"
                % (self.kind, self.http_status, self.body.get("accepted"),
                   self.measure_result or self.clear_result or ""))


class BaseRobotClient:
    """公共逻辑：request_id 生成、动作日志、现实时间预算"""

    def __init__(self, robot_id: str, arena_id: str = ARENA_ID,
                 max_retries: int = 3, retry_wait_s: float = 0.05):
        self.robot_id = robot_id
        self.arena_id = arena_id
        self.max_retries = int(max_retries)
        self.retry_wait_s = float(retry_wait_s)
        self._seq = 0
        self.log: List[Dict[str, Any]] = []
        self.remaining_real_duration_s: Optional[int] = None
        self._enter_wall: Optional[float] = None
        self.n_requests = 0

    # ---- request_id ----
    def next_request_id(self, kind: str) -> str:
        self._seq += 1
        return "%s-%d" % (kind, self._seq)

    def _base_payload(self, kind: str) -> Dict[str, Any]:
        return {"arena_id": self.arena_id, "robot_id": self.robot_id,
                "request_id": self.next_request_id(kind)}

    def _action_payload(self, kind: str, x: float, y: float, channel: int):
        p = self._base_payload(kind)
        p["position"] = {"x": float(x), "y": float(y)}
        p["channel"] = int(channel)
        return p

    # ---- 子类实现 ----
    def _post(self, path: str, payload: dict) -> ActionResult:
        raise NotImplementedError

    # ---- 对外动作 ----
    def enter(self) -> ActionResult:
        r = self._post("/enter", self._base_payload("enter"))
        if r.accepted:
            self.remaining_real_duration_s = int(
                r.body.get("remaining_real_duration_s", 0))
            self._enter_wall = time.monotonic()
        self._record("/enter", r)
        return r

    def measure(self, x: float, y: float, channel: int) -> ActionResult:
        r = self._post("/measure", self._action_payload("measure", x, y, channel))
        self._record("/measure", r)
        return r

    def clear(self, x: float, y: float, channel: int) -> ActionResult:
        r = self._post("/clear", self._action_payload("clear", x, y, channel))
        self._record("/clear", r)
        return r

    def exit(self) -> ActionResult:
        r = self._post("/exit", self._base_payload("exit"))
        self._record("/exit", r)
        return r

    # ---- 现实时间预算 ----
    def wall_elapsed(self) -> float:
        return 0.0 if self._enter_wall is None else time.monotonic() - self._enter_wall

    def wall_budget_left(self) -> float:
        """保守估计：本局还能用多少现实秒（未知时返回 inf）"""
        if self.remaining_real_duration_s is None:
            return float("inf")
        return self.remaining_real_duration_s - self.wall_elapsed()

    # ---- 日志 ----
    def _record(self, path: str, r: ActionResult):
        self.log.append({
            "path": path,
            "request_id": r.request_id,
            "attempts": r.attempts,
            "http_status": r.http_status,
            "transport_ok": r.transport_ok,
            "request": r.payload,
            "response": r.body,
            "error": r.error,
        })

    def dump_log(self, path: str):
        d = os.path.dirname(os.path.abspath(path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.log, f, ensure_ascii=False, indent=1)


class LocalRobotClient(BaseRobotClient):
    """进程内直调引擎（批量演练用），不经过 HTTP"""

    def __init__(self, engine, robot_id: Optional[str] = None, **kw):
        super().__init__(robot_id or engine.robot_id, **kw)
        self.engine = engine

    def _post(self, path: str, payload: dict) -> ActionResult:
        self.n_requests += 1
        try:
            status, body = self.engine.handle(path, payload)
            return ActionResult(path, status, body, request_id=payload["request_id"],
                                payload=payload)
        except Exception as e:                                  # noqa: BLE001
            status = getattr(e, "status", 0)
            return ActionResult(path, status, {}, transport_ok=False,
                                error=repr(e), request_id=payload["request_id"],
                                payload=payload)


class HttpRobotClient(BaseRobotClient):
    """HTTP+JSON 客户端（真实模拟器 / 协议测试）"""

    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 robot_id: str = "<参赛队号>", timeout_s: float = 5.0, **kw):
        super().__init__(robot_id, **kw)
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)

    def _post(self, path: str, payload: dict) -> ActionResult:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            self.n_requests += 1
            req = urllib.request.Request(
                self.base_url + path, data=data,
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    return ActionResult(path, resp.status, body, request_id=payload["request_id"],
                                        payload=payload, attempts=attempt)
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "replace")
                try:
                    body = json.loads(raw)
                except Exception:                                # noqa: BLE001
                    body = {"accepted": False, "raw": raw}
                # 4xx/5xx 属于"已收到明确响应"，不重试
                return ActionResult(path, e.code, body, request_id=payload["request_id"],
                                    payload=payload, attempts=attempt,
                                    error="HTTP %d" % e.code)
            except Exception as e:                               # 连接失败 / 超时
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_wait_s)
        return ActionResult(path, 0, {}, transport_ok=False, error=repr(last_err),
                            request_id=payload["request_id"], payload=payload,
                            attempts=self.max_retries)
