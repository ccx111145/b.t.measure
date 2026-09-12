# -*- coding: utf-8 -*-
"""
HTTP + JSON 服务：把 :class:`sim.engine.ArenaEngine` 暴露成附件2 规定的 4 条接口

默认监听 127.0.0.1:2026（与真实模拟器一致，便于机器狗程序零改动切换）。
所有能形成 HTTP 响应的错误都返回 JSON，且至少包含 accepted / real_timestamp_ms / virtual_time_s。
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from . import protocol as P
from .engine import ArenaEngine

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 2026


def _json_bytes(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class SimHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, engine: ArenaEngine):
        super().__init__(addr, _Handler)
        self.engine = engine
        self.busy = threading.Lock()          # 并发不同动作 → 409
        self.request_count = 0
        self.error_count = 0


class _Handler(BaseHTTPRequestHandler):
    server_version = "RadioJammerSim/1.0"
    protocol_version = "HTTP/1.1"
    sys_version = ""

    # ---- 输出 ----
    def log_message(self, fmt, *args):        # 静音默认日志
        pass

    def _send(self, status: int, obj, close: bool = False) -> None:
        body = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        if close:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _error(self, status: int, reason: str = "", close: bool = False) -> None:
        eng: ArenaEngine = self.server.engine
        self.server.error_count += 1
        self._send(status, {"accepted": False,
                            "real_timestamp_ms": int(eng._wall_ms()),
                            "virtual_time_s": 0,
                            "error": reason}, close=close)

    # ---- 方法 ----
    def do_POST(self):
        self._dispatch(need_post=True)

    def do_GET(self):
        self._dispatch(need_post=False)

    do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = do_GET

    # ---- 主流程 ----
    def _dispatch(self, need_post: bool):
        srv: SimHTTPServer = self.server
        # 1) 路径
        try:
            path = P.normalize_path(self.path)
        except P.ProtocolError as e:
            return self._error(e.status, e.reason)
        # 2) 方法
        if not need_post or self.command != "POST":
            return self._error(405, "已知路径必须使用 POST")
        # 3) Content-Type / Content-Encoding
        try:
            P.check_content_type(self.headers.get("Content-Type"))
            P.check_content_encoding(self.headers.get("Content-Encoding"))
        except P.ProtocolError as e:
            return self._error(e.status, e.reason)
        # 4) 长度
        try:
            n = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._error(400, "Content-Length 非法")
        if n > P.MAX_BODY:
            # 先把请求体读干净再回包：若在客户端仍在发送时就关闭连接，
            # 客户端会收到 ECONNABORTED（WinError 10053）而不是 413。
            # 这里完整读取 Content-Length 指定的字节数（BufferedReader.read 会读满），
            # 仅在异常巨大的情况下截断。
            try:
                to_drain = min(n, P.MAX_BODY + 8 * (1 << 20))
                got = 0
                while got < to_drain:
                    chunk = self.rfile.read(min(65536, to_drain - got))
                    if not chunk:
                        break
                    got += len(chunk)
            except Exception:                                    # noqa: BLE001
                pass
            time.sleep(0.02)          # 给客户端把剩余字节写完的时间窗
            return self._error(413, "请求体超过 %d 字节" % P.MAX_BODY, close=True)
        body = self.rfile.read(n) if n > 0 else b""
        # 5) 并发保护
        if not srv.busy.acquire(blocking=False):
            return self._error(409, "并发发送了不同动作")
        try:
            srv.request_count += 1
            payload = P.parse_body(body)
            status, resp = srv.engine.handle(path, payload)
            self._send(status, resp)
        except P.ProtocolError as e:
            self._error(e.status, e.reason)
        except Exception as e:                      # noqa: BLE001
            self._error(500, "内部错误: %r" % (e,))
        finally:
            srv.busy.release()


def make_server(engine: ArenaEngine, host: str = DEFAULT_HOST,
                port: int = DEFAULT_PORT) -> SimHTTPServer:
    return SimHTTPServer((host, port), engine)


def serve(engine: ArenaEngine, host: str = DEFAULT_HOST,
          port: int = DEFAULT_PORT, ready: Optional[threading.Event] = None):
    srv = make_server(engine, host, port)
    if ready is not None:
        ready.set()
    try:
        srv.serve_forever(poll_interval=0.05)
    finally:
        srv.server_close()

