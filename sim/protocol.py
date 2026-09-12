# -*- coding: utf-8 -*-
"""
附件2 通信协议：请求解析与结构校验

严格执行附件2 第 5 节的全部规则：
* 路径必须精确为 /enter、/measure、/clear、/exit（不接受尾随斜线与查询参数）
* Content-Type 必须为 application/json，可带且仅可带 charset=utf-8
* Content-Encoding 省略或为 identity
* 请求体为无 BOM 的 UTF-8 JSON 对象，无重复键，嵌套 ≤16 层，≤65536 字节
* 未声明字段 → HTTP 200 且 accepted=false（用于暴露拼写错误）
* 缺失/类型/取值错误 → HTTP 400
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

KNOWN_PATHS = ("/enter", "/measure", "/clear", "/exit")
ACTION_PATHS = ("/measure", "/clear")
BASE_FIELDS = ("arena_id", "robot_id", "request_id")
POSITION_FIELDS = ("x", "y")

MAX_BODY = 65536
MAX_DEPTH = 16
COORD_LIMIT = 2_000_000.0
ARENA_ID = "default"


class ProtocolError(Exception):
    """违反协议 → 直接映射到 HTTP 状态码"""

    def __init__(self, status: int, reason: str):
        super().__init__("%d %s" % (status, reason))
        self.status = status
        self.reason = reason


def _reject_constant(name: str):
    raise ProtocolError(400, "JSON 中出现非有限数值常量 %s" % name)


def _pairs_hook(pairs):
    seen = set()
    for k, _ in pairs:
        if k in seen:
            raise ProtocolError(400, "JSON 对象含重复键 %r" % k)
        seen.add(k)
    return dict(pairs)


def parse_body(body: bytes) -> Any:
    """解析请求体（含 BOM / 大小 / 编码 / 重复键 / 嵌套深度检查）"""
    if len(body) > MAX_BODY:
        raise ProtocolError(413, "请求体超过 %d 字节" % MAX_BODY)
    if body.startswith(b"\xef\xbb\xbf"):
        raise ProtocolError(400, "请求体不得带 BOM")
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ProtocolError(400, "请求体不是合法 UTF-8: %s" % e)
    try:
        obj = json.loads(text, object_pairs_hook=_pairs_hook,
                         parse_constant=_reject_constant)
    except ProtocolError:
        raise
    except Exception as e:
        raise ProtocolError(400, "JSON 语法错误: %s" % e)
    if _depth(obj) > MAX_DEPTH:
        raise ProtocolError(400, "JSON 嵌套超过 %d 层" % MAX_DEPTH)
    return obj


def _depth(o, d: int = 0) -> int:
    if isinstance(o, dict):
        return max([d + 1] + [_depth(v, d + 1) for v in o.values()])
    if isinstance(o, list):
        return max([d + 1] + [_depth(v, d + 1) for v in o])
    return d


def check_content_type(value: str) -> None:
    """Content-Type 必须是 application/json，可带且只允许 charset=utf-8"""
    if value is None:
        raise ProtocolError(415, "缺少 Content-Type")
    parts = [p.strip() for p in value.split(";")]
    if parts[0].lower() != "application/json":
        raise ProtocolError(415, "Content-Type 必须为 application/json")
    for p in parts[1:]:
        if not p:
            continue
        if "=" not in p:
            raise ProtocolError(415, "Content-Type 参数不受支持: %r" % p)
        k, v = (x.strip() for x in p.split("=", 1))
        if k.lower() != "charset" or v.lower().strip('"') != "utf-8":
            raise ProtocolError(415, "Content-Type 参数不受支持: %r" % p)


def check_content_encoding(value: str) -> None:
    if value is None or value.strip() == "" or value.strip().lower() == "identity":
        return
    raise ProtocolError(415, "Content-Encoding 只允许省略或 identity")


def normalize_path(raw_path: str) -> str:
    """路径规范化；不在已知路径中一律 404（含尾随斜线、查询参数）"""
    if raw_path in KNOWN_PATHS:
        return raw_path
    raise ProtocolError(404, "路径不存在或不精确: %r" % raw_path)


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def check_structure(path: str, payload: Any) -> List[str]:
    """结构校验；返回「未声明字段」列表（这些字段导致 200 + accepted=false）"""
    if not isinstance(payload, dict):
        raise ProtocolError(400, "请求体必须是 JSON 对象")

    allowed = set(BASE_FIELDS)
    if path in ACTION_PATHS:
        allowed |= {"position", "channel"}
    unknown = [k for k in payload.keys() if k not in allowed]

    # ---- 必填与类型 ----
    for f in BASE_FIELDS:
        if f not in payload:
            raise ProtocolError(400, "缺少字段 %s" % f)
        if not isinstance(payload[f], str):
            raise ProtocolError(400, "字段 %s 必须是字符串" % f)
    if payload["arena_id"] != ARENA_ID:
        pass                                    # 交给业务层 → 200 + accepted=false
    for f in ("robot_id", "request_id"):
        v = payload[f]
        if not (1 <= len(v.encode("utf-8")) <= (64 if f == "robot_id" else 128)):
            raise ProtocolError(400, "字段 %s 长度不合法" % f)
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in v) or \
           any(0x200B <= ord(c) <= 0x200F or 0x202A <= ord(c) <= 0x202E or
               0x2060 <= ord(c) <= 0x206F or ord(c) == 0xFEFF for c in v):
            raise ProtocolError(400, "字段 %s 含控制字符或不可见格式字符" % f)

    if path in ACTION_PATHS:
        if "position" not in payload:
            raise ProtocolError(400, "缺少字段 position")
        if "channel" not in payload:
            raise ProtocolError(400, "缺少字段 channel")
        pos = payload["position"]
        if not isinstance(pos, dict):
            raise ProtocolError(400, "position 必须是对象")
        for f in POSITION_FIELDS:
            if f not in pos:
                raise ProtocolError(400, "缺少字段 position.%s" % f)
            if not _is_num(pos[f]):
                raise ProtocolError(400, "position.%s 不是有限数值" % f)
            if abs(float(pos[f])) > COORD_LIMIT:
                raise ProtocolError(400, "position.%s 绝对值超过 %g" % (f, COORD_LIMIT))
        for k in pos.keys():
            if k not in POSITION_FIELDS:
                unknown.append("position.%s" % k)

        ch = payload["channel"]
        if isinstance(ch, float) and ch.is_integer():
            ch = int(ch)
        if not _is_int(ch) or not (1 <= ch <= 20):
            raise ProtocolError(400, "channel 必须是 1..20 的整数")

    return unknown
