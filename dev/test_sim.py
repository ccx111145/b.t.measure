# -*- coding: utf-8 -*-
"""
本地仿真器测试：协议一致性 + 物理语义 + 附录计时示例端到端复现

  A 附件2 第10节计时示例（进程内 + HTTP 两条路径）
  B 物理语义（no_signal / near / direction / clear / 误差同点固定）
  C 协议一致性（400/404/405/409/413/415 与 200+accepted=false）
  D 幂等与重放
  E 时间约束（现实预算、虚拟限时）
"""
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.client import HttpRobotClient, LocalRobotClient          # noqa: E402
from sim.arena import generate_scenario, positional_scenario        # noqa: E402
from sim.engine import ArenaEngine                                  # noqa: E402
from sim.server import make_server                                  # noqa: E402

FAIL = []
ROBOT = "TEAM-2026-B-001"


def check(name, cond, info=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, info))
    if not cond:
        FAIL.append(name)


def raw_post(base_url, path, body: bytes,
             content_type: str = "application/json",
             content_encoding=None, method="POST"):
    req = urllib.request.Request(base_url + path, data=body, method=method)
    if content_type is not None:
        req.add_header("Content-Type", content_type)
    if content_encoding is not None:
        req.add_header("Content-Encoding", content_encoding)
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        txt = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(txt)
        except Exception:
            return e.code, {"raw": txt}


def start_server(engine):
    srv = make_server(engine, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.02},
                          daemon=True)
    th.start()
    return srv, th, "http://127.0.0.1:%d" % srv.server_address[1]


# ================================================================ A
print("=" * 78)
print("A. 附录2 第10节计时示例端到端复现（期望虚拟时刻 0 → 105 → 111 → 194 → 199）")
SC = positional_scenario([
    {"channel": 1, "x": 1500, "y": 1200, "recv_radius": 1200},
    {"channel": 2, "x": -1500, "y": -1200, "recv_radius": 1200},
])   # 频道 3 无源 → /clear 应返回 no_target_in_range

eng = ArenaEngine(SC, ROBOT)
eng.arm()
cl = LocalRobotClient(eng, ROBOT)
check("enter 成功", cl.enter().accepted)
t = []
r = cl.measure(300, 400, 1)
t.append(r.virtual_time_s)
r = cl.measure(300, 400, 2)
t.append(r.virtual_time_s)
r = cl.clear(300, 0, 3)
t.append(r.virtual_time_s)
check("clear 未发现目标", r.clear_result == "no_target_in_range", str(r.clear_result))
r = cl.measure(300, 0, 2)
t.append(r.virtual_time_s)
cl.exit()
check("虚拟时刻序列 == [105, 111, 194, 199]", t == [105.0, 111.0, 194.0, 199.0], str(t))
check("测向机频道未被 /clear 改变（第5步 0 切换）",
      abs(t[3] - t[2] - 5.0) < 1e-9, "增量 = %.1f s" % (t[3] - t[2]))

# HTTP 同一条路径
eng2 = ArenaEngine(SC, ROBOT)
eng2.arm()
srv, th, base = start_server(eng2)
cl2 = HttpRobotClient(base, ROBOT)
try:
    check("HTTP: enter 成功", cl2.enter().accepted)
    th_ = [cl2.measure(300, 400, 1).virtual_time_s,
           cl2.measure(300, 400, 2).virtual_time_s,
           cl2.clear(300, 0, 3).virtual_time_s,
           cl2.measure(300, 0, 2).virtual_time_s]
    cl2.exit()
    check("HTTP: 虚拟时刻序列一致", th_ == [105.0, 111.0, 194.0, 199.0], str(th_))
finally:
    srv.shutdown()
    srv.server_close()

# ================================================================ B
print("=" * 78)
print("B. 物理语义")
# 源的 dir_deg 表示它向哪个方向辐射；覆盖角为其两侧各 90°
SC2 = positional_scenario([
    {"channel": 1, "x": 0, "y": 0, "recv_radius": 1000, "kind": "omni"},
    {"channel": 2, "x": 1400, "y": 0, "recv_radius": 1000, "kind": "omni"},
    {"channel": 3, "x": 0, "y": 800, "recv_radius": 1200,
     "kind": "dir", "dir_deg": 90.0},                     # 向正北辐射
    {"channel": 4, "x": 0, "y": -800, "recv_radius": 1200,
     "kind": "dir", "dir_deg": 90.0},                     # 向正北辐射
])
eng3 = ArenaEngine(SC2, ROBOT)
eng3.arm()
c3 = LocalRobotClient(eng3, ROBOT)
c3.enter()
check("频道5 无源 → no_signal", c3.measure(0, 0, 5).measure_result == "no_signal")
check("频道2 从 (0,0) 测：距离 1400 > R=1000 → no_signal",
      c3.measure(0, 0, 2).measure_result == "no_signal")
check("频道2 走近到 (1300,0)：距离 100 → direction",
      c3.measure(1300, 0, 2).measure_result == "direction")
r = c3.measure(0, 1300, 3)
check("频道3 定向源北侧（覆盖内）→ direction",
      r.measure_result == "direction", str(r.measure_result))
check("频道3 示向度 = 由检测点指向源的真实方位角 270°",
      r.svd_deg is not None and abs(((r.svd_deg - 270 + 180) % 360) - 180) <= 1.0,
      "svd=%.2f" % (r.svd_deg if r.svd_deg is not None else -1))
r = c3.measure(0, 300, 3)
check("频道3 定向源南侧（覆盖外）→ no_signal",
      r.measure_result == "no_signal", str(r.measure_result))
r = c3.measure(0, -300, 4)
check("频道4 位于其北侧（覆盖内）→ direction",
      r.measure_result == "direction", str(r.measure_result))
r = c3.measure(0, -1300, 4)
check("频道4 位于其南侧（覆盖外）→ no_signal",
      r.measure_result == "no_signal", str(r.measure_result))
check("距离 3 m 且在覆盖内 → near",
      c3.measure(0, 3, 1).measure_result == "near")
check("距离 4 m → near", c3.measure(0, 4, 1).measure_result == "near")
check("距离 7 m → direction（超过 5 m 阈值）",
      c3.measure(0, 7, 1).measure_result == "direction")

# 误差：同点重复检测读数完全相同
a = c3.measure(500, 500, 2)
b = c3.measure(500, 500, 2)
check("同点重复检测读数完全一致（误差固定）", a.svd_deg == b.svd_deg,
      "%.2f vs %.2f" % (a.svd_deg or -1, b.svd_deg or -1))
# 误差幅度 ≤1°
worst = 0.0
for ch, (sx, sy) in [(2, (1400, 0)), (3, (0, 800)), (4, (0, -800))]:
    src = SC2.by_channel(ch)
    for pt in [(200, 200), (1300, 100), (600, -400), (900, 900), (-800, 700)]:
        if not src.detectable(pt):
            continue
        tb = math.degrees(math.atan2(sy - pt[1], sx - pt[0])) % 360
        e = eng3.svd_error(pt, ch)
        worst = max(worst, abs(e))
        eng3.pos = pt
        got = eng3._do_measure({"x": pt[0], "y": pt[1]}, ch).get("svd_deg")
        if got is not None:
            worst = max(worst, abs((got - tb + 180) % 360 - 180))
check("示向度误差始终 ≤1°", worst <= 1.0 + 1e-9, "实测最大 %.4f°" % worst)

# 清除
r = c3.clear(0, 19, 1)
check("距频道1 源 19 m → success", r.clear_result == "success", str(r.clear_result))
r = c3.clear(0, 19, 1)
check("重复清除同一源 → no_target_in_range",
      r.clear_result == "no_target_in_range", str(r.clear_result))
r = c3.clear(0, 785, 3)
check("定向源在其覆盖范围之外仍可清除（clear 与角度无关）",
      r.clear_result == "success", str(r.clear_result))
r = c3.clear(0, -780, 4)
check("距频道4 源 20 m → success（边界含等号）",
      r.clear_result == "success", str(r.clear_result))
r = c3.clear(1400, 10, 2)
check("距频道2 源 10 m → success", r.clear_result == "success")
r = c3.clear(0, 0, 4)
check("距源 800 m → no_target_in_range", r.clear_result == "no_target_in_range")
check("全部 4 个源已清除", eng3.scenario.remaining() == 0,
      "剩余 %d" % eng3.scenario.remaining())

# ================================================================ C
print("=" * 78)
print("C. 协议一致性（HTTP）")
eng4 = ArenaEngine(positional_scenario([{"channel": 1, "x": 0, "y": 0}]), ROBOT)
eng4.arm()
srv4, th4, base4 = start_server(eng4)
try:
    B = ROBOT
    ok_enter = json.dumps({"arena_id": "default", "robot_id": B,
                           "request_id": "e1"}).encode()
    st, r = raw_post(base4, "/enter", ok_enter)
    check("正常 /enter → 200 accepted=true", st == 200 and r.get("accepted") is True,
          "%s %s" % (st, r.get("accepted")))
    check("响应含 max_virtual_duration_s / max_real_duration_s / remaining_real_duration_s",
          all(k in r for k in ("max_virtual_duration_s", "max_real_duration_s",
                               "remaining_real_duration_s")), str(sorted(r.keys())))

    st, _ = raw_post(base4, "/measure", b"{}", method="GET")
    check("GET 已知路径 → 405", st == 405, str(st))
    st, _ = raw_post(base4, "/measure/", ok_enter)
    check("尾随斜线 → 404", st == 404, str(st))
    st, _ = raw_post(base4, "/enter?x=1", ok_enter)
    check("带查询参数 → 404", st == 404, str(st))
    st, _ = raw_post(base4, "/unknown", ok_enter)
    check("未知路径 → 404", st == 404, str(st))
    st, _ = raw_post(base4, "/enter", ok_enter, content_type="text/plain")
    check("Content-Type=text/plain → 415", st == 415, str(st))
    st, _ = raw_post(base4, "/enter", ok_enter,
                     content_type="application/json; charset=gbk")
    check("charset=gbk → 415", st == 415, str(st))
    st, _ = raw_post(base4, "/enter", ok_enter, content_encoding="gzip")
    check("Content-Encoding=gzip → 415", st == 415, str(st))
    st, _ = raw_post(base4, "/enter", ok_enter,
                     content_type="application/json; charset=utf-8")
    check("charset=utf-8 → 接受", st == 200, str(st))

    dup = b'{"arena_id":"default","arena_id":"default","robot_id":"t","request_id":"d1"}'
    st, _ = raw_post(base4, "/enter", dup)
    check("重复键 → 400", st == 400, str(st))
    st, _ = raw_post(base4, "/measure",
                     json.dumps({"arena_id": "default", "robot_id": B,
                                 "request_id": "m-bad", "position": {"x": 0, "y": 0},
                                 "channel": 1.5}).encode())
    check("channel=1.5 → 400", st == 400, str(st))
    st, _ = raw_post(base4, "/measure",
                     b'{"arena_id":"default","robot_id":"' + B.encode() +
                     b'","request_id":"m1","channel":1}')
    check("缺少 position → 400", st == 400, str(st))
    st, _ = raw_post(base4, "/measure",
                     b'{"arena_id":"default","robot_id":"' + B.encode() +
                     b'","request_id":"m1","position":{"x":NaN,"y":0},"channel":1}')
    check("NaN 坐标 → 400", st == 400, str(st))
    st, _ = raw_post(base4, "/measure",
                     b'{"arena_id":"default","robot_id":"' + B.encode() +
                     b'","request_id":"m1","position":{"x":1e9,"y":0},"channel":1}')
    check("坐标超 2e6 → 400", st == 400, str(st))
    st, _ = raw_post(base4, "/measure",
                     b'{"arena_id":"default","robot_id":"' + B.encode() +
                     b'","request_id":"m1","position":{"x":0,"y":0},"channel":1,'
                     b'"typo_field":1}')
    check("未声明字段 → HTTP 200", st == 200, str(st))
    st, r = raw_post(base4, "/measure",
                     b'{"arena_id":"default","robot_id":"' + B.encode() +
                     b'","request_id":"m1","position":{"x":0,"y":0},"channel":1,'
                     b'"typo_field":1}')
    check("未声明字段 → accepted=false", r.get("accepted") is False, str(r.get("accepted")))
    check("未声明字段响应 virtual_time_s=0", r.get("virtual_time_s") == 0)
    st, r = raw_post(base4, "/measure",
                     b'{"arena_id":"default","robot_id":"' + B.encode() +
                     b'","request_id":"m1","position":{"x":0,"y":0},"channel":1,'
                     b'"position_extra":2}')
    check("position 以外的顶层未声明字段同样 accepted=false",
          st == 200 and r.get("accepted") is False)
    st, r = raw_post(base4, "/measure",
                     b'{"arena_id":"default","robot_id":"' + B.encode() +
                     b'","request_id":"m1","position":{"x":0,"y":0,"z":1},"channel":1}')
    check("position 下的未声明字段 → accepted=false",
          st == 200 and r.get("accepted") is False, str(st))
    st, r = raw_post(base4, "/measure",
                     b'{"arena_id":"other","robot_id":"' + B.encode() +
                     b'","request_id":"m1","position":{"x":0,"y":0},"channel":1}')
    check("arena_id 不匹配 → 200 accepted=false",
          st == 200 and r.get("accepted") is False, str(st))
    st, r = raw_post(base4, "/measure",
                     b'{"arena_id":"default","robot_id":"someone-else",'
                     b'"request_id":"m1","position":{"x":0,"y":0},"channel":1}')
    check("robot_id 不匹配 → 200 accepted=false",
          st == 200 and r.get("accepted") is False, str(st))

    big = json.dumps({"arena_id": "default", "robot_id": B, "request_id": "big",
                      "position": {"x": 0, "y": 0}, "channel": 1,
                      "pad": "x" * 70000}).encode()
    # 服务端在丢弃超大请求体后才回 413；Windows 上客户端偶发抢先收到 RST
    # （WinError 10053）。两种表现都表示"服务端拒绝了超大请求体"，都算通过。
    try:
        st, _ = raw_post(base4, "/measure", big)
    except ConnectionAbortedError:
        st = 413
    check("请求体 >65536 字节 → 413", st == 413, str(st))

    # 幂等
    p1 = json.dumps({"arena_id": "default", "robot_id": B, "request_id": "idem-1",
                     "position": {"x": 100, "y": 0}, "channel": 1}).encode()
    st1, r1 = raw_post(base4, "/measure", p1)
    vt_after_1 = r1["virtual_time_s"]
    st2, r2 = raw_post(base4, "/measure", p1)
    check("同 id 同内容 → 重放，虚拟时钟不重复推进",
          r1 == r2 and st1 == st2 == 200, "vt %s → %s" % (vt_after_1, r2["virtual_time_s"]))
    check("重放响应与首次完全相同（含 real_timestamp_ms）", r1 == r2)
    p2 = json.dumps({"arena_id": "default", "robot_id": B, "request_id": "idem-1",
                     "position": {"x": 200, "y": 0}, "channel": 1}).encode()
    st3, _ = raw_post(base4, "/measure", p2)
    check("同 id 改内容 → 409", st3 == 409, str(st3))
    # 结构错误不占用 id
    p4 = json.dumps({"arena_id": "default", "robot_id": B, "request_id": "reuse-1",
                     "position": {"x": 0, "y": 0}, "channel": 99}).encode()
    st4, _ = raw_post(base4, "/measure", p4)
    st5, r5 = raw_post(base4, "/measure",
                       json.dumps({"arena_id": "default", "robot_id": B,
                                   "request_id": "reuse-1",
                                   "position": {"x": 0, "y": 0},
                                   "channel": 2}).encode())
    check("400 不占用 request_id，修正后可复用",
          st4 == 400 and st5 == 200 and r5.get("accepted") is True, "%s/%s" % (st4, st5))
finally:
    srv4.shutdown()
    srv4.server_close()

# ================================================================ D
print("=" * 78)
print("D. 业务状态与错误码")
eng5 = ArenaEngine(positional_scenario([{"channel": 1, "x": 0, "y": 0}]), ROBOT)
eng5.arm()
srv5, th5, base5 = start_server(eng5)
try:
    B = ROBOT

    def act(rid, x=0.0, y=0.0, ch=1, path="/measure"):
        return raw_post(base5, path, json.dumps(
            {"arena_id": "default", "robot_id": B, "request_id": rid,
             "position": {"x": x, "y": y}, "channel": ch}).encode())

    st, r = act("pre-1")
    check("/enter 之前 /measure → 200 accepted=false",
          st == 200 and r.get("accepted") is False, "%s %s" % (st, r.get("accepted")))
    raw_post(base5, "/enter", json.dumps({"arena_id": "default", "robot_id": B,
                                          "request_id": "e"}).encode())
    st, r = raw_post(base5, "/enter", json.dumps({"arena_id": "default", "robot_id": B,
                                                  "request_id": "e2"}).encode())
    check("重复 /enter → accepted=false", r.get("accepted") is False)
    st, r = act("a1", 0, 0, 1)
    check("正常 measure accepted=true", r.get("accepted") is True)
    st, r = raw_post(base5, "/exit", json.dumps({"arena_id": "default", "robot_id": B,
                                                 "request_id": "x"}).encode())
    check("/exit → exit_reason=user_exit", r.get("exit_reason") == "user_exit")
    st, r = act("a2", 0, 0, 1)
    check("退出后动作 → accepted=false", r.get("accepted") is False)
finally:
    srv5.shutdown()
    srv5.server_close()

# ================================================================ E
print("=" * 78)
print("E. 时间约束（用可控时钟）")
clock = {"t": 0.0}
sc_e = positional_scenario([{"channel": 1, "x": 0, "y": 0}])
engE = ArenaEngine(sc_e, ROBOT, clock=lambda: clock["t"])
engE.arm()
clE = LocalRobotClient(engE, ROBOT)
r = clE.enter()
check("窗口刚打开时 remaining_real_duration_s=1200",
      r.body.get("remaining_real_duration_s") == 1200, str(r.body.get("remaining_real_duration_s")))
engF = ArenaEngine(sc_e, ROBOT, clock=lambda: clock["t"])
clock["t"] = 0.0
engF.arm()
clock["t"] = 300.0        # 窗口开始 5 分钟后才 /enter
clF = LocalRobotClient(engF, ROBOT)
r = clF.enter()
check("晚 300 s 进入 → remaining_real_duration_s=1200（窗口 1500 s 内）",
      r.body.get("remaining_real_duration_s") == 1200, str(r.body.get("remaining_real_duration_s")))
clock["t"] = 600.0
engG = ArenaEngine(sc_e, ROBOT, clock=lambda: clock["t"])
clock["t"] = 0.0
engG.arm()
clock["t"] = 600.0        # 窗口开始 10 分钟后才 /enter
clG = LocalRobotClient(engG, ROBOT)
r = clG.enter()
check("晚 600 s 进入 → remaining=min(1200, 1500-600)=900",
      r.body.get("remaining_real_duration_s") == 900, str(r.body.get("remaining_real_duration_s")))
clock["t"] = 600.0 + 901.0
r = clG.measure(0, 0, 1)
check("超出实时预算 → accepted=false（并标记 real_timeout）",
      r.accepted is False and engG.end_reason == "real_timeout",
      "%s / %s" % (r.accepted, engG.end_reason))

# 虚拟限时
engH = ArenaEngine(sc_e, ROBOT, max_virtual=200.0)
engH.arm()
clH = LocalRobotClient(engH, ROBOT)
clH.enter()
clH.measure(0, 1000, 2)          # 200 s 移动 + 5 s（ch2 无源 → no_signal），越过 200 s
check("越限动作本身允许完成（截止前已登记）",
      engH.virtual_time > 200.0 and not engH.ended,
      "vt=%.1f ended=%s" % (engH.virtual_time, engH.ended))
r = clH.measure(0, 0, 1)
check("下一次请求触发虚拟超时", engH.ended and engH.end_reason == "virtual_timeout",
      "reason=%s" % engH.end_reason)
check("虚拟超时后动作 → accepted=false", r.accepted is False)

# ================================================================ F
print("=" * 78)
print("F. 场景生成与统计量")
sc = generate_scenario(seed=42, directional_ratio=0.5)
check("源数在 10..16", 10 <= sc.n_sources <= 16, "n=%d" % sc.n_sources)
check("频道互不相同", len(set(s.channel for s in sc.sources)) == sc.n_sources)
check("接收半径都在 1000..1500", all(1000 <= s.recv_radius <= 1500 for s in sc.sources))
check("所有源在目标区域内",
      all(math.hypot(s.x, s.y) <= 1800 + 1e-9 for s in sc.sources))
check("定向源有方向、全向源无方向",
      all((s.dir_deg is not None) == (s.kind == "dir") for s in sc.sources))
check("定向占比约 1/2", abs(sc.summary()["n_dir"] / sc.n_sources - 0.5) < 0.2,
      str(sc.summary()))

engI = ArenaEngine(generate_scenario(seed=7), ROBOT)
engI.arm()
clI = LocalRobotClient(engI, ROBOT)
clI.enter()
clI.measure(0, 0, 1)
rep = engI.report()
check("report 含平均定位清除时间与清除比例",
      "avg_locate_clear_time_s" in rep and "clear_ratio" in rep)
check("action_log 记录指令序列", len(engI.action_log) >= 1)

print("=" * 78)
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
print("全部测试通过")
