# -*- coding: utf-8 -*-
"""
robot/main.py（机器狗主程序）的端到端测试

在同进程内起一个线程版模拟器服务，然后直接调用 `main()`：
  M1 连通性探测（--probe）无副作用：不进入目标区域、不消耗测试机会
  M2 问题3 全流程走真实 HTTP：/enter → 策略 → /exit，并写出表 1 记录
  M3 问题4 全流程走真实 HTTP
  M4 写出的动作日志结构正确（含 request/response，可供支撑材料）
  M5 等待接口就绪逻辑（wait_s 内接口未开放时应超时退出而不是死等）
"""
import json
import os
import shutil
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot import main as robot_main                      # noqa: E402
from sim.arena import generate_scenario                   # noqa: E402
from sim.engine import ArenaEngine                        # noqa: E402
from sim.server import make_server                        # noqa: E402

FAIL = []
TMP = tempfile.mkdtemp(prefix="b_sim_e2e_")


def check(name, cond, info=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, info))
    if not cond:
        FAIL.append(name)


def start_server(seed, directional_ratio, robot_id):
    sc = generate_scenario(seed=seed, directional_ratio=directional_ratio)
    eng = ArenaEngine(sc, robot_id)
    eng.arm()
    srv = make_server(eng, "127.0.0.1", 0)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01},
                          daemon=True)
    th.start()
    return srv, eng, "http://127.0.0.1:%d" % srv.server_address[1]


print("=" * 78)
print("M1 连通性探测（--probe）无副作用")
srv, eng, base = start_server(9, 0.5, "TEAM-CLI")
try:
    rc = robot_main.main(["--probe", "--allow-placeholder-id", "--base-url", base, "--robot-id", "TEAM-CLI"])
    check("--probe 返回 0", rc == 0, "rc=%d" % rc)
    check("探测过程未进入目标区域（entered=False）", eng.entered is False,
          "entered=%s" % eng.entered)
    check("探测过程未推进虚拟时钟", eng.virtual_time == 0.0,
          "vt=%.1f" % eng.virtual_time)
finally:
    srv.shutdown()
    srv.server_close()

print("=" * 78)
print("M2 问题3 全流程（真实 HTTP）")
srv, eng, base = start_server(9, 0.0, "TEAM-CLI")
try:
    out = os.path.join(TMP, "q3")
    rc = robot_main.main(["--problem", "3", "--allow-placeholder-id", "--base-url", base, "--robot-id", "TEAM-CLI",
                          "--no-wait", "--tag", "e2e_q3", "--out-dir", out,
                          "--case-code", "CASE-Q3-01", "--quiet"])
    check("main 返回 0", rc == 0, "rc=%d" % rc)
    rep = json.load(open(os.path.join(out, "e2e_q3_report.json"), encoding="utf-8"))
    check("表 1 记录含案例编码", rep["case_code"] == "CASE-Q3-01")
    check("清除干扰源个数 == 干扰源总数",
          rep["清除干扰源个数"] == rep["干扰源总数(本机真值, 仅本地模拟器可得)"],
          "%s / %s" % (rep["清除干扰源个数"], rep["干扰源总数(本机真值, 仅本地模拟器可得)"]))
    check("平均定位清除时间 > 0", rep["平均定位清除时间_s"] > 0,
          "%.1f s" % rep["平均定位清除时间_s"])
    check("未结案频道为空", rep["未结案频道"] == [])
    log = json.load(open(os.path.join(out, "e2e_q3_log.json"), encoding="utf-8"))
    check("动作日志非空且含 request/response",
          len(log) > 0 and "request" in log[0] and "response" in log[0],
          "%d 条" % len(log))
    check("日志首条为 /enter", log[0]["path"] == "/enter")
    check("日志末条为 /exit", log[-1]["path"] == "/exit")
finally:
    srv.shutdown()
    srv.server_close()

print("=" * 78)
print("M3 问题4 全流程（真实 HTTP）")
srv, eng, base = start_server(5, 1.0, "TEAM-CLI")
try:
    out = os.path.join(TMP, "q4")
    rc = robot_main.main(["--problem", "4", "--allow-placeholder-id", "--base-url", base, "--robot-id", "TEAM-CLI",
                          "--no-wait", "--tag", "e2e_q4", "--out-dir", out,
                          "--case-code", "CASE-Q4-01", "--quiet"])
    check("main 返回 0", rc == 0, "rc=%d" % rc)
    rep = json.load(open(os.path.join(out, "e2e_q4_report.json"), encoding="utf-8"))
    check("问题4 全部清除",
          rep["清除干扰源个数"] == rep["干扰源总数(本机真值, 仅本地模拟器可得)"],
          "%s / %s" % (rep["清除干扰源个数"], rep["干扰源总数(本机真值, 仅本地模拟器可得)"]))
    check("问题4 未结案频道为空", rep["未结案频道"] == [])
    check("问题4 走 hex 巡测网（动作数 > 500）", rep["动作数"] > 500,
          "%d" % rep["动作数"])
finally:
    srv.shutdown()
    srv.server_close()

print("=" * 78)
print("M5 等待接口就绪：接口未开放时应超时退出")
srv, eng, base = start_server(9, 0.0, "TEAM-CLI")
srv.shutdown()
srv.server_close()
time.sleep(0.2)
t0 = time.monotonic()
try:
    rc = robot_main.main(["--problem", "3", "--allow-placeholder-id", "--base-url", base, "--robot-id", "TEAM-CLI",
                          "--wait-s", "2", "--tag", "timeout_test", "--out-dir", TMP,
                          "--quiet"])
    got = ("returned", rc)
except TimeoutError as e:
    got = ("timeout", str(e))
dt = time.monotonic() - t0
check("服务不在时应超时退出而非死等", got[0] == "timeout" and 1.5 < dt < 8.0,
      "%s，耗时 %.1f s" % (got, dt))

print("=" * 78)
shutil.rmtree(TMP, ignore_errors=True)
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
print("全部测试通过")

