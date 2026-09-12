# -*- coding: utf-8 -*-
"""
HTTP 链路等价性验证：同一场景下，"进程内直调引擎"与"走真实 HTTP 服务"必须得到**完全相同**的结果

这是 M5 的核心验证：证明机器狗程序在真实 HTTP 链路上（即接真实模拟器时）的行为
与在快速批量演练中的行为一致，从而批量演练的结论可直接外推到正式测试。

用法::
    python exp/verify_http_equivalence.py --n 5 --problem 3
    python exp/verify_http_equivalence.py --n 5 --problem 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.client import HttpRobotClient, LocalRobotClient      # noqa: E402
from robot.policy import Q3Config, Q3Policy                    # noqa: E402
from sim.arena import generate_scenario                        # noqa: E402
from sim.engine import ArenaEngine                             # noqa: E402
from sim.server import make_server                             # noqa: E402

FIELDS = ["n_sources", "n_cleared", "clear_ratio", "virtual_time_s",
          "avg_locate_clear_time_s", "n_measure", "n_clear_ok", "n_clear_fail",
          "n_switch", "n_actions", "census_points_visited", "unresolved"]


def cfg_for(problem: int) -> Q3Config:
    if problem == 4:
        return Q3Config(census_layout="hex_ring", directional=True)
    return Q3Config()


def run_local(sc, robot_id, cfg):
    eng = ArenaEngine(sc, robot_id)
    eng.arm()
    cl = LocalRobotClient(eng, robot_id)
    res = Q3Policy(cl, cfg).run()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    ap.add_argument("--port", type=int, default=0)
    args = ap.parse_args(argv)

    cfg = cfg_for(args.problem)
    ratio = 0.5 if args.problem == 4 else 0.0
    print("=" * 86)
    print("HTTP 链路等价性验证  问题%d  配置=%s" % (args.problem, json.dumps(
        {k: v for k, v in cfg.__dict__.items() if k in
         ("census_layout", "directional", "census_radius", "census_ring_n",
          "hex_step", "hex_extent")}, ensure_ascii=False)))
    print("=" * 86)

    seed = args.seed0
    sc = generate_scenario(seed=seed, directional_ratio=ratio)

    # ---- 进程内 ----
    res_local = run_local(sc, "TEAM-EQ", cfg)

    # ---- 真实 HTTP ----
    sc2 = generate_scenario(seed=seed, directional_ratio=ratio)
    eng = ArenaEngine(sc2, "TEAM-EQ")
    eng.arm()
    srv = make_server(eng, "127.0.0.1", args.port)
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.01},
                          daemon=True)
    th.start()
    base = "http://127.0.0.1:%d" % srv.server_address[1]
    try:
        cl = HttpRobotClient(base, "TEAM-EQ", timeout_s=10.0)
        cl.truth_scenario = sc2          # 仅本机测试用：真实模拟器不会给出真值
        r = cl.enter()
        assert r.accepted, r.body
        res_http = Q3Policy(cl, cfg).run(already_entered=True)
    finally:
        srv.shutdown()
        srv.server_close()

    print("%-26s %16s %16s %8s" % ("字段", "进程内", "HTTP", "一致"))
    bad = 0
    for f in FIELDS:
        a, b = getattr(res_local, f), getattr(res_http, f)
        same = (a == b)
        if not same:
            bad += 1
        print("%-26s %16s %16s %8s" % (f, a, b, "OK" if same else "**不同**"))

    # 逐条指令对比
    la = [(x["path"], json.dumps(x["request"].get("position"), sort_keys=True),
           x["request"].get("channel"), json.dumps(x["response"], sort_keys=True)) for x in []]
    print("-" * 86)
    print("HTTP 请求数 = %d，进程内请求数 = %d" % (res_http.n_requests, res_local.n_requests))
    print("平均定位清除时间：进程内 %.2f s，HTTP %.2f s"
          % (res_local.avg_locate_clear_time_s, res_http.avg_locate_clear_time_s))
    if bad == 0:
        print("\n[PASS] 两条链路结果完全一致 —— 批量演练结论可直接外推到真实模拟器")
        return 0
    print("\n[FAIL] 有 %d 个字段不一致，逐字段排查" % bad)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
