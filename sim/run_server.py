# -*- coding: utf-8 -*-
"""
启动本地模拟器的 HTTP 服务（默认 127.0.0.1:2026，与真实模拟器一致）

用法::

    python sim/run_server.py --seed 42 --robot-id TEAM-2026-B-001
    python sim/run_server.py --directional-ratio 0.5 --port 2027 --n-sources 13
    python sim/run_server.py --list-seeds 11 22 33        # 打印场景真值摘要

机器狗程序只需把 BASE_URL / ROBOT_ID 指向本服务即可（与真实模拟器零改动切换）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sim.arena import generate_scenario                       # noqa: E402
from sim.engine import ArenaEngine                            # noqa: E402
from sim.server import make_server                            # noqa: E402


def build_engine(args) -> ArenaEngine:
    sc = generate_scenario(seed=args.seed, n_sources=args.n_sources,
                           directional_ratio=args.directional_ratio,
                           min_sep=args.min_sep)
    eng = ArenaEngine(sc, args.robot_id, max_virtual=args.max_virtual,
                      max_real=args.max_real, window_s=args.window_s,
                      svd_quant=args.svd_quant)
    return eng


def main(argv=None):
    ap = argparse.ArgumentParser(description="B题 本地无线电干扰源环境模拟器")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=2026)
    ap.add_argument("--robot-id", default="TEAM-2026-B-001",
                    help="参赛队号（机器狗必须使用同一个 robot_id）")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--n-sources", type=int, default=None,
                    help="干扰源个数，默认在 10..16 随机")
    ap.add_argument("--directional-ratio", type=float, default=0.0,
                    help="定向源占比：问题3 用 0，问题4 用 0.5 之类")
    ap.add_argument("--min-sep", type=float, default=0.0, help="源之间最小间距（m）")
    ap.add_argument("--svd-quant", type=float, default=1.0,
                    help="示向度误差的'同一地点'量化尺度（m）")
    ap.add_argument("--max-virtual", type=float, default=360000.0)
    ap.add_argument("--max-real", type=float, default=1200.0)
    ap.add_argument("--window-s", type=float, default=1500.0, help="25 分钟测试窗口")
    ap.add_argument("--list-seeds", type=int, nargs="*", default=None,
                    help="只打印这些种子的场景真值摘要后退出")
    args = ap.parse_args(argv)

    if args.list_seeds is not None:
        seeds = args.list_seeds or [None]
        for s in seeds:
            sc = generate_scenario(seed=s, n_sources=args.n_sources,
                                   directional_ratio=args.directional_ratio)
            print(json.dumps(sc.summary(), ensure_ascii=False))
        return 0

    eng = build_engine(args)
    srv = make_server(eng, args.host, args.port)
    print("=" * 70)
    print("本地模拟器已启动: http://%s:%d" % (args.host, srv.server_address[1]))
    print("robot_id       :", args.robot_id)
    print("场景摘要(真值) :", json.dumps(eng.scenario.summary(), ensure_ascii=False))
    print("干扰源真值     :", json.dumps(eng.scenario.truth_table(), ensure_ascii=False))
    print("窗口 %g s，实时上限 %g s，虚拟上限 %g s" % (args.window_s, args.max_real,
                                                      args.max_virtual))
    print("按 Ctrl+C 结束；结束后打印本局报告")
    print("=" * 70)
    eng.arm()
    stop = threading.Event()
    th = threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.05},
                          daemon=True)
    th.start()
    try:
        while th.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        srv.shutdown()
        srv.server_close()
        print(json.dumps(eng.report(), ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
