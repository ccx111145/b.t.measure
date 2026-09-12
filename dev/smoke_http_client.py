# -*- coding: utf-8 -*-
"""
HTTP 端到端冒烟测试客户端：对着一个已启动的本地模拟器（或真实模拟器）跑一遍最小动作序列

用法::
    python sim/run_server.py --seed 42 --port 2026 &      # 另开一个终端
    python dev/smoke_http_client.py --base-url http://127.0.0.1:2026 --robot-id TEAM-TEST
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.client import HttpRobotClient      # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:2026")
    ap.add_argument("--robot-id", default="TEAM-TEST")
    ap.add_argument("--out", default=None, help="把动作日志写到该 json 文件")
    args = ap.parse_args(argv)

    cl = HttpRobotClient(args.base_url, args.robot_id)
    r = cl.enter()
    print("enter:", r)
    if not r.accepted:
        print("进入失败，检查模拟器是否已开放接口、robot_id 是否与登录队号一致")
        return 1
    print("remaining_real_duration_s =", r.body.get("remaining_real_duration_s"))

    # 附录2 第10节的示例动作（用真实场景跑，结果值不重要，只验证链路与计时）
    seq = [("measure", cl.measure, (300, 400, 1)),
           ("measure", cl.measure, (300, 400, 2)),
           ("clear", cl.clear, (300, 0, 3)),
           ("measure", cl.measure, (300, 0, 2))]
    for kind, fn, a in seq:
        res = fn(*a)
        print("%-8s %-16s http=%s accepted=%s vt=%.1f %s"
              % (kind, str(a), res.http_status, res.body.get("accepted"),
                 res.virtual_time_s,
                 res.measure_result or res.clear_result or ""))
        if not res.accepted:
            print("  动作未被接受:", json.dumps(res.body, ensure_ascii=False))
            break
    e = cl.exit()
    print("exit:", e)
    print("总请求数:", cl.n_requests)
    if args.out:
        cl.dump_log(args.out)
        print("日志已写入", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
