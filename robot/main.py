# -*- coding: utf-8 -*-
"""
机器狗主程序（联调 / 演练 / 正式测试统一入口）

特点
----
* 只通过 HTTP+JSON 与模拟器通信（附件2 的 4 条指令），与真实模拟器**零改动切换**；
* 启动时**轮询 `/enter`** 直到接口就绪（真实模拟器要先点"开始测试"并等 5 s 倒计时）；
* 严格串行、每个新动作新 `request_id`、重试复用原 id（由 `HttpRobotClient` 保证）；
* 自行记录指令序列与响应（附件2 §12 要求），并输出题目表 1 所需的一行结果。

用法
----
    # 1) 先确认连通性（不进入目标区域）
    python robot/main.py --probe

    # 2) 在模拟器里点"问题3 演练测试"，然后运行
    python robot/main.py --problem 3 --tag q3_rehearsal_01

    # 3) 问题4 正式测试（把案例编码填进去，便于与加密日志对应）
    python robot/main.py --problem 4 --tag q4_official_1 --case-code XXXX-XXXX
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot.client import HttpRobotClient                       # noqa: E402
from robot.policy import Q3Config, Q3Policy                    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(path: Optional[str]) -> dict:
    p = path or os.path.join(ROOT, "config.json")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def make_cfg(problem: int, cfg_json: dict, over: Optional[dict] = None) -> Q3Config:
    pol = cfg_json.get("policy", {})
    kw = dict(
        grid_cell=float(pol.get("grid_cell_m", 10.0)),
        homing_step_max=float(pol.get("homing_step_max_m", 400.0)),
        clear_radius=float(pol.get("clear_radius_m", 20.0)),
    )
    if problem == 4:
        kw.update(census_layout="hex_ring", directional=True)
    else:
        kw.update(census_layout="ring",
                  census_radius=float(pol.get("census_radius_m", 1200.0)),
                  census_ring_n=int(pol.get("census_ring_points", 6)))
    if over:
        kw.update(over)
    return Q3Config(**kw)


def wait_for_interface(cl, wait_s: float, poll_s: float = 0.5, verbose: bool = True):
    """轮询 `/enter` 直到接口就绪

    真实模拟器在倒计时结束前接口未开放，连接会直接失败；测试窗口 25 min、
    程序运行时间上限 20 min，故必须在窗口开启后尽快 enter。

    健壮性：``/enter`` 返回 ``accepted=false`` 时无法区分"还没开放"与
    "上一次 /enter 其实成功了、只是响应丢了"。此时补发一条无害的 ``/measure``：
    若它被接受，说明**已经在区域内**，按已进入处理。
    """
    t0 = time.monotonic()
    last = None
    while time.monotonic() - t0 < wait_s:
        r = cl.enter()
        if r.accepted:
            if verbose:
                print("[enter] 成功：remaining_real_duration_s = %s（等待 %.1f s）"
                      % (r.body.get("remaining_real_duration_s"), time.monotonic() - t0))
            return r
        last = r
        if r.transport_ok and r.body.get("accepted") is False:
            probe = cl.measure(0.0, 0.0, 1)
            if probe.accepted:
                if verbose:
                    print("[enter] /enter 返回 false 但 /measure 被接受 → 判定为已进入"
                          "（上一次 enter 的响应丢失）")
                return probe
            if verbose:
                print("[enter] 接口已通但暂未接受（可能在倒计时/未开放），继续等待…")
        time.sleep(poll_s)
    raise TimeoutError("等待 %.0f s 仍未成功 /enter；最近一次响应：%r" % (wait_s, last))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="B题 机器狗主程序")
    ap.add_argument("--config", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--robot-id", default=None)
    ap.add_argument("--problem", type=int, default=3, choices=[3, 4])
    ap.add_argument("--tag", default=None, help="本次运行的标签（用于文件命名）")
    ap.add_argument("--case-code", default=None, help="模拟器显示的测试案例编码（表 1 需要）")
    ap.add_argument("--wait-s", type=float, default=420.0, help="等待接口就绪的最长时间")
    ap.add_argument("--probe", action="store_true", help="只做连通性探测，不进入目标区域")
    ap.add_argument("--no-wait", action="store_true", help="不轮询 /enter，直接尝试一次")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--allow-placeholder-id", action="store_true",
                    help="仅本地联调：允许用占位符 robot_id 运行（接真实模拟器时禁用）")
    args = ap.parse_args(argv)

    cfg_json = load_config(args.config)
    base_url = args.base_url or cfg_json.get("base_url", "http://127.0.0.1:2026")
    robot_id = args.robot_id or cfg_json.get("robot_id", "TEAM-2026-B-001")
    tag = args.tag or ("probe" if args.probe else "run")
    out_dir = args.out_dir or os.path.join(ROOT, cfg_json.get("outputs", {}).get("log_dir", "runs"))
    os.makedirs(out_dir, exist_ok=True)

    # 安全闸：robot_id 必须是真实参赛队号，否则正式测试必然全部失败（且会浪费测试机会）
    PLACEHOLDERS = ("<参赛队号>", "TEAM-2026-B-001", "TEAM-TEST", "TEAM-LOCAL",
                    "TEAM-CLI", "TEAM-E2E", "TEAM-EQ", "TEAM-TRACE", "")
    if robot_id in PLACEHOLDERS and not args.allow_placeholder_id:
        print("!" * 74)
        print("拒绝运行：robot_id 还是占位符 %r" % robot_id)
        print("模拟器要求 robot_id **逐字节等于当前登录的参赛队号**，否则所有动作")
        print("都会返回 accepted=false —— 而正式测试机会一旦开始就被占用。")
        print("请把 config.json 的 robot_id 改成竞赛报名系统的参赛队号（或用 --robot-id）。")
        print("（仅本地联调可用 --allow-placeholder-id 跳过本检查）")
        print("!" * 74)
        return 3

    to = cfg_json.get("timeouts", {})
    cl = HttpRobotClient(base_url, robot_id,
                         timeout_s=float(to.get("http_timeout_s", 5.0)),
                         max_retries=int(to.get("max_retries", 3)),
                         retry_wait_s=float(to.get("retry_wait_s", 0.05)))

    print("=" * 74)
    print("机器狗主程序  base_url=%s  robot_id=%s  问题%d  tag=%s"
          % (base_url, robot_id, args.problem, tag))
    print("=" * 74)

    if args.probe:
        # 注意：**不能**用 /enter 探测——它会真正进入目标区域并占用本次测试机会。
        # 进入之前 /measure 必然返回 accepted=false，既能证明接口连通，又无副作用。
        r = cl.measure(0.0, 0.0, 1)
        print("探测结果：http=%s transport_ok=%s body=%s"
              % (r.http_status, r.transport_ok,
                 json.dumps(r.body, ensure_ascii=False)))
        if not r.transport_ok:
            print("→ 接口未开放或未连通（正常现象：请先在模拟器里点开始测试）")
        elif r.body.get("accepted") is False:
            print("→ 接口已开放且连通；/measure 在 /enter 之前被拒（符合预期，未消耗测试机会）")
            print("→ 可以点模拟器的开始测试，然后运行 robot/main.py --problem 3")
        else:
            print("→ 意外：进入前的 /measure 竟被接受，请检查模拟器状态")
        return 0

    cfg = make_cfg(args.problem, cfg_json)
    print("策略配置：%s" % json.dumps(cfg.__dict__, ensure_ascii=False, default=str))

    entered = False
    if args.no_wait:
        r = cl.enter()
        if not r.accepted:
            print("!/enter 未成功：", json.dumps(r.body, ensure_ascii=False))
            return 2
        entered = True
    else:
        wait_for_interface(cl, args.wait_s, verbose=not args.quiet)
        entered = True

    pol = Q3Policy(cl, cfg, verbose=not args.quiet)
    t0 = time.monotonic()
    res = pol.run(already_entered=entered)
    wall = time.monotonic() - t0

    log_path = os.path.join(out_dir, "%s_log.json" % tag)
    cl.dump_log(log_path)
    row = {
        "tag": tag,
        "case_code": args.case_code,
        "problem": args.problem,
        "robot_id": robot_id,
        "base_url": base_url,
        "清除干扰源个数": res.n_cleared,
        "干扰源总数(本机真值, 仅本地模拟器可得)": res.n_sources,
        "清除比例": res.clear_ratio,
        "平均定位清除时间_s": res.avg_locate_clear_time_s,
        "定位清除总时间_s": res.virtual_time_s,
        "程序运行时间_s": round(wall, 3),
        "动作数": res.n_actions,
        "请求数": res.n_requests,
        "未结案频道": res.unresolved,
        "证明无源频道数": len(res.proven_empty),
        "结束原因": res.ended_reason,
        "action_log": log_path,
    }
    rep_path = os.path.join(out_dir, "%s_report.json" % tag)
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(row, f, ensure_ascii=False, indent=1)

    print("-" * 74)
    print("清除干扰源个数        : %d" % res.n_cleared)
    print("平均定位清除时间      : %.1f s" % res.avg_locate_clear_time_s)
    print("定位清除总时间        : %.1f s" % res.virtual_time_s)
    print("程序运行时间          : %.1f s（本机真实耗时 %.1f s）"
          % (res.program_run_time_s, wall))
    print("动作数 / 请求数       : %d / %d" % (res.n_actions, res.n_requests))
    print("未结案频道            : %s" % (res.unresolved or "无"))
    print("结束原因              : %s" % res.ended_reason)
    print("动作日志              : %s" % log_path)
    print("表 1 记录             : %s" % rep_path)
    print("-" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
