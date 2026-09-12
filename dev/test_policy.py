# -*- coding: utf-8 -*-
"""
问题3 策略测试

 P1 一致性：多局演练必须 100% 清除，且无未结案频道
 P2 外近似的安全性：**任何真实存在干扰源的频道，绝不能被"证明无源"**
    （这是栅格/多边形外近似正确性的核心断言，一旦违反说明证据算子有 bug）
 P3 清除成功的判据是紧的：凡被判定"保证可清除"的点，实际 /clear 必成功
 P4 巡测网覆盖性：中心 + 6 环 ρ=1200 的最大未覆盖距离 ≤ 1000 m（R_min）
 P5 时间与动作预算：单局请求数远小于 max_actions，程序运行时间远小于 1200 s
 P6 与真实删除的对照：引擎里的真值源全部被清除，且清除点确实落在真值 20 m 内
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np                                        # noqa: E402

from robot import geometry as G                           # noqa: E402
from robot.policy import Q3Config, Q3Policy, run_episode   # noqa: E402
from robot.client import LocalRobotClient                 # noqa: E402
from sim.arena import generate_scenario                   # noqa: E402
from sim.engine import ArenaEngine                        # noqa: E402

FAIL = []


def check(name, cond, info=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, info))
    if not cond:
        FAIL.append(name)


print("=" * 78)
print("P4 巡测网覆盖性")
for rho, k, want_ok in ((1200.0, 6, True), (1100.0, 6, False), (1000.0, 5, False)):
    pts = [(0.0, 0.0)] + [(rho * math.cos(2 * math.pi * i / k),
                           rho * math.sin(2 * math.pi * i / k)) for i in range(k)]
    xs = np.arange(-1800, 1800.01, 10.0)
    XX, YY = np.meshgrid(xs, xs, indexing="ij")
    XX, YY = XX.ravel(), YY.ravel()
    keep = XX * XX + YY * YY <= 1800 ** 2
    XX, YY = XX[keep], YY[keep]
    d = np.full(len(XX), np.inf)
    for p in pts:
        d = np.minimum(d, np.hypot(XX - p[0], YY - p[1]))
    mx = float(d.max())
    ok = mx <= 1000.0
    check("中心+%d环 ρ=%.0f → 最大未覆盖 %.1f m" % (k, rho, mx), ok == want_ok,
          "符合预期" if ok == want_ok else "与预期不符")

print("=" * 78)
print("P1/P2/P3/P5/P6 多局演练一致性（20 局）")
cfg = Q3Config()
bad = {"not_full": 0, "unresolved": 0, "false_empty": 0, "budget": 0, "far_clear": 0}
worst_actions = 0
worst_real = 0.0
for seed in range(1, 21):
    sc = generate_scenario(seed=seed)
    eng = ArenaEngine(sc, "T")
    eng.arm()
    cl = LocalRobotClient(eng, "T")
    pol = Q3Policy(cl, cfg)
    res = pol.run()

    if res.n_cleared != res.n_sources:
        bad["not_full"] += 1
    if res.unresolved:
        bad["unresolved"] += 1
    # P2：真实有源的频道绝不能被判为"证明无源"
    src_ch = set(s.channel for s in sc.sources)
    for ch in res.proven_empty:
        if ch in src_ch:
            bad["false_empty"] += 1
    # P3/P6：每次成功的 /clear 都必须落在某个真值源 20 m 内
    for rec in cl.log:
        resp = rec.get("response") or {}
        if rec.get("path") != "/clear" or not resp.get("accepted"):
            continue
        if resp.get("clear_result") != "success":
            continue
        pos = rec["request"]["position"]
        ch = rec["request"]["channel"]
        s = sc.by_channel(ch)
        if s is None or math.dist((pos["x"], pos["y"]), (s.x, s.y)) > 20.0 + 1e-6:
            bad["far_clear"] += 1
    if res.n_requests > cfg.max_actions * 0.4:
        bad["budget"] += 1
    worst_actions = max(worst_actions, res.n_requests)
    worst_real = max(worst_real, res.real_elapsed_s)

for k, v in bad.items():
    check("异常计数 %s" % k, v == 0, "= %d" % v)
check("单局最大请求数 < 400", worst_actions < 400, "实际 %d" % worst_actions)
check("单局本机耗时 < 5 s", worst_real < 5.0, "实际 %.2f s" % worst_real)

print("=" * 78)
print("P2' 外近似算子安全性（直接仿真：真值所在单元永不被剔除）")
rng = np.random.default_rng(2026)
bad_ops = 0
r = G.get_raster(10.0)
for _ in range(300):
    P = tuple(rng.uniform(-1500, 1500, 2))
    d = float(rng.uniform(20, 1600))
    th = float(rng.uniform(0, 360))
    src = (P[0] + d * math.cos(th * G.D2R), P[1] + d * math.sin(th * G.D2R))
    if math.hypot(*src) > 1800:
        continue
    m = np.ones(r.n, dtype=bool)
    # no_signal：源在 1000 m 外
    if d > 1000 + 1e-9:
        m &= r.keep_outside_ball(P, 1000.0)
    # direction：源在 1500 m 内且在楔形内
    if d <= 1500 + 1e-9:
        m &= r.keep_wedge(P, th)
        m &= r.keep_ball(P, 1500.0)
    # near
    if d <= 5.0:
        m &= r.keep_ball(P, 5.0)
    # clear 失败：源在 20 m 外
    if d > 20.0:
        m &= r.keep_outside_ball(P, 20.0)
    k = int(np.argmin((r.x - src[0]) ** 2 + (r.y - src[1]) ** 2))
    if not m[k]:
        bad_ops += 1
check("300 组随机证据下真值单元未被剔除", bad_ops == 0, "误删 = %d" % bad_ops)


print("=" * 78)
print("P7 回归：源正好落在巡测点 5 m 内（/measure 返回 near）时不得被误判为无源")
from sim.arena import positional_scenario                        # noqa: E402
from robot.policy import Q3Policy as _P2                          # noqa: E402
from robot.client import LocalRobotClient as _C2                  # noqa: E402
from sim.engine import ArenaEngine as _E2                         # noqa: E402
# 把源放在第一个巡测点 (0,0) 旁边 3 m 处：/measure 必然返回 near
sc_near = positional_scenario([{"channel": 7, "x": 3.0, "y": 0.0,
                                "recv_radius": 1200.0, "kind": "omni"}])
eng_n = _E2(sc_near, "T"); eng_n.arm()
cl_n = _C2(eng_n, "T")
pol_n = _P2(cl_n, Q3Config())
res_n = pol_n.run()
check("near 源被清除（未误判为无源）",
      res_n.n_cleared == 1 and 7 in pol_n.tr.cleared_channels(),
      "清除 %d，未结案 %s，证否 %s" % (res_n.n_cleared, res_n.unresolved,
                                       res_n.proven_empty))
check("near 计入检测证据（n_detect 至少包含那次 near）",
      pol_n.tr.st[7].n_detect > 0,
      "n_detect=%d n_dir=%d" % (pol_n.tr.st[7].n_detect, pol_n.tr.st[7].n_dir))

print("=" * 78)
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
print("全部测试通过")
