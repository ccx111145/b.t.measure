# -*- coding: utf-8 -*-
"""
问题4（含定向源）策略测试

 Q1 检测网保证：六角点阵 step ≤ R_min 且覆盖到区域外时，**任意位置、任意朝向**的定向源都可被检出
    （判据 $G\\in\\mathrm{conv}(S_G)$，$S_G$ = 网中到 G 距离 ≤ 1000 m 的点）
 Q2 对照：问题3 的"中心+6环 ρ=1200"网**不满足**该保证 → 定量解释为什么问题4 必须加密
 Q3 外圈的必要性：把点阵裁剪到目标圆内会破坏保证（边界朝外的源不可检出）
 Q4 栅格扫描覆盖性：格距 28.28 m 时可行域内任一点到最近格点 ≤ 20 m
 Q5 多局一致性：全定向 30 局必须 100% 清除；有源频道绝不被判"证明无源"；清除点必落在真值 20 m 内
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot import geometry as G                            # noqa: E402
from robot.feasible import FeasibleTracker                 # noqa: E402
from robot.policy import Q3Config, Q3Policy, run_episode    # noqa: E402
from robot.client import LocalRobotClient                  # noqa: E402
from sim.arena import generate_scenario                    # noqa: E402
from sim.engine import ArenaEngine                         # noqa: E402

FAIL = []
R_LO = 1000.0
ARENA_R = 1800.0


def check(name, cond, info=""):
    print("  [%s] %s %s" % ("PASS" if cond else "FAIL", name, info))
    if not cond:
        FAIL.append(name)


def _in_hull(p, pts, tol=1e-6):
    if len(pts) == 0:
        return False
    d = pts - p
    r = np.hypot(d[:, 0], d[:, 1])
    if (r <= tol).any():
        return True
    a = np.sort(np.arctan2(d[:, 1], d[:, 0]))
    gaps = np.diff(np.concatenate([a, [a[0] + 2 * math.pi]]))
    return bool(gaps.max() <= math.pi + 1e-9)


def detectable_ratio(P, n=6000, seed=11, extent=ARENA_R):
    rng = np.random.default_rng(seed)
    r = extent * np.sqrt(rng.random(n))
    a = rng.random(n) * 2 * math.pi
    Gp = np.stack([r * np.cos(a), r * np.sin(a)], 1)
    bad = 0
    for g in Gp:
        d = np.hypot(P[:, 0] - g[0], P[:, 1] - g[1])
        if not _in_hull(g, P[d <= R_LO + 1e-9]):
            bad += 1
    return 1.0 - bad / n


def hex_pts(step, extent):
    pts = []
    dy = step * math.sqrt(3) / 2
    ny = int(2 * extent / dy) + 2
    nx = int(2 * extent / step) + 2
    for j in range(-ny, ny + 1):
        y = j * dy
        off = (step / 2) if (j % 2) else 0.0
        for i in range(-nx, nx + 1):
            x = i * step + off
            if math.hypot(x, y) <= extent + 1e-9:
                pts.append((x, y))
    return np.array(pts)


print("=" * 78)
print("Q1 六角点阵的定向源检测保证")
for step, extent, want in ((1000.0, 2800.0, True), (1100.0, 2800.0, False)):
    P = hex_pts(step, extent)
    ratio = detectable_ratio(P)
    check("step=%.0f extent=%.0f（%d 点）可检出比例 = %.4f"
          % (step, extent, len(P), ratio),
          (ratio >= 0.9999) == want, "符合预期")

print("=" * 78)
print("Q2 对照：问题3 的环状巡测网不满足定向源保证")
ring = np.array([[0.0, 0.0]] + [(1200 * math.cos(2 * math.pi * k / 6),
                                 1200 * math.sin(2 * math.pi * k / 6)) for k in range(6)])
ratio = detectable_ratio(ring)
check("中心+6环 ρ=1200 可检出比例 = %.4f（远低于 1）" % ratio, ratio < 0.5,
      "定量说明问题4 必须加密巡测网")

print("=" * 78)
print("Q3 外圈的必要性")
P_in = hex_pts(1000.0, 1800.0)
r_in = detectable_ratio(P_in)
P_out = hex_pts(1000.0, 2800.0)
r_out = detectable_ratio(P_out)
check("裁剪到区域内的六角点阵可检出比例 = %.4f（< 1）" % r_in, r_in < 0.9999,
      "%d 点" % len(P_in))
check("扩展到区域外后可检出比例 = %.4f" % r_out, r_out >= 0.9999)

print("=" * 78)
print("Q4 栅格扫描覆盖性（格距 28.28 m）")
r = G.get_raster(10.0)
rng = np.random.default_rng(5)
worst = 0.0
for _ in range(40):
    cx, cy = rng.uniform(-1200, 1200, 2)
    half = float(rng.uniform(30, 200))
    m = ((r.x - cx) ** 2 + (r.y - cy) ** 2) <= half * half
    tr = FeasibleTracker(cell=10.0)
    tr.st[11].mask = m
    pts = tr.probe_points(11, spacing=20.0 * math.sqrt(2.0), limit=4000)
    if not pts:
        continue
    P = np.array(pts)
    idx = np.flatnonzero(m)
    X = np.stack([r.x[idx], r.y[idx]], 1)
    d = np.min(np.hypot(X[:, None, 0] - P[None, :, 0],
                        X[:, None, 1] - P[None, :, 1]), axis=1)
    worst = max(worst, float(d.max()))
check("可行域内任一点到最近扫描格点 ≤ 20 m", worst <= 20.0 + 1e-6,
      "实测最大 %.2f m" % worst)

print("=" * 78)
print("Q5 全定向 30 局一致性")
cfg = Q3Config(census_layout="hex_ring", directional=True)
bad = {"not_full": 0, "unresolved": 0, "false_empty": 0, "far_clear": 0}
for seed in range(1, 31):
    sc = generate_scenario(seed=seed, directional_ratio=1.0)
    eng = ArenaEngine(sc, "T")
    eng.arm()
    cl = LocalRobotClient(eng, "T")
    pol = Q3Policy(cl, cfg)
    res = pol.run()
    if res.n_cleared != res.n_sources:
        bad["not_full"] += 1
    if res.unresolved:
        bad["unresolved"] += 1
    src_ch = set(s.channel for s in sc.sources)
    for ch in res.proven_empty:
        if ch in src_ch:
            bad["false_empty"] += 1
    for rec in cl.log:
        resp = rec.get("response") or {}
        if rec.get("path") != "/clear" or not resp.get("accepted"):
            continue
        if resp.get("clear_result") != "success":
            continue
        pos = rec["request"]["position"]
        s = sc.by_channel(rec["request"]["channel"])
        if s is None or math.dist((pos["x"], pos["y"]), (s.x, s.y)) > 20.0 + 1e-6:
            bad["far_clear"] += 1
for k, v in bad.items():
    check("异常计数 %s" % k, v == 0, "= %d" % v)


print("=" * 78)
print("Q6 新巡测网 hex_ring（25 点）的检测保证与成本")
from robot.policy import Q3Policy as _P                        # noqa: E402
for layout in ("hex", "hex_ring"):
    p = _P.__new__(_P)
    p.cfg = Q3Config(census_layout=layout, directional=True)
    wp = p.census_waypoints()
    L = sum(math.dist(wp[i - 1], wp[i]) for i in range(1, len(wp)))
    P = np.array(wp)
    r = detectable_ratio(P, n=4000)
    extra = ""
    if layout == "hex_ring":
        globals()["_WP_RING"] = wp
    check("%s：%d 点，路线 %.0f m（%.0f s），可检出比例 %.4f"
          % (layout, len(wp), L, L / 5, r), r >= 0.9999, extra)
    print("        点数 %d，路线 %.0f m" % (len(wp), L))

_wp_hex = _P.__new__(_P)
_wp_hex.cfg = Q3Config(census_layout="hex", directional=True)
_wh = _wp_hex.census_waypoints()
_Lh = sum(math.dist(_wh[i - 1], _wh[i]) for i in range(1, len(_wh)))
_Lr = sum(math.dist(_WP_RING[i - 1], _WP_RING[i]) for i in range(1, len(_WP_RING)))
check("hex_ring 比 hex 更省", _Lr < _Lh,
      "%.0f m vs %.0f m（省 %.0f s）" % (_Lr, _Lh, (_Lh - _Lr) / 5))

print("=" * 78)
print("Q7 回归：清除扫描与探测点的访问记录必须分离")
# 若把 /measure 过点也算作"/clear 过点"，28.28 m 扫描的覆盖保证就会失效
import inspect                                              # noqa: E402
src = inspect.getsource(_P._clear_sweep)
check("_clear_sweep 只按 clear_visited 过滤", "clear_visited" in src
      and "probe_visited" not in src, "")
src2 = inspect.getsource(_P._clear)
check("_clear 同时登记 clear_visited", "clear_visited" in src2, "")

print("=" * 78)
if FAIL:
    print("失败项:", FAIL)
    sys.exit(1)
print("全部测试通过")
