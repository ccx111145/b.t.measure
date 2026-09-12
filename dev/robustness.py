# -*- coding: utf-8 -*-
"""鲁棒性压力测试：把论文 §7 表格里的数字实测出来"""
import io
import json
import math
import statistics as st
import sys

sys.path.insert(0, ".")

from robot.policy import Q3Config, run_episode
from sim.arena import generate_scenario


def batch(seeds, cfg, ratio=0.0, engine_kw=None, scen_kw=None):
    res = []
    for s in seeds:
        sc = generate_scenario(seed=s, directional_ratio=ratio, **(scen_kw or {}))
        r, _ = run_episode(sc, "T", cfg, engine_kwargs=engine_kw)
        res.append(r)
    ok = [r for r in res if r.n_sources == r.n_cleared and not r.unresolved]
    t = [r.avg_locate_clear_time_s for r in res if r.n_cleared]
    return {"n": len(res), "full": len(ok) / len(res),
            "mean": st.fmean(t) if t else float("nan"),
            "std": st.pstdev(t) if len(t) > 1 else 0.0,
            "max": max(t) if t else float("nan")}


SEEDS = list(range(200, 220))
print("=" * 78)
print("A. 示向度误差量化尺度 q（问题3，20 局）")
cfg3 = Q3Config()
base = None
for q in (0.1, 1.0, 10.0):
    r = batch(SEEDS, cfg3, engine_kw={"svd_quant": q})
    if base is None:
        base = r["mean"]
    print("  q=%-5s 满清除率=%.3f  平均=%.1f±%.1f s  相对 q=0.1 变化 %+.1f%%"
          % (q, r["full"], r["mean"], r["std"], 100 * (r["mean"] / base - 1)))

print("=" * 78)
print("B. 源间最小间距（问题3，20 局）")
for ms in (0.0, 50.0):
    r = batch(SEEDS, cfg3, scen_kw={"min_sep": ms})
    print("  min_sep=%-5s 满清除率=%.3f  平均=%.1f±%.1f s" % (ms, r["full"], r["mean"], r["std"]))

print("=" * 78)
print("C. 源数极端情形（问题3，20 局）")
for k in (10, 16):
    r = batch(SEEDS, cfg3, scen_kw={"n_sources": k})
    print("  k=%-3d 满清除率=%.3f  平均=%.1f±%.1f s" % (k, r["full"], r["mean"], r["std"]))

print("=" * 78)
print("D. 接收半径全部取下界 1000 m（最坏情形，问题3，20 局）")
rng_seeds = SEEDS
rows = []
for s in rng_seeds:
    sc = generate_scenario(seed=s)
    for src in sc.sources:
        src.recv_radius = 1000.0
    r, _ = run_episode(sc, "T", cfg3)
    rows.append(r)
t = [r.avg_locate_clear_time_s for r in rows if r.n_cleared]
full = sum(1 for r in rows if r.n_sources == r.n_cleared and not r.unresolved) / len(rows)
print("  R≡1000 m  满清除率=%.3f  平均=%.1f±%.1f s" % (full, st.fmean(t), st.pstdev(t)))
r_norm = batch(SEEDS, cfg3)
print("  对照(均匀 R) 满清除率=%.3f 平均=%.1f s  最坏情形上升 %+.1f%%"
      % (r_norm["full"], r_norm["mean"], 100 * (st.fmean(t) / r_norm["mean"] - 1)))

print("=" * 78)
print("E. 问题4 的 q 灵敏度（20 局，全定向）")
cfg4 = Q3Config(census_layout="hex_ring", directional=True)
b4 = None
for q in (0.1, 1.0, 10.0):
    r = batch(SEEDS, cfg4, ratio=1.0, engine_kw={"svd_quant": q})
    if b4 is None:
        b4 = r["mean"]
    print("  q=%-5s 满清除率=%.3f  平均=%.1f±%.1f s  相对变化 %+.1f%%"
          % (q, r["full"], r["mean"], r["std"], 100 * (r["mean"] / b4 - 1)))
