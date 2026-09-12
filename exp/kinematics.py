# -*- coding: utf-8 -*-
"""运动学可行性验证（修正版）：先剔除落在障碍内的网络点，再用可见图算绕障行程"""

from __future__ import annotations

import heapq
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "exp"))

from nonconvex import ARENA_R, Rect, los          # noqa: E402
from nonconvex_run import OBSTACLES, two_layer    # noqa: E402

SAFE = 40.0


def inflate(r, m):
    return Rect(r.x0 - m, r.y0 - m, r.x1 + m, r.y1 + m)


def corners(obs):
    out = []
    for r in obs:
        out += [(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)]
    return out


def build_graph(nodes, obs):
    n = len(nodes)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if los(nodes[i], nodes[j], obs):
                d = math.dist(nodes[i], nodes[j])
                adj[i].append((j, d)); adj[j].append((i, d))
    return adj


def dijkstra(adj, src):
    dist = [float("inf")] * len(adj); dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w; heapq.heappush(pq, (d + w, v))
    return dist


def report(name, P):
    print("=" * 84)
    print(name)
    print("=" * 84)
    inf_obs = [tuple(p) for p in P if any(r.contains((p[0], p[1])) for r in OBSTACLES)]
    if inf_obs:
        print("  ** %d 个网络点落在障碍内部，物理不可达，需重定位：" % len(inf_obs))
        for q in inf_obs[:6]:
            print("       (%.0f, %.0f)" % q)
    free = [(float(p[0]), float(p[1])) for p in P
            if not any(r.contains((p[0], p[1])) for r in OBSTACLES)]
    print("  可用（自由空间内）网络点：%d" % len(free))

    obs_inf = [inflate(r, SAFE) for r in OBSTACLES]
    nodes = free + corners(obs_inf)
    adj = build_graph(nodes, obs_inf)
    deg = [len(a) for a in adj]
    iso = sum(1 for d in deg if d == 0)
    print("  可见图：%d 节点 / %d 边 / 平均度 %.1f / 孤立点 %d"
          % (len(nodes), sum(deg) // 2, np.mean(deg), iso))

    start = int(np.argmin([math.hypot(p[0], p[1]) for p in free]))
    dist = dijkstra(adj, start)
    reach = [i for i in range(len(free)) if dist[i] < float("inf")]
    print("  从起点可见图可达的网络点：%d / %d" % (len(reach), len(free)))

    # 可行巡游（最近邻 + 最短路）
    unv = set(range(len(free))) - {start}
    cur, total, order = start, 0.0, [start]
    while unv:
        d = dijkstra(adj, cur)
        nxt = min(unv, key=lambda k: d[k])
        if d[nxt] == float("inf"):
            print("  [警告] 剩余 %d 个点不可达" % len(unv)); break
        total += d[nxt]; unv.discard(nxt); order.append(nxt); cur = nxt

    # 直线（不可行）对照
    cur = np.asarray(free[start]); left = set(range(len(free))) - {start}
    L_str, bad = 0.0, 0
    while left:
        j = min(left, key=lambda i: float(np.hypot(*(np.asarray(free[i]) - cur))))
        nxt = np.asarray(free[j])
        if not los(tuple(cur), tuple(nxt), OBSTACLES):
            bad += 1
        L_str += float(np.hypot(*(nxt - cur)))
        left.discard(j); cur = nxt
    print()
    print("  直线连接（直线位移假设）: %8.0f m，其中 **%d 段穿越障碍**" % (L_str, bad))
    print("  可见图绕障（运动学可行）: %8.0f m  → 移动 %.0f s" % (total, total / 5))
    if L_str > 0:
        print("  绕障代价：+%.0f m（+%.1f%%）" % (total - L_str, 100 * (total / L_str - 1)))
    print()
    return total, L_str


if __name__ == "__main__":
    report("圆盘最优 25 点网（未针对障碍调整）", two_layer())
    try:
        Pg = np.load("tab/nonconvex_net.npy")
        report("贪心沿阴影修复后的网络", Pg)
    except Exception as e:                                   # noqa: BLE001
        print("（未找到修复网：%s）" % e)
