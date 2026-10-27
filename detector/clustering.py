# detector/clustering.py
# 共用的 XY 密度聚類工具，供 CloudBallTracker、BatCloudTracker 共用。
#
# 2026-07-27 從 detector/ball_tracker_cloud.py、detector/bat_tracker_cloud.py
# 兩邊逐字重複的 _dist/_make_xy_clusters 抽出（見架構審查 Candidate B）。

import math


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def make_xy_clusters(pts: list, radius: float) -> list:
    """貪心 XY 聚類：距現有質心 < radius 合入，否則新建群。"""
    clusters: list[list] = []
    for p in pts:
        placed = False
        for c in clusters:
            cx = sum(q.x for q in c) / len(c)
            cy = sum(q.y for q in c) / len(c)
            if dist(p.x, p.y, cx, cy) < radius:
                c.append(p)
                placed = True
                break
        if not placed:
            clusters.append([p])
    return clusters
