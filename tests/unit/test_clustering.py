# tests/unit/test_clustering.py
# detector/clustering.py 的自動化單元測試——純幾何函式，合成座標點即可測試。

import math

from detector.centroid import BallPosition
from detector.clustering import dist, make_xy_clusters


def _pos(x: float, y: float) -> BallPosition:
    return BallPosition(x=x, y=y, frame_index=0, confidence=1.0)


def test_dist_basic():
    assert dist(0, 0, 3, 4) == 5.0
    assert dist(1, 1, 1, 1) == 0.0


def test_make_xy_clusters_empty_list():
    assert make_xy_clusters([], radius=10.0) == []


def test_make_xy_clusters_single_point():
    clusters = make_xy_clusters([_pos(5, 5)], radius=10.0)
    assert len(clusters) == 1
    assert len(clusters[0]) == 1


def test_make_xy_clusters_two_points_within_radius_merge():
    pts = [_pos(0, 0), _pos(3, 4)]  # 距離 5
    clusters = make_xy_clusters(pts, radius=10.0)
    assert len(clusters) == 1
    assert len(clusters[0]) == 2


def test_make_xy_clusters_two_points_outside_radius_split():
    pts = [_pos(0, 0), _pos(100, 100)]
    clusters = make_xy_clusters(pts, radius=10.0)
    assert len(clusters) == 2


def test_make_xy_clusters_multiple_distinct_groups():
    # 兩個緊密群，彼此相距很遠
    pts = [
        _pos(0, 0), _pos(1, 1), _pos(2, 0),
        _pos(500, 500), _pos(501, 501),
    ]
    clusters = make_xy_clusters(pts, radius=10.0)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2, 3]


def test_make_xy_clusters_centroid_recomputed_as_group_grows():
    """質心是動態平均，不是固定第一點——第三點跟第一點的距離已經超過
    半徑（12 > 10），但跟「目前質心」(3,0) 的距離仍在半徑內（9 < 10），
    所以還是會被合併進同一群。"""
    pts = [_pos(0, 0), _pos(6, 0), _pos(12, 0)]
    assert dist(0, 0, 12, 0) > 10.0  # 前提：跟第一點比較的話不會合併

    clusters = make_xy_clusters(pts, radius=10.0)

    assert len(clusters) == 1
    assert len(clusters[0]) == 3
