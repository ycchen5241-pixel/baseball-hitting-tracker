# tests/unit/test_pipeline.py
# physics/pipeline.py::compute_ball_physics() 的自動化單元測試。
#
# 純合成資料，不需要影片檔案或 YOLO 模型——這正是把 Ball Physics
# 抽成一個深模組後才第一次能做到的事（見 CONTEXT.md「Ball Physics」）。

import pytest

from detector.ball_tracker_cloud import CloudTrackResult
from detector.centroid import BallPosition
from physics.pipeline import BallPhysicsResult, compute_ball_physics


def _pos(frame_index: int, x: float, y: float, confidence: float = 0.9) -> BallPosition:
    return BallPosition(x=x, y=y, frame_index=frame_index, confidence=confidence)


def _cloud_result(positions, contact_frame) -> CloudTrackResult:
    return CloudTrackResult(
        positions=positions,
        static_blacklist=[],
        tstand_cluster=None,
        contact_frame=contact_frame,
        scenario="tstand",
    )


def test_uses_cloud_contact_frame_when_available():
    """CloudBallTracker 內建判定不是 None 時，直接採用，不需要備援。"""
    positions = [_pos(10, 100.0, 50.0), _pos(11, 110.0, 50.5), _pos(12, 120.0, 51.0)]
    ball_result = _cloud_result(positions, contact_frame=10)

    result = compute_ball_physics(ball_result, {}, [], None, fps=240.0)

    assert result.contact == 10
    assert [p.frame_index for p in result.velocity_post] == [10, 11, 12]


def test_falls_back_to_find_contact_frame_when_cloud_contact_missing(monkeypatch):
    """contact_frame 為 None 時退回 find_contact_frame() 備援。"""
    positions = [_pos(5, 0.0, 0.0), _pos(6, 10.0, 0.0)]
    ball_result = _cloud_result(positions, contact_frame=None)

    monkeypatch.setattr(
        "physics.pipeline.find_contact_frame", lambda *a, **kw: 6,
    )

    result = compute_ball_physics(ball_result, {}, [], None, fps=240.0)

    assert result.contact == 6
    assert [p.frame_index for p in result.velocity_post] == [6]


def test_forwards_barrel_positions_to_find_contact_frame(monkeypatch):
    """迴歸測試：確保 barrel_positions 真的被傳進 find_contact_frame() 的
    備援呼叫——tests/test_cloud_tracker.py 曾經悄悄漏傳這個參數，備援的
    Path A（距離最小值）訊號因此從未真正生效過。共用 compute_ball_physics()
    後，三個呼叫端都不可能再犯這個錯誤。
    """
    positions = [_pos(1, 0.0, 0.0)]
    ball_result = _cloud_result(positions, contact_frame=None)
    sentinel_barrel_positions = ["sentinel"]

    captured = {}

    def fake_find_contact_frame(ball_positions, bat_head_dets, ball_dets, **kwargs):
        captured.update(kwargs)
        return None

    monkeypatch.setattr(
        "physics.pipeline.find_contact_frame", fake_find_contact_frame,
    )

    compute_ball_physics(
        ball_result, {}, sentinel_barrel_positions, None, fps=240.0,
    )

    assert captured.get("barrel_positions") is sentinel_barrel_positions


def test_returns_empty_result_when_no_contact_found(monkeypatch):
    """兩種來源都判定不出接觸幀時，回傳全空的 BallPhysicsResult。"""
    ball_result = _cloud_result([_pos(1, 0.0, 0.0)], contact_frame=None)
    monkeypatch.setattr(
        "physics.pipeline.find_contact_frame", lambda *a, **kw: None,
    )

    result = compute_ball_physics(ball_result, {}, [], None, fps=240.0)

    assert result == BallPhysicsResult()


def test_computes_launch_angle_and_exit_velocity_for_valid_window():
    """接觸後有一段乾淨、接近水平的軌跡時，LA 跟 EV 都應該算出數值。"""
    # 水平向左移動、y 幾乎不變的合成軌跡，模擬側拍下接近打平的擊球
    positions = [
        _pos(100, 1000.0, 500.0),
        _pos(101, 970.0, 500.5),
        _pos(102, 940.0, 501.0),
        _pos(103, 910.0, 501.2),
        _pos(104, 880.0, 501.5),
    ]
    ball_result = _cloud_result(positions, contact_frame=100)
    hp_bbox = (100.0, 0.0, 300.0, 50.0)  # 寬 200px

    result = compute_ball_physics(ball_result, {}, [], hp_bbox, fps=240.0)

    assert result.contact == 100
    assert result.launch_angle is not None
    assert abs(result.launch_angle) < 5.0  # 幾乎打平
    assert result.exit_velocity is not None
    assert result.exit_velocity > 0


def test_exit_velocity_none_without_hp_bbox():
    """沒有鎖定本壘板 bbox 時，Exit Velocity 略過不計算，但 LA 仍可計算。"""
    positions = [
        _pos(100, 1000.0, 500.0), _pos(101, 970.0, 500.0),
        _pos(102, 940.0, 500.0), _pos(103, 910.0, 500.0),
    ]
    ball_result = _cloud_result(positions, contact_frame=100)

    result = compute_ball_physics(ball_result, {}, [], None, fps=240.0)

    assert result.exit_velocity is None
    assert result.launch_angle is not None
