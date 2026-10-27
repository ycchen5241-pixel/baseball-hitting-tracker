# physics/pipeline.py
# 球體物理計算的唯一入口：Contact Frame → Velocity Window → Launch Angle → Exit Velocity。
#
# 2026-07-26 從 main.py／tests/test_side_view_ev.py／tests/test_cloud_tracker.py
# 三份各自獨立、已產生行為漂移的實作中抽出（見 待決問題.md 一、已解決問題表格
# 第 4 項）。三個呼叫端改為呼叫本模組，不再各自維護這段邏輯。

from dataclasses import dataclass, field

from detector.ball_tracker_cloud import CloudTrackResult
from detector.centroid import BallPosition
from detector.yolo_tracker import Detection
from physics.angle import calculate_launch_angle
from physics.contact import find_contact_frame
from physics.velocity import calculate_exit_velocity_side_view


@dataclass
class BallPhysicsResult:
    """Ball Physics 計算結果（見 CONTEXT.md）。"""
    contact: int | None = None
    velocity_post: list = field(default_factory=list)
    launch_angle: float | None = None
    exit_velocity: float | None = None


def compute_ball_physics(
    ball_result: CloudTrackResult,
    bat_head_dets: dict[int, Detection],
    barrel_positions: list,
    hp_locked_bbox: tuple[float, float, float, float] | None,
    fps: float,
) -> BallPhysicsResult:
    """Contact Frame 判定 + Velocity Window 建構 + Launch Angle + Exit Velocity。

    Contact Frame 優先採用 `ball_result.contact_frame`（CloudBallTracker 內建
    判定）；為 None 時退回 `find_contact_frame()` 的距離最小值／方向改變備援。

    Velocity Window 直接取 `ball_result.positions` 中 `frame_index >= contact`
    的子集，不額外略過幀數、不重新篩選——`positions` 已是 CloudBallTracker
    驗證完成的最終軌跡點，這裡只做單純的判定與數值計算，不重複判斷資料品質
    （點數不足的情況由 `calculate_launch_angle`／`calculate_exit_velocity_side_view`
    內部的 `ValueError` 檢查接住）。

    Args:
        ball_result: `CloudBallTracker().process()` 的輸出。
        bat_head_dets: {frame_index: Detection}，Contact Frame 備援判定用。
        barrel_positions: `BarrelTracker` 輸出，Contact Frame 備援判定用。
        hp_locked_bbox: 本壘板 YOLO bbox；None 時 Exit Velocity 略過不計算。
        fps: 影片幀率（Hz）。

    Returns:
        BallPhysicsResult；Contact Frame 判定失敗時其餘欄位皆為 None／空清單。
    """
    contact = ball_result.contact_frame
    if contact is None:
        contact = find_contact_frame(
            ball_result.positions, bat_head_dets, {},
            barrel_positions=barrel_positions,
        )
    if contact is None:
        return BallPhysicsResult()

    velocity_post: list[BallPosition] = [
        p for p in ball_result.positions if p.frame_index >= contact
    ]

    launch_angle = None
    try:
        launch_angle = calculate_launch_angle(velocity_post, contact, fps)
    except ValueError:
        pass

    exit_velocity = None
    if hp_locked_bbox is not None:
        try:
            exit_velocity = calculate_exit_velocity_side_view(
                velocity_post, hp_locked_bbox, fps,
            )
        except ValueError:
            pass

    return BallPhysicsResult(
        contact=contact,
        velocity_post=velocity_post,
        launch_angle=launch_angle,
        exit_velocity=exit_velocity,
    )
