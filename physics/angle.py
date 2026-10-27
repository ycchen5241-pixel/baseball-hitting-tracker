# physics/angle.py
# Launch Angle（發射仰角）計算
#
# 公式：θ = tan⁻¹( Vy0 / Vx0 )
#   Vy0：擊出瞬間的垂直速度分量（向上為正）
#   Vx0：擊出瞬間的水平速度分量（取絕對值，方向不影響仰角）
#
# 攝影機仰角補正：
#   仰角攝影機使影像 y 軸壓縮真實高度方向：
#   vy_physical = vy_image / cos(elevation_deg)
#   tan(corrected) = tan(raw) / cos(elevation_deg)
#   僅在 |elevation_deg| ≥ 5° 時啟動（避免浮點誤差放大）

import math

import numpy as np

from detector.centroid import BallPosition
from config import constants


def _correct_for_camera_elevation(
    raw_angle_deg: float,
    elevation_deg: float,
) -> float:
    """補正攝影機仰角對 launch angle 的垂直分量壓縮效應。

    仰角攝影機（俯攝）使影像 y 軸只涵蓋真實高度的 cos(θ) 比例：
    vy_true = vy_image / cos(elevation_deg)
    因此：tan(corrected) = tan(raw) / cos(elevation_deg)

    補正範圍限制：
    - |elevation_deg| < 5°：近水平拍攝，補正量 < 0.4%，跳過
    - cos(elevation_deg) < 0.1（>84°）：幾乎俯視，幾何意義退化，跳過

    Args:
        raw_angle_deg: 未補正的原始發射仰角（degrees）。
        elevation_deg: 攝影機仰角（degrees，俯攝為正）。

    Returns:
        補正後的發射仰角（degrees）。
    """
    if abs(elevation_deg) < 5.0:
        return raw_angle_deg
    cos_elev = math.cos(math.radians(elevation_deg))
    if cos_elev < 0.1:
        return raw_angle_deg
    tan_corrected = math.tan(math.radians(raw_angle_deg)) / cos_elev
    return math.degrees(math.atan(tan_corrected))


def calculate_launch_angle(
    positions: list[BallPosition],
    contact_frame_index: int,
    fps: float = constants.VIDEO_FPS,
    camera_elevation_deg: float = 0.0,
) -> float:
    """以接觸幀後最近的 N 幀套用 θ = tan⁻¹(Vy0 / Vx0) 計算仰角。

    採樣幀數依時間視窗（ANGLE_SAMPLE_WINDOW_MS）與 fps 動態決定，
    確保在各種幀率下覆蓋相同的物理時間長度。
    可選傳入攝影機仰角進行幾何補正。

    Args:
        positions: 球位置序列（需包含接觸幀及之後的幀）。
        contact_frame_index: 擊球幀 frame_index（作為時間原點 t=0）。
        fps: 影片實際幀率（Hz）。
        camera_elevation_deg: 攝影機仰角（degrees）；0 表示正側拍，不補正。

    Returns:
        發射仰角（degrees），向上為正。

    Raises:
        ValueError: 有效點數不足。
    """
    post = [p for p in positions if p.frame_index >= contact_frame_index]
    n_frames = max(
        constants.ANGLE_SAMPLE_MIN_FRAMES,
        min(len(post), round(fps * constants.ANGLE_SAMPLE_WINDOW_MS / 1000.0)),
    )
    sample = post[:n_frames]

    if len(sample) < constants.ANGLE_SAMPLE_MIN_FRAMES:
        raise ValueError(
            f"Need at least {constants.ANGLE_SAMPLE_MIN_FRAMES} positions "
            f"for launch angle, got {len(sample)}"
        )

    deg = constants.ANGLE_REGRESSION_DEGREE
    frame_offsets = np.array(
        [p.frame_index - contact_frame_index for p in sample],
        dtype=np.float64,
    )
    x_coords_px = np.array([p.x for p in sample], dtype=np.float64)
    y_coords_px = np.array([p.y for p in sample], dtype=np.float64)

    vx_px_frame = np.polyfit(frame_offsets, x_coords_px, deg)[0]
    vy_img_px_frame = np.polyfit(frame_offsets, y_coords_px, deg)[0]

    raw_angle = math.degrees(math.atan2(-vy_img_px_frame, abs(vx_px_frame)))
    return _correct_for_camera_elevation(raw_angle, camera_elevation_deg)
