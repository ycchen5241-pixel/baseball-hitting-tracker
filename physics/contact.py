# physics/contact.py
# 擊球幀判定（三路徑：距離最小值 → T 座 → 方向改變後備）

import math

from detector.centroid import BallPosition
from detector.yolo_tracker import Detection
from config import constants


def _bbox_edge_distance(
    point: tuple[float, float],
    bbox: tuple[float, float, float, float],
) -> float:
    """計算點到 bbox 最近邊緣的距離（點在 bbox 內時回傳 0）。"""
    px, py = point
    x1, y1, x2, y2 = bbox
    nx = max(x1, min(px, x2))
    ny = max(y1, min(py, y2))
    return math.sqrt((px - nx) ** 2 + (py - ny) ** 2)


def _vector_angle_deg(
    v_before: tuple[float, float],
    v_after: tuple[float, float],
) -> float:
    """計算兩個向量的夾角（degrees）。"""
    dot = v_before[0] * v_after[0] + v_before[1] * v_after[1]
    mag_before = math.sqrt(v_before[0] ** 2 + v_before[1] ** 2)
    mag_after = math.sqrt(v_after[0] ** 2 + v_after[1] ** 2)
    if mag_before == 0 or mag_after == 0:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (mag_before * mag_after)))
    return math.degrees(math.acos(cos_theta))


def _direction_change_angle_at(
    ball_positions: list[BallPosition],
    i: int,
    offset: int,
) -> float:
    """計算位置序列中索引 i 的方向改變角度（相對於前後 offset 個點）。"""
    p_before = ball_positions[i - offset]
    p_center = ball_positions[i]
    p_after = ball_positions[i + offset]
    v_before = (p_center.x - p_before.x, p_center.y - p_before.y)
    v_after = (p_after.x - p_center.x, p_after.y - p_center.y)
    return _vector_angle_deg(v_before, v_after)


def _bat_spatially_nearby(
    ball_pos: BallPosition,
    bat_head_dets: dict[int, Detection],
) -> bool:
    """判斷 bat_head 是否在指定幀附近且與球空間接近。

    在 ±CONTACT_BAT_PROXIMITY_FRAMES 範圍內，找到與球心距離
    < CONTACT_PROXIMITY_THRESHOLD_PX 的 bat_head 偵測。
    純時間鄰近無法排除球碰網/架等誤接觸情境；加入空間距離驗證
    確保方向改變時球棒確實在球附近。
    """
    radius = constants.CONTACT_BAT_PROXIMITY_FRAMES
    threshold = constants.CONTACT_PROXIMITY_THRESHOLD_PX
    frame_idx = ball_pos.frame_index
    for f in range(frame_idx - radius, frame_idx + radius + 1):
        det = bat_head_dets.get(f)
        if det is None:
            continue
        if _bbox_edge_distance((ball_pos.x, ball_pos.y), det.bbox) < threshold:
            return True
    return False


def _sustained_direction_change(
    ball_positions: list[BallPosition],
    contact_fi: int,
) -> bool:
    """確認接觸幀附近存在持續的方向改變（至少 CONTACT_DIRECTION_SUSTAINED_FRAMES 幀）。

    在 contact_fi 附近的窗口內，計算有幾幀的方向改變角度 ≥ 閾值。
    這防止量測噪聲引起的瞬間方向偏移被誤判為擊球。
    """
    offset = constants.CONTACT_VECTOR_FRAME_OFFSET
    sustained = constants.CONTACT_DIRECTION_SUSTAINED_FRAMES
    threshold = constants.CONTACT_MIN_DIRECTION_CHANGE_DEG

    contact_idx = next(
        (i for i, p in enumerate(ball_positions)
         if p.frame_index == contact_fi),
        None,
    )
    if contact_idx is None or contact_idx < offset:
        return False

    count = 0
    end = min(len(ball_positions) - offset, contact_idx + sustained + offset)
    for i in range(max(offset, contact_idx), end):
        if _direction_change_angle_at(ball_positions, i, offset) >= threshold:
            count += 1

    return count >= sustained


def _find_by_distance_minimum(
    ball_positions: list[BallPosition],
    barrel_positions: list,
) -> int | None:
    """Path A：球-棒頭距離最小值 + 接近率符號翻轉判定接觸幀。

    當球從逐漸靠近棒頭（接近率 < 0）轉為逐漸遠離（接近率 > 0）時，
    距離局部最小值即為接觸瞬間。再以方向改變持續性確認。

    比 IoU 方法更穩健：不依賴球/棒頭 bbox 重疊（幾乎從不發生）；
    不依賴 90° 方向改變（允許切線擊球場景）。
    """
    barrel_by_frame = {bp.frame_index: bp for bp in barrel_positions}
    max_dist = constants.CONTACT_PROXIMITY_THRESHOLD_PX

    dist_frames = []
    for bp in ball_positions:
        barrel = barrel_by_frame.get(bp.frame_index)
        if barrel is None:
            continue
        dist = math.sqrt((bp.x - barrel.x) ** 2 + (bp.y - barrel.y) ** 2)
        if dist <= max_dist:
            dist_frames.append((bp.frame_index, dist))

    if len(dist_frames) < 3:
        return None

    for i in range(1, len(dist_frames) - 1):
        _, d_prev = dist_frames[i - 1]
        fi, d_curr = dist_frames[i]
        _, d_next = dist_frames[i + 1]

        if d_curr <= d_prev and d_curr <= d_next:
            if _sustained_direction_change(ball_positions, fi):
                return fi

    return None


def _find_tstand_in_swing(
    tstand_candidates: list[int],
    swing_windows: list[tuple[int, int]],
) -> int | None:
    """Path B：找出在揮棒視窗內的 T 座接觸候選幀。

    T 座球被 BallTracker pre-filter 攔截，無法用軌跡方法偵測接觸；
    改用 TStandDetector 記錄的靜止球消失事件與揮棒視窗交叉比對。
    """
    margin = constants.CONTACT_WINDOW_MARGIN
    for candidate in sorted(tstand_candidates):
        for start, end in swing_windows:
            if (start - margin) <= candidate <= (end + margin):
                return candidate
    return None


def _find_contact_direction_fallback(
    ball_positions: list[BallPosition],
    bat_head_dets: dict[int, Detection],
) -> int | None:
    """Path C：最大方向改變幀 + bat 附近出現條件（後備安全網）。

    不需 ball/bat_head bbox 重疊，使用較寬鬆的 30° 閾值。
    適用於 Path A/B 均失敗的邊緣情境。
    """
    offset = constants.CONTACT_VECTOR_FRAME_OFFSET
    best_frame = None
    best_angle = constants.CONTACT_MIN_DIRECTION_CHANGE_DEG

    for i in range(offset, len(ball_positions) - offset):
        angle = _direction_change_angle_at(ball_positions, i, offset)
        p_center = ball_positions[i]
        nearby = _bat_spatially_nearby(p_center, bat_head_dets)
        if angle > best_angle and nearby:
            best_angle = angle
            best_frame = p_center.frame_index

    return best_frame


def find_contact_frame(
    ball_positions: list[BallPosition],
    bat_head_detections_by_frame: dict[int, Detection],
    ball_detections_by_frame: dict[int, Detection],
    barrel_positions: list | None = None,
    swing_windows: list | None = None,
    tstand_candidates: list | None = None,
) -> int | None:
    """執行完整的擊球幀判定（三路徑策略）。

    Path A（距離最小值）：球-棒頭距離局部最小 + 接近率符號翻轉 + 方向改變持續確認。
        適用於一般投球（前方來球）與側拋場景。
    Path B（T 座）：TStandDetector 記錄的靜止球消失事件落在揮棒視窗內。
        適用於 T 座打擊（靜止球被 pre-filter 隔離，Path A 看不到距離資訊）。
    Path C（後備）：最大方向改變幀 + 棒頭空間接近驗證（30° 閾值）。
        作為 Path A/B 均失敗時的安全網。

    Args:
        ball_positions: 排序過的球位置序列（raw YOLO，無 RANSAC 過濾）。
        bat_head_detections_by_frame: {frame_index: bat_head Detection}
        ball_detections_by_frame: {frame_index: ball Detection}（預留，Path C 備用）
        barrel_positions: BarrelTracker 輸出（含 Bat-top / 三角定位 / Bat body）。
        swing_windows: detect_swing_windows() 輸出的揮棒時間視窗。
        tstand_candidates: TStandDetector.contact_candidates。

    Returns:
        擊球幀索引（frame_index），若找不到則回傳 None。
    """
    _ = ball_detections_by_frame  # retained for API compatibility

    if barrel_positions:
        result = _find_by_distance_minimum(ball_positions, barrel_positions)
        if result is not None:
            print(f"[contact] Path A (distance minimum): frame {result}")
            return result

    if tstand_candidates and swing_windows:
        result = _find_tstand_in_swing(tstand_candidates, swing_windows)
        if result is not None:
            print(f"[contact] Path B (T-stand): frame {result}")
            return result

    result = _find_contact_direction_fallback(
        ball_positions, bat_head_detections_by_frame,
    )
    if result is not None:
        print(f"[contact] Path C (direction fallback): frame {result}")
    return result
