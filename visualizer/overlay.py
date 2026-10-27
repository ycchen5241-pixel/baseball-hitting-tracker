# visualizer/overlay.py
# 軌跡疊加與數據面板繪製

import cv2
import numpy as np

from detector.centroid import BallPosition
from config import constants


def _interpolate_color(
    color_start: tuple[int, int, int],
    color_end: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    """在兩個 BGR 顏色之間線性插值。

    Args:
        color_start: 起點 BGR 顏色。
        color_end: 終點 BGR 顏色。
        ratio: 插值比例（0.0 = 起點，1.0 = 終點）。

    Returns:
        插值後的 BGR 顏色。
    """
    b = int(color_start[0] + (color_end[0] - color_start[0]) * ratio)
    g = int(color_start[1] + (color_end[1] - color_start[1]) * ratio)
    r = int(color_start[2] + (color_end[2] - color_start[2]) * ratio)
    return (b, g, r)


def _lerp_ratio(i: int, n: int) -> float:
    """計算索引 i 在長度 n 序列中的線性插值比例（0.0 = 第一段起點，1.0 = 最後一段）。"""
    return (i - 1) / (n - 1) if n > 1 else 0.0


def smooth_trajectory(
    positions: list[BallPosition],
    window: int,
) -> list[BallPosition]:
    """對軌跡座標做移動平均平滑，消除偵測抖動。

    對每個點以左右各 window//2 個鄰點取均值，邊緣點縮短視窗以避免截斷。

    Args:
        positions: 按 frame_index 排序的位置序列。
        window: 移動平均視窗大小（幀數）。

    Returns:
        平滑後的位置序列（frame_index 與 confidence 不變）。
    """
    n = len(positions)
    if n < 2 or window <= 1:
        return positions

    half = window // 2
    xs = np.array([p.x for p in positions], dtype=np.float64)
    ys = np.array([p.y for p in positions], dtype=np.float64)
    xs_s = np.empty(n, dtype=np.float64)
    ys_s = np.empty(n, dtype=np.float64)

    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        xs_s[i] = np.mean(xs[lo:hi])
        ys_s[i] = np.mean(ys[lo:hi])

    return [
        BallPosition(
            x=float(xs_s[i]),
            y=float(ys_s[i]),
            frame_index=positions[i].frame_index,
            confidence=positions[i].confidence,
        )
        for i in range(n)
    ]


def _draw_gradient_trajectory(
    frame: np.ndarray,
    trajectory: list[BallPosition],
    color_start: tuple[int, int, int],
    color_end: tuple[int, int, int],
    thickness: int,
) -> np.ndarray:
    """在影像上繪製漸層顏色軌跡線段（供 draw_trajectory / draw_bat_trajectory 共用）。

    直接在傳入的 frame 上畫（in-place，不複製）——呼叫端需自行確保 frame
    是可安全修改的 buffer，避免每一層疊圖都各自防禦性複製同一張影格。
    """
    n = len(trajectory)
    if n < 2:
        return frame
    for i in range(1, n):
        color = _interpolate_color(color_start, color_end, _lerp_ratio(i, n))
        pt1 = (int(trajectory[i - 1].x), int(trajectory[i - 1].y))
        pt2 = (int(trajectory[i].x), int(trajectory[i].y))
        cv2.line(frame, pt1, pt2, color, thickness)
    return frame


def draw_trajectory(
    frame: np.ndarray,
    trajectory: list[BallPosition],
) -> np.ndarray:
    """在影像上繪製球的軌跡（紅→紫漸層）。

    Args:
        frame: BGR 影像陣列，會直接在此陣列上繪製（in-place）。
        trajectory: 按 frame_index 排序的球位置序列。

    Returns:
        疊加軌跡後的 BGR 影像（與傳入的 frame 是同一個物件）。
    """
    return _draw_gradient_trajectory(
        frame, trajectory,
        constants.TRAJECTORY_COLOR_START,
        constants.TRAJECTORY_COLOR_END,
        constants.TRAJECTORY_LINE_THICKNESS,
    )


def draw_bat_trajectory(
    frame: np.ndarray,
    trajectory: list[BallPosition],
) -> np.ndarray:
    """在影像上繪製球棒揮棒軌跡（白→黃漸層）。

    Args:
        frame: BGR 影像陣列，會直接在此陣列上繪製（in-place）。
        trajectory: 按 frame_index 排序の bat_head 位置序列。

    Returns:
        疊加揮棒軌跡後的 BGR 影像（與傳入的 frame 是同一個物件）。
    """
    return _draw_gradient_trajectory(
        frame, trajectory,
        constants.BAT_TRAJECTORY_COLOR_START,
        constants.BAT_TRAJECTORY_COLOR_END,
        constants.BAT_TRAJECTORY_LINE_THICKNESS,
    )


def _panel_lines(exit_velocity: float, launch_angle: float) -> list[str]:
    return [
        "Exit Velocity, EV",
        f"  {exit_velocity:.1f} km/h",
        "Launch Angle, LA",
        f"  {launch_angle:.2f} deg",
    ]


def _compute_panel_layout(
    img_w: int,
    img_h: int,
    lines: list[str],
) -> dict:
    """計算面板的字型、尺寸與位置。依影像短邊自適應縮放，橫直式通用。"""
    font = cv2.FONT_HERSHEY_SIMPLEX
    ref = min(img_w, img_h)
    font_scale = max(
        constants.PANEL_FONT_SCALE_MIN,
        min(ref / constants.PANEL_FONT_SCALE_REF,
            constants.PANEL_FONT_SCALE_MAX),
    )
    thickness = max(1, round(font_scale))
    line_height = max(
        constants.PANEL_LINE_HEIGHT_BASE,
        ref // constants.PANEL_LINE_HEIGHT_DIV,
    )
    padding = max(
        constants.PANEL_PADDING_BASE, ref // constants.PANEL_PADDING_DIV,
    )

    max_text_width = max(
        cv2.getTextSize(line, font, font_scale, thickness)[0][0]
        for line in lines
    )
    panel_w = max_text_width + padding * 2
    panel_h = line_height * len(lines) + padding * 2
    panel_x = img_w - panel_w - constants.PANEL_POSITION_OFFSET_X
    panel_y = img_h - panel_h - constants.PANEL_POSITION_OFFSET_Y

    return {
        "font": font,
        "font_scale": font_scale,
        "thickness": thickness,
        "line_height": line_height,
        "padding": padding,
        "panel_x": panel_x,
        "panel_y": panel_y,
        "panel_w": panel_w,
        "panel_h": panel_h,
    }


def _render_panel_background(result: np.ndarray, layout: dict) -> None:
    """在 result 上就地繪製半透明黑色背景矩形。"""
    panel_x, panel_y = layout["panel_x"], layout["panel_y"]
    panel_w, panel_h = layout["panel_w"], layout["panel_h"]
    overlay = result.copy()
    cv2.rectangle(
        overlay, (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h), (0, 0, 0), -1,
    )
    cv2.addWeighted(overlay, 0.6, result, 0.4, 0, result)


def _render_panel_text(
    result: np.ndarray, lines: list[str], layout: dict,
) -> None:
    """在 result 上就地繪製面板文字。"""
    panel_x, panel_y = layout["panel_x"], layout["panel_y"]
    padding, line_height = layout["padding"], layout["line_height"]
    for i, line in enumerate(lines):
        text_y = panel_y + padding + (i + 1) * line_height
        cv2.putText(
            result, line, (panel_x + padding, text_y),
            layout["font"], layout["font_scale"],
            (255, 255, 255), layout["thickness"],
        )


def draw_data_panel(
    frame: np.ndarray,
    exit_velocity: float,
    launch_angle: float,
) -> np.ndarray:
    """在影像右下角繪製數據面板。

    直接在傳入的 frame 上畫（in-place，不複製），理由同
    _draw_gradient_trajectory。

    Args:
        frame: BGR 影像陣列，會直接在此陣列上繪製（in-place）。
        exit_velocity: 擊球初速（km/h）。
        launch_angle: 發射仰角（degrees）。

    Returns:
        疊加數據面板後的 BGR 影像（與傳入的 frame 是同一個物件）。
    """
    h, w = frame.shape[:2]
    lines = _panel_lines(exit_velocity, launch_angle)
    layout = _compute_panel_layout(w, h, lines)
    _render_panel_background(frame, layout)
    _render_panel_text(frame, lines, layout)
    return frame
