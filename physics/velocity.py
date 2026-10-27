# physics/velocity.py
# Exit Velocity（擊球初速）計算

import math
import statistics

import numpy as np

from detector.centroid import BallPosition
from config import constants


def _theil_sen_slope(t_vals: np.ndarray, vals: np.ndarray) -> float:
    """Theil-Sen 中位數迴歸：取所有點對斜率的中位數。

    對離群值高度魯棒（最多 29.3% 資料點可為離群值）。
    適用於接觸後早期幀存在 Kalman 殘差、量測噪聲的場景。

    Args:
        t_vals: 幀號陣列（自變量）。
        vals: 對應座標陣列（x_px 或 y_cm）。

    Returns:
        中位數斜率；無有效點對時回傳 0。
    """
    slopes = []
    n = len(t_vals)
    for i in range(n):
        for j in range(i + 1, n):
            dt = t_vals[j] - t_vals[i]
            if abs(dt) < 1e-9:
                continue
            slopes.append((vals[j] - vals[i]) / dt)
    return statistics.median(slopes) if slopes else 0.0


def _sample_positions_pixel(
    positions: list[BallPosition],
    fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """依時間視窗採樣位置，回傳像素座標（不套用 Homography 轉換）。

    Returns:
        (t_vals, x_px, y_px, n_target) — 前三項為 float64 陣列。
    """
    n_target = max(
        constants.VELOCITY_SAMPLE_MIN_FRAMES,
        round(fps * constants.VELOCITY_SAMPLE_WINDOW_MS / 1000.0),
    )
    sample = positions[:n_target]
    t_vals = np.array([p.frame_index for p in sample], dtype=np.float64)
    x_px = np.array([p.x for p in sample], dtype=np.float64)
    y_px = np.array([p.y for p in sample], dtype=np.float64)
    return t_vals, x_px, y_px, n_target


def calculate_exit_velocity_side_view(
    positions: list[BallPosition],
    hp_bbox: tuple[float, float, float, float],
    fps: float,
) -> float | None:
    """側拍 xy 合成法計算 Exit Velocity。

    以本壘板 bbox 的 x 軸長度（≈ HOME_PLATE_SIDE_VIEW_FULL_LENGTH_CM）做像素定標，
    vx、vy 各自用 Theil-Sen 對同一組取樣點求斜率，再用 √(vx²+vy²) 直接合成
    速度大小——不再依賴另一個函式（calculate_launch_angle）獨立算出的仰角
    做 cos(θ) 除法補正。

    2026-07-26 與使用者討論後，從原本的「只測 vx，另外拿 calculate_launch_angle
    算出的角度做 cos(θ) 補正」改為此法，原因：
        1. 舊法的 vx（本函式，Theil-Sen，速度取樣視窗）與 θ（另一函式，
           最小平方法，角度取樣視窗，點數與窗口都不同）並非同一組資料
           算出來的，兩者是否描述同一個真實速度向量並無保證，只是碰巧
           在低仰角時數值接近。
        2. 前提「側拍下 scale_x=scale_y」若成立，vy 可直接用同樣的
           px_per_cm 轉換，不需要另外量測角度。
        3. 誤差敏感度上，1/cos(θ) 在仰角變大時會放大角度量測誤差，
           實測模擬顯示仰角 60° 時舊法對偵測噪聲的敏感度是本法的
           2.6 倍；本法（畢氏定理式合成）誤差敏感度不隨仰角變化。

    前提：相機接近純側拍（光軸垂直於投手-捕手線），image_x、image_y
    的 px/cm 尺度相同。

    Args:
        positions: 從擊球幀起的球位置序列（像素座標）。
        hp_bbox: 本壘板 YOLO bbox (x1, y1, x2, y2)。
        fps: 影片幀率（Hz）。

    Returns:
        擊球初速（km/h）；本壘板 bbox 退化時回傳 None。

    Raises:
        ValueError: 位置點數不足或 fps ≤ 0。

    Note:
        假設 positions 已是上游（CloudBallTracker/TrajectoryFitter）驗證
        完成的最終軌跡點，這裡只做單純的速度數值計算，不重複判斷資料
        品不品質。2026-07-26 與使用者討論後移除以下原有的可信度 gate，
        紀錄邏輯供之後需要時參考：
          1. _truncate_at_impact(positions)：截斷撞網/牆壁後的急停資料
             （改用 CloudBallTracker 輸出後已不含反彈段之後的點，無事可做）
          2. coverage = len(t_vals)/n_target；< EV_MIN_DETECTION_COVERAGE
             視為資料不足
          3. R²（np.polyfit 最小平方法擬合 + 決定係數）；< EV_MIN_R2
             視為軌跡不夠直，不可信
          4. 最終數值範圍檢查 EV_MIN_SPEED_KMH ~ EV_MAX_SPEED_KMH
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    if len(positions) < constants.VELOCITY_SAMPLE_MIN_FRAMES:
        raise ValueError(
            f"Need at least {constants.VELOCITY_SAMPLE_MIN_FRAMES} "
            f"positions, got {len(positions)}"
        )

    px_per_cm = (hp_bbox[2] - hp_bbox[0]) / constants.HOME_PLATE_SIDE_VIEW_FULL_LENGTH_CM
    if px_per_cm <= 0:
        print("[velocity_sv] no home plate detected (invalid bbox width)")
        return None

    # n_target/採樣視窗（VELOCITY_SAMPLE_WINDOW_MS）暫予保留：限制只取契合
    # 後一小段時間做迴歸，避免空氣阻力使全程平均速度失真代表初速——這是
    # 計算邏輯本身的一部分，跟上面移除的資料品質判斷性質不同，待用短
    # 距離擊球實際影片測試確認是否有影響後再決定去留（同上討論）。
    t_vals, x_px, y_px, _ = _sample_positions_pixel(positions, fps)

    vx_px = _theil_sen_slope(t_vals, x_px)
    vy_px = _theil_sen_slope(t_vals, y_px)

    # [px/frame] / [px/cm] × [frame/s] → [cm/s] → [m/s] → [km/h]
    v_forward_cm_s = math.hypot(vx_px, vy_px) * fps / px_per_cm
    speed_km_h = v_forward_cm_s * 0.01 * constants.MS_TO_KMH

    return speed_km_h
