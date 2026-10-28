# main.py
# Baseball Hitting Tracker — 整合入口
# Phase 1 Python 原型：驗證核心偵測與計算功能

import math
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

from config import constants
from detector.barrel_tracker import BarrelTracker
from detector.bat_tracker_cloud import BatCloudTracker
from detector.ball_tracker_cloud import CloudBallTracker, CloudTrackResult
from detector.centroid import make_raw_positions, BatPositionsInput
from detector.yolo_tracker import load_model, run_inference, detect_content_roi

from physics.pipeline import compute_ball_physics
from physics.trajectory import (
    extract_bat_swing_trajectory,
    fill_bat_top_gaps_with_body,
    filter_dets_by_p25,
)
from physics.trajectory_validator import (
    validate_trajectory, TrajectoryValidation,
)
from visualizer.overlay import (
    draw_trajectory, draw_bat_trajectory, draw_data_panel, smooth_trajectory,
)


MODEL_PATH = "Training/best_v7.pt"
OUTPUT_SUFFIX = "_tracked"
HP_LOCK_MIN_CONF = 0.5


@dataclass
class _PipelineData:  # pylint: disable=too-many-instance-attributes
    """資料收集階段結果。"""
    all_ball_dets: list        # list[(frame_index, Detection)]：所有幀的全部 ball 偵測（供 CloudBallTracker）
    ball_dets: dict             # 每幀最高信心 ball YOLO 偵測（raw bbox，供 debug overlay/來源標籤）
    bat_head_dets: dict
    bat_body_dets: dict        # Bat 整體偵測（供 fill_bat_top_gaps_with_body 補洞存在性判斷）
    hp_dets: dict
    hp_locked_bbox: tuple | None   # 首個高信心 home_plate bbox（供側拍 EV 計算定標）
    barrel_pos: list           # BarrelTracker 輸出（contact_frame 判定備援用）
    frames: list


@dataclass
class _PhysicsResult:
    """物理計算結果。"""
    contact: int | None
    exit_velocity: float | None
    launch_angle: float | None
    bat_trajectory: list
    # Velocity Window：與 EV/LA 計算同源的接觸後乾淨球位置子集，繪製飛行軌跡線
    # 直接用這份資料（見 tests/test_side_view_ev.py 的繪製邏輯），不重新篩選。
    velocity_post: list = field(default_factory=list)
    # 軌跡驗證結果（僅供維護/除錯；不影響使用者數據面板）
    validation: TrajectoryValidation | None = None
    ball_result: CloudTrackResult | None = None


@dataclass
class _VideoCtx:
    """輸出影片相關參數。"""
    fps: float
    width: int
    height: int
    output_path: str


def _collect_class_detections(
    data: _PipelineData,
    frame_detections,
    frame_index: int,
) -> None:
    """從單幀 YOLO 結果中收集 ball（全部）/ bat_head / bat / home_plate 偵測。"""
    thr = constants.YOLO_CONFIDENCE_THRESHOLD
    max_side = constants.BALL_MAX_BBOX_SIDE_PX
    for det in frame_detections.get_by_class(constants.CLASS_BALL):
        if det.confidence >= thr and (det.bbox[2] - det.bbox[0]) <= max_side:
            data.all_ball_dets.append((frame_index, det))

    ball = frame_detections.best_by_class(constants.CLASS_BALL)
    if (ball and ball.confidence > thr
            and (ball.bbox[2] - ball.bbox[0]) <= max_side
            and (ball.bbox[3] - ball.bbox[1]) <= max_side):
        data.ball_dets[frame_index] = ball

    bat_head = frame_detections.best_by_class(constants.CLASS_BAT_HEAD)
    if bat_head:
        data.bat_head_dets[frame_index] = bat_head

    bat_body = frame_detections.best_by_class(constants.CLASS_BAT)
    if bat_body:
        data.bat_body_dets[frame_index] = bat_body

    hp_det = frame_detections.best_by_class(constants.CLASS_HOME_PLATE)
    if hp_det:
        data.hp_dets[frame_index] = hp_det
        if data.hp_locked_bbox is None and hp_det.confidence >= HP_LOCK_MIN_CONF:
            data.hp_locked_bbox = hp_det.bbox
            print(f"[main] HP locked @ f{frame_index} conf={hp_det.confidence:.2f}")


def _collect_data(
    model,
    cap: cv2.VideoCapture,
    total_frames: int,
) -> _PipelineData:
    """執行 YOLO 推論，收集所有幀的完整偵測資料（供 CloudBallTracker 等後續分析使用）。"""
    data = _PipelineData(
        all_ball_dets=[],
        ball_dets={},
        bat_head_dets={},
        bat_body_dets={},
        hp_dets={},
        hp_locked_bbox=None,
        barrel_pos=[],
        frames=[],
    )
    barrel = BarrelTracker()

    # 自動偵測 content ROI（過濾螢幕錄影的 UI 黑邊）
    seed_frames = []
    for _ in range(min(10, total_frames)):
        ret, f = cap.read()
        if ret:
            seed_frames.append(f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    crop_roi = detect_content_roi(seed_frames) if seed_frames else None

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_detections = run_inference(model, frame, frame_index, crop_roi)
        _collect_class_detections(data, frame_detections, frame_index)

        bp = barrel.update(frame_detections, frame_index)
        if bp is not None:
            data.barrel_pos.append(bp)

        data.frames.append(frame.copy())
        frame_index += 1
        if frame_index % 100 == 0:
            print(f"[main] {frame_index}/{total_frames} frames processed")

    return data


def _swing_window_points(result, swing_windows: list) -> list:
    """從點序列中抽出落在任一揮棒視窗範圍內的點——這才是最終要用的揮棒軌跡。

    刻意從 result.positions + result.removed_outlier（精確重建出 Stage 5'
    之前的 flight_pts）取點，不是只用 result.positions——Stage 5' 的 MAD
    離群值濾除本來就可能誤刪揮棒本身的加速段。
    """
    if not swing_windows:
        return []
    flight_pts = sorted(
        result.positions + result.removed_outlier, key=lambda p: p.frame_index,
    )
    return [
        p for p in flight_pts
        if any(lo <= p.frame_index <= hi for lo, hi in swing_windows)
    ]


def _build_bat_trajectory(
    bat_head_dets: dict,
    bat_body_dets: dict,
    fps: float,
    bat_fps: float,
    frame_width: int,
) -> list:
    """球棒揮棒軌跡構建：BatCloudTracker（靜止叢集/雜訊過濾 → 轉向角+片段合併
    找揮棒視窗 → MAD 離群值濾除），回傳揮棒視窗內的點。

    swing_windows 本身已用 _cut_at_extremum 裁到延伸極限（不含拉回段），
    不需要再額外裁切。

    P25 過濾只在此函式內部套用、只影響這份揮棒軌跡——呼叫端傳入的
    bat_head_dets 本身維持未過濾（與 tests/test_side_view_ev.py::_build_bat_trail
    一致），CloudBallTracker 的球棒佐證訊號與 compute_ball_physics() 的
    contact 備援訊號才不會意外吃到跟這裡不同調的過濾版本。

    球棒軌跡判定專用 bat_fps（cv2 實測平均幀率，與
    tests/test_bat_trail_cloud.py 同源）暫時覆蓋全域 fps 縮放常數——靜止
    叢集/雜訊過濾/轉向角切窗全靠這組全域常數決定門檻，跟 EV/LA 計算用的
    宣告幀率 fps（constants.VIDEO_FPS，理由見 process_video() 註解）是兩回
    事，兩者 2026-10-27 討論後刻意分離：EV/LA 要準確的物理時間換算，球棒
    軌跡判定則要跟 test_bat_trail_cloud.py 同調。離開本函式前必須還原成
    fps，否則會汙染後續 CloudBallTracker/compute_ball_physics 用到的門檻。
    """
    constants.apply_fps_scaling(bat_fps)
    try:
        bat_head_f = filter_dets_by_p25(bat_head_dets)
        bat_head_filled = fill_bat_top_gaps_with_body(bat_head_f, bat_body_dets)
        raw = extract_bat_swing_trajectory(bat_head_filled)
        if not raw:
            return []
        result = BatCloudTracker().process(raw, bat_fps, frame_width)
        return _swing_window_points(result, result.swing_windows)
    finally:
        constants.apply_fps_scaling(fps)


def _log_velocity_diagnostics(velocity_post: list, fps: float) -> None:
    """輸出速度計算診斷資訊：可用點數與各幀像素座標。"""
    n_target = max(
        constants.VELOCITY_SAMPLE_MIN_FRAMES,
        round(fps * constants.VELOCITY_SAMPLE_WINDOW_MS / 1000.0),
    )
    n_avail = len(velocity_post)
    print(
        f"[diag] velocity_positions after contact:"
        f" {n_avail} (need {n_target} for full window)"
    )
    for p in velocity_post[:min(n_target, n_avail)]:
        print(
            f"  f{p.frame_index:04d}"
            f" px=({p.x:.1f},{p.y:.1f})"
            f" conf={p.confidence:.2f}"
        )


def _run_physics(
    data: _PipelineData,
    fps: float,
    bat_fps: float,
    total_frames: int,
    frame_width: int,
) -> _PhysicsResult:
    """CloudBallTracker 球體分析、軌跡擬合，Ball Physics 交給 compute_ball_physics()。

    2026-07-26 抽出 physics/pipeline.py::compute_ball_physics()（見 CONTEXT.md
    「Ball Physics」），main.py／test_side_view_ev.py／test_cloud_tracker.py
    三邊改為共用同一份 contact_frame 判定 + velocity_post 建構 + Launch
    Angle + Exit Velocity 實作，不再各自維護（原待決問題 A 節分歧的根因）。
    """
    bat_traj = _build_bat_trajectory(data.bat_head_dets, data.bat_body_dets, fps, bat_fps, frame_width)
    print(f"[main] Bat swing trajectory: {len(bat_traj)} frames")

    print("[main] CloudBallTracker: analyzing ball trajectory...")
    # 原始逐幀球棒偵測（不是 bat_traj 那份清洗過的版本）——現狀行為刻意
    # 保留不變，見待決問題.md Candidate E 討論；.raw() 讓這個選擇在呼叫端
    # 明確可見，不再靠巧合跟 test_cloud_tracker.py 的 .cleaned() 撞出同型別
    bat_positions = BatPositionsInput.raw(make_raw_positions(data.bat_head_dets))
    ball_result = CloudBallTracker().process(
        data.all_ball_dets, bat_positions, total_frames,
    )
    print(
        f"[main] Cloud scenario={ball_result.scenario}"
        f"  static={len(ball_result.static_blacklist)}"
        f"  clean={len(ball_result.positions)}"
    )

    phys = compute_ball_physics(
        ball_result, data.bat_head_dets, data.barrel_pos, data.hp_locked_bbox, fps,
    )
    print(f"[main] Contact frame: {phys.contact}")
    if phys.contact is None:
        print("[main] No contact frame detected")
        return _PhysicsResult(None, None, None, bat_traj, [], None, ball_result)

    _log_velocity_diagnostics(phys.velocity_post, fps)
    if phys.launch_angle is not None:
        print(f"[main] Launch Angle: {phys.launch_angle:.2f} deg")
    else:
        print("[main] Launch Angle: 分析失敗（資料品質不足）")
    if phys.exit_velocity is not None:
        print(f"[main] Exit Velocity: {phys.exit_velocity:.1f} km/h")
    elif data.hp_locked_bbox is None:
        print("[main] Exit Velocity: N/A (home plate not locked)")
    else:
        print("[main] Exit Velocity: 分析失敗（資料品質不足）")

    pre_positions = [p for p in ball_result.flight_pts if p.frame_index < phys.contact]

    val = validate_trajectory(
        pre_positions, phys.velocity_post, bat_traj or None,
        contact_fi=phys.contact,
    )
    val.log()

    return _PhysicsResult(
        phys.contact, phys.exit_velocity, phys.launch_angle, bat_traj,
        phys.velocity_post, val, ball_result,
    )


class _DebugOverlay:
    """偵測視覺化 debug 疊加層。

    2026-07-27 從 main.py 裡 7 個只在彼此之間互相呼叫、只讀單一影片
    的固定情境（`data`/`result`/`ball_pos_map` 全片不變）的函式收攏成
    一個類別——建構一次，`render()` 是唯一對外介面。

    繪製內容：
    - 黃框：YOLO 原始偵測 bbox
    - 青色圓點：CloudBallTracker 乾淨球心
    - 綠色邊框 + CONTACT 標籤：接觸幀
    - 仰角箭頭：從接觸點沿仰角方向延伸
    - 左上角 HUD：幀號、偵測狀態
    - 右上角灰字：軌跡驗證摘要（僅供維護/除錯）
    """

    def __init__(
        self,
        data: _PipelineData,
        result: _PhysicsResult,
        ball_pos_map: dict,
    ):
        self._data = data
        self._result = result
        self._ball_pos_map = ball_pos_map
        self._font = cv2.FONT_HERSHEY_SIMPLEX

    def render(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """在 frame 上疊加偵測/除錯資訊（in-place，frame 需為可安全修改的 buffer）。"""
        out = frame
        self._draw_yolo_bbox(out, frame_idx)
        self._draw_tracked_ball(out, frame_idx)
        self._draw_contact_overlay(out, frame_idx)
        self._draw_hud(out, frame_idx)
        self._draw_validation_debug(out)
        return out

    def _draw_yolo_bbox(self, out: np.ndarray, frame_idx: int) -> None:
        """繪製 YOLO 原始偵測框（黃色）。"""
        if frame_idx not in self._data.ball_dets:
            return
        det = self._data.ball_dets[frame_idx]
        x1, y1, x2, y2 = (int(v) for v in det.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 220, 255), 2)
        cv2.putText(
            out, f"YOLO {det.confidence:.2f}",
            (x1, y1 - 6), self._font, 0.5, (0, 220, 255), 1,
        )

    def _draw_tracked_ball(self, out: np.ndarray, frame_idx: int) -> None:
        """繪製 CloudBallTracker 乾淨球心（青色實心圓）。"""
        if frame_idx not in self._ball_pos_map:
            return
        pos = self._ball_pos_map[frame_idx]
        center = (int(pos.x), int(pos.y))
        cv2.circle(out, center, 7, (255, 220, 0), -1)
        cv2.circle(out, center, 7, (255, 255, 255), 1)

    def _get_contact_ball_pos(
        self, frame_idx: int, w: int, h: int,
    ) -> tuple[float, float]:
        """取得接觸幀球心座標；未找到時退回影像中心。"""
        contact_pos = self._ball_pos_map.get(frame_idx)
        if contact_pos is not None:
            return contact_pos.x, contact_pos.y
        det = self._data.ball_dets.get(frame_idx)
        if det is not None:
            return (det.bbox[0] + det.bbox[2]) / 2, (det.bbox[1] + det.bbox[3]) / 2
        return w / 2, h / 2

    def _draw_contact_overlay(self, out: np.ndarray, frame_idx: int) -> None:
        """繪製接觸幀綠色邊框、CONTACT 標籤與仰角箭頭。"""
        result = self._result
        if result.contact is None or frame_idx != result.contact:
            return
        h, w = out.shape[:2]
        cv2.rectangle(out, (4, 4), (w - 4, h - 4), (0, 255, 80), 6)
        cv2.putText(out, "CONTACT", (20, 70), self._font, 2.0, (0, 255, 80), 3)
        if result.launch_angle is None:
            return
        bx, by = self._get_contact_ball_pos(frame_idx, w, h)
        angle_rad = math.radians(result.launch_angle)
        arrow_len = constants.LAUNCH_ARROW_LEN_PX
        # LA 數值本身不含水平方向（calculate_launch_angle 對 vx 取絕對值），
        # 箭頭改參考 velocity_post 頭尾點的 x 位移正負號決定往左或往右畫。
        direction = (
            1 if result.velocity_post[-1].x >= result.velocity_post[0].x else -1
        )
        ex = int(bx + direction * arrow_len * math.cos(angle_rad))
        ey = int(by - arrow_len * math.sin(angle_rad))   # 影像 y 軸向下
        cv2.arrowedLine(
            out, (int(bx), int(by)), (ex, ey), (0, 255, 80), 3, tipLength=0.25,
        )
        cv2.putText(
            out, f"LA {result.launch_angle:.1f}deg",
            (ex + 8, ey), self._font, 0.7, (0, 255, 80), 2,
        )

    def _get_detection_source(
        self, frame_idx: int,
    ) -> tuple[str, tuple[int, int, int]]:
        """回傳偵測來源標籤與對應顏色。"""
        if frame_idx in self._data.ball_dets:
            return "BALL", (0, 220, 255)    # 黃色：YOLO
        return "----", (120, 120, 120)

    def _draw_hud(self, out: np.ndarray, frame_idx: int) -> None:
        """繪製左上角 HUD（幀號、偵測來源、追蹤狀態）。"""
        src, src_color = self._get_detection_source(frame_idx)
        cloud_hit = "TRK" if frame_idx in self._ball_pos_map else "---"
        cv2.putText(
            out, f"F:{frame_idx:04d}  {src}  {cloud_hit}",
            (16, 36), self._font, 0.7, src_color, 1,
        )

    def _draw_validation_debug(self, out: np.ndarray) -> None:
        """軌跡驗證摘要面板（右上角，灰色小字）。

        ⚠️  此資訊僅供開發維護與除錯使用，實際使用者無需理解這些數據。
        顯示軌跡合法性驗證器的原始輸出：偵測品質指標、物理公理核對結果、
        速度突刺幀索引，以及軌跡是否被抑制等內部決策狀態。
        """
        val = self._result.validation
        if val is None:
            return

        post_str = "OK" if val.post_valid else "FAIL"
        bat_str = (
            f"  bat={'OK' if val.bat_valid else 'FAIL'}"
            if val.bat_valid is not None else ""
        )
        lines: list[str] = [
            "-- DEV DEBUG --",
            "[僅供維護/除錯]",
            f"{'VALID' if val.overall_valid else 'INVALID'}"
            f"  conf={val.confidence:.2f}",
            f"cov={val.detection_coverage:.0%}  pre={val.pre_mode.mode}",
            f"post={post_str}{bat_str}",
        ]
        if val.spike_frames:
            lines.append(f"spikes:{val.spike_frames[:4]}")
        suppressed = (
            (["traj"] if val.suppress_trajectory else [])
            + (["bat"] if val.suppress_bat else [])
        )
        if suppressed:
            lines.append(f"suppressed:{','.join(suppressed)}")

        line_h = 16
        x = out.shape[1] - 200
        y_start = 30
        color_hdr = (200, 160, 100)  # 暖橘灰：標示開發資訊區塊
        color_txt = (150, 150, 150)  # 冷灰：數值，不干擾正常輸出

        for k, line in enumerate(lines):
            cv2.putText(
                out, line, (x, y_start + k * line_h),
                self._font, 0.4, color_hdr if k < 2 else color_txt, 1,
            )


def _write_output(
    data: _PipelineData,
    result: _PhysicsResult,
    ctx: _VideoCtx,
) -> None:
    """繪製軌跡與數據面板並寫出標注影片。"""
    # fps 取整避免非整數 timebase 超出 MPEG-4 規格限制（上限 65535）
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        ctx.output_path, fourcc, round(ctx.fps), (ctx.width, ctx.height)
    )
    ball_positions = result.ball_result.positions if result.ball_result else []
    ball_pos_map = {p.frame_index: p for p in ball_positions}
    debug_overlay = _DebugOverlay(data, result, ball_pos_map)

    # 球棒軌跡平滑
    bat_traj_smoothed = smooth_trajectory(
        result.bat_trajectory, constants.BAT_TRAJECTORY_SMOOTH_WINDOW
    )
    bat_traj_map = {p.frame_index: p for p in bat_traj_smoothed}

    # 球飛行軌跡：與 EV/LA 計算同源的 Velocity Window（見 CONTEXT.md），跟
    # tests/test_side_view_ev.py 的繪製邏輯一致，不再額外跳過接觸幀本身。
    actual_flight = list(result.velocity_post)

    # ── 維護/除錯決策：依驗證結果修正軌跡與繪製行為 ──────────────
    # 以下三個決策對實際使用者不可見（僅影響輸出影片品質），
    # 實際使用者無需理解這些數據，詳情請見 _draw_validation_debug。
    val = result.validation
    if val is not None and val.spike_frames:
        # 移除 Validator 標記的速度突刺幀（通常為兩靜止球誤連的特徵）
        spike_set = set(val.spike_frames)
        actual_flight = [
            p for p in actual_flight if p.frame_index not in spike_set
        ]
    # 品質不足時抑制軌跡繪製，避免誤導使用者
    show_ball_traj = val is None or not val.suppress_trajectory
    show_bat_traj = val is None or not val.suppress_bat

    # 不做移動平均平滑：velocity_post 的點之間 frame_index 間距不固定（接觸
    # 瞬間常有偵測缺口），對索引鄰居取平均會跨過真實時間間距硬拉點位置，
    # 跟 tests/test_side_view_ev.py 一致改成直接畫原始點。
    actual_traj_map = {p.frame_index: p for p in actual_flight}

    bat_drawn: list = []
    drawn: list = []

    for i, frame in enumerate(data.frames):
        # 每幀固定只複製一次：data.frames[i] 是 Phase 1 收集的原始影格，
        # 後面所有疊圖步驟都直接在這份複製後的 buffer 上原地繪製
        # （in-place），不再各自防禦性複製一次。
        out = frame.copy()
        # 球棒軌跡：從揮棒開始逐幀累積（BatCloudTracker 的 swing_windows
        # 已裁到延伸極限，不含拉回弧，不需再額外裁切）
        if i in bat_traj_map:
            bat_drawn.append(bat_traj_map[i])
        # 球飛行軌跡：只累積實際偵測點（非多項式插值）
        if i in actual_traj_map:
            drawn.append(actual_traj_map[i])
        if show_bat_traj:
            draw_bat_trajectory(out, bat_drawn)
        if show_ball_traj:
            draw_trajectory(out, drawn)
        # 偵測 debug 疊加（含右上角維護/除錯用驗證摘要）
        debug_overlay.render(out, i)
        if result.exit_velocity is not None or result.launch_angle is not None:
            ev = result.exit_velocity or 0.0
            la = result.launch_angle or 0.0
            draw_data_panel(out, ev, la)
        writer.write(out)

    writer.release()
    print(f"[main] Output saved: {ctx.output_path}")


def process_video(video_path: str) -> None:
    """對指定影片執行完整擊球分析流程。

    Args:
        video_path: 輸入影片路徑。
    """
    print(f"[main] Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    # 直接採用宣告的錄影幀率，不用 cap.get(CAP_PROP_FPS) 的全片平均值
    # （理由與 tests/test_side_view_ev.py 相同，2026-07-26 討論記錄）。
    fps = constants.VIDEO_FPS
    # 球棒揮棒軌跡判定改用 cv2 實測平均幀率——跟 EV/LA 計算的時間基準
    # （宣告幀率較準，見上）是兩回事：球棒軌跡只在乎跟
    # tests/test_bat_trail_cloud.py 同調，不涉及真實物理時間換算
    # （2026-10-27 討論記錄，見 _build_bat_trajectory()）。
    bat_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(
        f"[main] {width}x{height} @ {fps:.0f}fps (assumed, EV/LA)"
        f" / {bat_fps:.0f}fps (measured, bat trajectory)  {total} frames"
    )

    constants.apply_fps_scaling(fps)
    data = _collect_data(model, cap, total)
    cap.release()

    print(f"[main] Raw ball detections: {len(data.all_ball_dets)}")

    result = _run_physics(data, fps, bat_fps, total, width)

    out_path = video_path.rsplit(".", 1)[0] + OUTPUT_SUFFIX + ".mp4"
    ctx = _VideoCtx(
        fps=fps, width=width, height=height, output_path=out_path
    )
    _write_output(data, result, ctx)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.12 main.py <video_path>")
        print("Example: py -3.12 main.py Test_vod/videoplayback.mp4")
        sys.exit(1)

    process_video(sys.argv[1])
