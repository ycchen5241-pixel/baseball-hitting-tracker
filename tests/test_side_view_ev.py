# tests/test_side_view_ev.py
# 側拍單軸 EV 主線腳本（CloudBallTracker + BatCloudTracker）
#
# 三個明確階段：
#   Phase 1 — 單趟 YOLO 原始偵測
#               全部 ball 偵測（非 best-per-frame，供 CloudBallTracker）
#               bat_head / bat_body best-per-frame（供 BatCloudTracker）
#               BarrelTracker（接觸幀判定備援用）
#               HP：首次高信心度偵測即鎖定 bbox，後續幀不再查詢
#   Phase 2 — 純計算篩選（不再呼叫 YOLO）
#               CloudBallTracker 六階段管線（球）
#               BatCloudTracker（棒：靜止叢集移除 → 雜訊過濾 → 揮棒視窗 → 速率離群值移除）
#               Contact 幀判定（優先用 CloudBallTracker 內建判定，備援走 physics.contact）
#               Launch Angle（像素軌跡比例法）
#               Exit Velocity（側拍單軸法）
#   Phase 3 — 重新讀取影片輸出標注影片
#
# 使用方式：
#   py -3.12 tests/test_side_view_ev.py <video_path> [--conf 0.25]
#
# 輸出：
#   <video_name>_sv_ev.mp4

import argparse
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import constants
from detector.ball_tracker_cloud import CloudBallTracker
from detector.bat_tracker_cloud import BatCloudTracker
from detector.barrel_tracker import BarrelTracker
from detector.centroid import make_raw_positions, BatPositionsInput
from detector.yolo_tracker import load_model, run_inference, detect_content_roi
from physics.pipeline import compute_ball_physics, BallPhysicsResult
from physics.trajectory import (
    extract_bat_swing_trajectory, filter_dets_by_p25, fill_bat_top_gaps_with_body,
)
from visualizer.overlay import draw_bat_trajectory, smooth_trajectory

MODEL_PATH    = "Training/best_v7.pt"
EXAMPLE_VIDEO = "Test_vod/793888764.103208.mp4"

HP_LOCK_MIN_CONF = 0.5   # HP 鎖定所需最低信心度

# ── 顏色常數（BGR）────────────────────────────────────────────────
_COL_BALL  = (0, 220, 255)
_COL_TRACK = (255, 220, 0)
_COL_HP    = (0, 255, 128)
_COL_EV    = (255, 255, 255)
_COL_TRAJ  = (0, 100, 255)


# ══════════════════════════════════════════════════════════════════
# 資料結構
# ══════════════════════════════════════════════════════════════════

@dataclass
class _RawData:
    """Phase 1 收集的純原始偵測結果，不含任何篩選邏輯。"""
    all_ball_dets: list = field(default_factory=list)   # [(frame_index, Detection)]：全部 ball 偵測
    bat_head_dets: dict = field(default_factory=dict)   # {frame_index: Detection}：best-per-frame Bat-top
    bat_body_dets: dict = field(default_factory=dict)   # {frame_index: Detection}：best-per-frame Bat
    hp_locked_bbox: tuple | None = None                 # 首次鎖定的 HP bbox
    barrel_pos: list = field(default_factory=list)      # BarrelTracker 輸出（接觸幀判定備援）
    frame_width: int = 0
    frame_height: int = 0




# ══════════════════════════════════════════════════════════════════
# Phase 1 — 單趟 YOLO 原始偵測
# ══════════════════════════════════════════════════════════════════

def _collect_raw(
    model,
    cap: cv2.VideoCapture,
    total: int,
    crop_roi,
) -> _RawData:
    """單趟全幀 YOLO 推論，同時收集所有類別。

    HP 首次達到 HP_LOCK_MIN_CONF 即鎖定 bbox，後續幀跳過 HP 查詢。
    Ball 收集全部偵測（非 best-per-frame，供 CloudBallTracker 的密度分析）。
    Bat-head / Bat-body 收集 best-per-frame。
    """
    raw    = _RawData()
    barrel = BarrelTracker()
    thr    = constants.YOLO_CONFIDENCE_THRESHOLD
    max_s  = constants.BALL_MAX_BBOX_SIDE_PX

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if fi == 0:
            raw.frame_height, raw.frame_width = frame.shape[:2]

        fd = run_inference(model, frame, fi, crop_roi)

        # ── HP 早退鎖定（靜止目標，鎖定後不再查詢）────────────
        if raw.hp_locked_bbox is None:
            hp = fd.best_by_class(constants.CLASS_HOME_PLATE)
            if hp and hp.confidence >= HP_LOCK_MIN_CONF:
                raw.hp_locked_bbox = hp.bbox
                print(
                    f"[sv_ev] HP locked @ frame {fi}"
                    f"  conf={hp.confidence:.2f}"
                    f"  bbox=({hp.bbox[0]:.0f},{hp.bbox[2]:.0f})"
                )

        # ── Ball：收集全部偵測（非 best-per-frame）───────────────
        for det in fd.get_by_class(constants.CLASS_BALL):
            if det.confidence >= thr and (det.bbox[2] - det.bbox[0]) <= max_s:
                raw.all_ball_dets.append((fi, det))

        # ── Bat 全幀收集（best-per-frame）────────────────────────
        bh = fd.best_by_class(constants.CLASS_BAT_HEAD)
        if bh:
            raw.bat_head_dets[fi] = bh
        bb = fd.best_by_class(constants.CLASS_BAT)
        if bb:
            raw.bat_body_dets[fi] = bb

        # ── 接觸偵測輔助收集 ───────────────────────────────────
        bp = barrel.update(fd, fi)
        if bp is not None:
            raw.barrel_pos.append(bp)

        if fi % 200 == 0:
            hp_status = "LOCKED" if raw.hp_locked_bbox is not None else "searching"
            print(
                f"  [collect] {fi}/{total}"
                f"  ball={len(raw.all_ball_dets)}"
                f"  bat={len(raw.bat_head_dets)}"
                f"  hp={hp_status}"
            )

    return raw


# ══════════════════════════════════════════════════════════════════
# Phase 2 — 純計算篩選（不呼叫 YOLO）
# ══════════════════════════════════════════════════════════════════

def _build_bat_trail(raw: _RawData, fps: float):
    """Bat trail pipeline：P25 過濾 → gap-fill → BatCloudTracker。

    Returns:
        (swing_pts, bat_result)：揮棒視窗內的最終軌跡點、BatCloudTrackResult。
        bat_result 為 None 表示無任何 Bat-top 偵測。
    """
    bat_head_f = filter_dets_by_p25(raw.bat_head_dets)
    bat_head_filled = fill_bat_top_gaps_with_body(bat_head_f, raw.bat_body_dets)
    bat_raw = extract_bat_swing_trajectory(bat_head_filled)
    if not bat_raw:
        return [], None

    bat_result = BatCloudTracker().process(bat_raw, fps, raw.frame_width)
    swing_pts = _swing_window_points(bat_result)
    return swing_pts, bat_result


def _swing_window_points(bat_result) -> list:
    """從乾淨軌跡中抽出落在揮棒視窗內的點——這才是最終輸出的揮棒段。

    刻意從 positions + removed_outlier（重建 Stage 5' 之前的 flight_pts）
    取點，因為 Stage 5' 的 MAD 離群值濾除可能誤刪揮棒本身的加速段。
    """
    if not bat_result.swing_windows:
        return []
    flight_pts = sorted(
        bat_result.positions + bat_result.removed_outlier,
        key=lambda p: p.frame_index,
    )
    return [
        p for p in flight_pts
        if any(lo <= p.frame_index <= hi for lo, hi in bat_result.swing_windows)
    ]


def _compute_physics(
    raw: _RawData,
    ball_result,
    fps: float,
) -> BallPhysicsResult:
    """Ball Physics 計算（接觸幀判定 + Velocity Window + Launch Angle + Exit
    Velocity），實作見 physics/pipeline.py::compute_ball_physics()（2026-07-26
    抽出，main.py／test_cloud_tracker.py 共用同一份實作，見 CONTEXT.md）。
    """
    phys = compute_ball_physics(
        ball_result, raw.bat_head_dets, raw.barrel_pos, raw.hp_locked_bbox, fps,
    )
    print(f"[sv_ev] Contact frame: {phys.contact}")
    if phys.contact is None:
        return phys

    print(f"[sv_ev] Velocity window: {len(phys.velocity_post)} pts")
    if phys.launch_angle is not None:
        print(f"[sv_ev] Launch Angle: {phys.launch_angle:.2f} deg")
    else:
        print("[sv_ev] Launch angle error: 資料品質不足")

    if raw.hp_locked_bbox is None:
        print("[sv_ev] Exit Velocity: N/A (no HP detected)")
    elif phys.exit_velocity is not None:
        print(f"[sv_ev] Exit Velocity: {phys.exit_velocity:.1f} km/h")
    else:
        print("[sv_ev] Exit Velocity: 分析失敗（資料品質不足）")

    return phys


# ══════════════════════════════════════════════════════════════════
# Phase 3 — 輸出標注影片
# ══════════════════════════════════════════════════════════════════

def _draw_hp_scale(frame: np.ndarray, bbox: tuple, px_per_cm: float) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cy = (y1 + y2) // 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), _COL_HP, 2)
    cv2.arrowedLine(frame, (x1, cy), (x2, cy), _COL_HP, 2, tipLength=0.1)
    cv2.putText(
        frame,
        f"{constants.HOME_PLATE_SIDE_VIEW_FULL_LENGTH_CM:.1f}cm = {x2 - x1}px"
        f" ({px_per_cm:.2f}px/cm)",
        (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, _COL_HP, 1,
    )


def _draw_contact_overlay(
    frame: np.ndarray, bx: float, by: float, la_deg: float, direction: int,
) -> None:
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (4, 4), (w - 4, h - 4), (0, 255, 80), 5)
    cv2.putText(frame, "CONTACT", (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 80), 3)
    rad = math.radians(la_deg)
    # LA 數值本身不含水平方向（calculate_launch_angle 對 vx 取絕對值），
    # 箭頭改參考呼叫端傳入的 velocity_post 頭尾點位移方向。
    ex = int(bx + direction * 120 * math.cos(rad))
    ey = int(by - 120 * math.sin(rad))
    cv2.arrowedLine(frame, (int(bx), int(by)), (ex, ey),
                    (0, 255, 80), 3, tipLength=0.25)
    cv2.putText(frame, f"LA {la_deg:.1f}deg", (ex + 8, ey),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 80), 2)


def _draw_data_panel(
    frame: np.ndarray, ev: float | None, la: float | None, px_per_cm: float | None,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = []
    if px_per_cm is not None:
        lines.append(f"scale {px_per_cm:.2f} px/cm")
    lines.append(f"LA  {la:.1f} deg" if la is not None else "LA  N/A")
    lines.append(f"EV  {ev:.1f} km/h" if ev is not None else "EV  N/A")

    x0 = 12
    lh = 28
    y0 = frame.shape[0] - 10 - len(lines) * lh
    for i, line in enumerate(lines):
        y = y0 + i * lh
        (tw, th), bl = cv2.getTextSize(line, font, 0.8, 2)
        cv2.rectangle(frame, (x0 - 4, y - th - bl - 2),
                      (x0 + tw + 4, y + bl), (0, 0, 0), -1)
        cv2.putText(frame, line, (x0, y), font, 0.8, _COL_EV, 2)


def _write_output(
    video_path: str,
    raw: _RawData,
    ball_result,
    bat_traj: list,
    phys: BallPhysicsResult,
    fps: float,
    width: int,
    height: int,
    out_path: str,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[sv_ev] Cannot reopen video: {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, round(fps), (width, height))

    px_per_cm = (
        (raw.hp_locked_bbox[2] - raw.hp_locked_bbox[0])
        / constants.HOME_PLATE_SIDE_VIEW_FULL_LENGTH_CM
        if raw.hp_locked_bbox is not None else None
    )

    raw_ball_map: dict[int, list] = {}
    for fi, det in raw.all_ball_dets:
        raw_ball_map.setdefault(fi, []).append(det)

    ball_map = {p.frame_index: p for p in ball_result.positions}
    vel_map = {p.frame_index: p for p in phys.velocity_post}   # Phase 2 過濾後，與 EV/LA 計算同源
    bat_traj_smoothed = smooth_trajectory(bat_traj, constants.BAT_TRAJECTORY_SMOOTH_WINDOW)
    bat_map = {p.frame_index: p for p in bat_traj_smoothed}

    sample_start = phys.contact
    traj_trail: list = []
    bat_drawn: list = []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        out = frame.copy()

        # HP 定標常駐顯示
        if raw.hp_locked_bbox is not None and px_per_cm is not None:
            _draw_hp_scale(out, raw.hp_locked_bbox, px_per_cm)

        # Ball YOLO bbox（全部偵測，細框）
        for det in raw_ball_map.get(fi, []):
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), _COL_BALL, 1)

        # Ball 追蹤點（CloudBallTracker 乾淨軌跡）
        if fi in ball_map:
            cx, cy = int(ball_map[fi].x), int(ball_map[fi].y)
            cv2.circle(out, (cx, cy), 7, _COL_TRACK, -1)

        # 飛行軌跡線：接觸後使用 Phase 2 過濾後位置，與 EV/LA 計算完全同源
        if sample_start is not None and fi >= sample_start and fi in vel_map:
            traj_trail.append(vel_map[fi])

        pts = [(int(p.x), int(p.y)) for p in traj_trail]
        for k in range(1, len(pts)):
            cv2.line(out, pts[k - 1], pts[k], _COL_TRAJ, 2)
        if pts:
            cv2.circle(out, pts[-1], 5, _COL_TRAJ, -1)

        # Bat trail（累積揮棒視窗內的軌跡）
        if fi in bat_map:
            bat_drawn.append(bat_map[fi])
        out = draw_bat_trajectory(out, bat_drawn)

        # 接觸幀疊加
        if fi == phys.contact and phys.launch_angle is not None:
            bpos = ball_map.get(fi)
            bx = bpos.x if bpos else width / 2
            by = bpos.y if bpos else height / 2
            direction = (
                1 if phys.velocity_post[-1].x >= phys.velocity_post[0].x else -1
            )
            _draw_contact_overlay(out, bx, by, phys.launch_angle, direction)

        # HUD
        cv2.putText(out, f"F:{fi:04d}", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        _draw_data_panel(out, phys.exit_velocity, phys.launch_angle, px_per_cm)
        writer.write(out)

    cap.release()
    writer.release()
    print(f"[sv_ev] Output → {out_path}")


# ══════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════

def run_side_view_ev_test(video_path: str, conf: float) -> None:
    print(f"[sv_ev] Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)

    # 直接採用宣告的錄影幀率（240fps），不用 cap.get(CAP_PROP_FPS) 的全片
    # 平均值——2026-07-26 與使用者確認：影片本身有規律性丟幀（ffprobe
    # 檢查 IMG_3294.MOV 顯示每 13-14 幀就有一次丟 2 幀的間隔，平均分布
    # 全片，導致平均值降到 209.7fps），用單一全片平均值換算任何局部
    # frame_index 差都會有系統性偏差，直接假設感光元件本身跑在名目
    # 240fps 上是較簡單的近似（見與使用者討論記錄）。
    fps    = constants.VIDEO_FPS
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[sv_ev] {width}x{height} @ {fps:.1f}fps (assumed)  ({total} frames)")

    constants.apply_fps_scaling(fps)
    constants.YOLO_CONFIDENCE_THRESHOLD = conf

    seed_frames = []
    for _ in range(min(10, total)):
        ret, f = cap.read()
        if ret:
            seed_frames.append(f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    crop_roi = detect_content_roi(seed_frames) if seed_frames else None

    # ── Phase 1: 單趟 YOLO 原始偵測 ──────────────────────────────
    print("\n[sv_ev] Phase 1: YOLO raw detection…")
    raw = _collect_raw(model, cap, total, crop_roi)
    cap.release()
    print(
        f"[sv_ev] Collected — ball={len(raw.all_ball_dets)}"
        f"  bat={len(raw.bat_head_dets)}"
        f"  hp={'LOCKED' if raw.hp_locked_bbox is not None else 'NOT FOUND'}"
    )

    if raw.hp_locked_bbox is not None:
        px_per_cm = (raw.hp_locked_bbox[2] - raw.hp_locked_bbox[0]) / constants.HOME_PLATE_SIDE_VIEW_FULL_LENGTH_CM
        print(
            f"[sv_ev] HP bbox x: {raw.hp_locked_bbox[0]:.0f}→{raw.hp_locked_bbox[2]:.0f}px"
            f"  = {constants.HOME_PLATE_SIDE_VIEW_FULL_LENGTH_CM}cm"
            f"  → {px_per_cm:.3f} px/cm"
        )

    # ── Phase 2: CloudBallTracker + BatCloudTracker + 物理計算 ───
    print("\n[sv_ev] Phase 2: CloudBallTracker…")
    # 原始逐幀球棒偵測——現狀行為刻意保留不變，見待決問題.md Candidate E
    # 討論；.raw() 讓這個選擇在呼叫端明確可見
    bat_positions = BatPositionsInput.raw(make_raw_positions(raw.bat_head_dets))
    ball_result = CloudBallTracker().process(raw.all_ball_dets, bat_positions, total)
    print(
        f"[sv_ev] Ball clean trajectory: {len(ball_result.positions)} pts"
        f"  scenario={ball_result.scenario}"
    )

    print("\n[sv_ev] Phase 2: BatCloudTracker…")
    bat_traj, bat_result = _build_bat_trail(raw, fps)
    print(
        f"[sv_ev] Bat trail: {len(bat_traj)} pts"
        f"  swing_windows={bat_result.swing_windows if bat_result else []}"
    )

    print("\n[sv_ev] Phase 2: Physics…")
    phys = _compute_physics(raw, ball_result, fps)

    # ── Phase 3: 輸出影片 ─────────────────────────────────────────
    print("\n[sv_ev] Phase 3: Writing output…")
    out_path = str(Path(video_path).with_suffix("")) + "_sv_ev.mp4"
    _write_output(video_path, raw, ball_result, bat_traj, phys, fps, width, height, out_path)

    # ── 結果摘要 ──────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  SIDE-VIEW EV REPORT (Cloud pipeline)")
    print("=" * 55)
    print(f"  Input        : {video_path}")
    if raw.hp_locked_bbox is not None:
        px_per_cm = (raw.hp_locked_bbox[2] - raw.hp_locked_bbox[0]) / constants.HOME_PLATE_SIDE_VIEW_FULL_LENGTH_CM
        print(f"  HP bbox x    : {raw.hp_locked_bbox[0]:.0f}→{raw.hp_locked_bbox[2]:.0f}px"
              f"  ({constants.HOME_PLATE_SIDE_VIEW_FULL_LENGTH_CM}cm)"
              f"  → {px_per_cm:.3f} px/cm")
    else:
        print("  HP bbox      : NOT FOUND")
    print(f"  Scenario     : {ball_result.scenario}")
    print(f"  Contact frame: {phys.contact}")
    print(f"  Launch Angle : {f'{phys.launch_angle:.2f} deg' if phys.launch_angle is not None else 'N/A'}")
    print(f"  Exit Velocity: {f'{phys.exit_velocity:.1f} km/h' if phys.exit_velocity is not None else 'N/A'}")
    print("=" * 55)


def _parse_args():
    parser = argparse.ArgumentParser(description="Side-View Single-Axis EV Test (Cloud pipeline)")
    parser.add_argument("video_path", help=f"Input video (e.g. {EXAMPLE_VIDEO})")
    parser.add_argument(
        "--conf", type=float, default=constants.YOLO_CONFIDENCE_THRESHOLD,
        help=f"YOLO confidence threshold (default: {constants.YOLO_CONFIDENCE_THRESHOLD})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_side_view_ev_test(args.video_path, args.conf)
