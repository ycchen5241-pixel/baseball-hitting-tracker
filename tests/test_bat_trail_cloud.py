# tests/test_bat_trail_cloud.py
# Bat-top 點雲式（cloud）軌跡視覺化 — 對照 test_bat_trail.py 的 arc-chain pipeline
#
# 流程：
#   Phase 1: 逐幀 YOLO 偵測 + P25 bbox 過濾（沿用 BAT_HEAD_BBOX_P25_RATIO）
#   Pipeline: BatCloudTracker（靜止叢集移除 → 雜訊段落過濾 → 速率離群值移除）
#   Phase 2: 逐幀繪製各 stage 分類 + 揮棒視窗標示 + 累積軌跡線
#   Diag:    印出各 pipeline 步驟點數（終端機）
#
# 使用方式：
#   py -3.12 tests/test_bat_trail_cloud.py <video_path> [--conf 0.25]
#
# 輸出：
#   <video_name>_bat_trail_cloud.mp4   標注影片
#   <video_name>_bat_cloud.png         點雲分布圖（XY / X-vs-time / Y-vs-time）

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import constants                                       # noqa: E402
from detector.bat_tracker_cloud import BatCloudTracker              # noqa: E402
from detector.yolo_tracker import load_model, run_inference, detect_content_roi  # noqa: E402
from physics.trajectory import (                                   # noqa: E402
    extract_bat_swing_trajectory, filter_dets_by_p25, fill_bat_top_gaps_with_body,
)
from visualizer.overlay import draw_bat_trajectory, smooth_trajectory  # noqa: E402

MODEL_PATH    = "Training/best_v7.pt"
EXAMPLE_VIDEO = "Test_vod/793471920.220478.mp4"

# ── 各 pipeline 階段的 bbox 顏色 (BGR) ───────────────────────────────
_C_CLEAN    = (  0, 255,   0)   # 綠：最終乾淨軌跡
_C_STATIC   = (100, 100, 255)   # 粉紅：被靜止叢集移除（站姿/跟隨動作定格）
_C_NOISE    = (  0, 165, 255)   # 橙：被雜訊段落移除
_C_OUTLIER  = (255,   0, 255)   # 紫：被速率離群值移除
_C_CLUSTER  = (100, 100, 255)   # 靜止叢集中心標記顏色（同 _C_STATIC）
_C_SWING    = (  0, 255, 255)   # 黃：揮棒視窗高亮

_LEGEND = [
    (_C_CLEAN,   "Bat-top: clean trajectory (final)"),
    (_C_STATIC,  "Bat-top: removed (static cluster)"),
    (_C_NOISE,   "Bat-top: removed (noise segment)"),
    (_C_OUTLIER, "Bat-top: removed (speed outlier)"),
]


# ─────────────────────────────────────────────────────────────────────
# 共用工具
# ─────────────────────────────────────────────────────────────────────

def _read_seed_frames(cap: cv2.VideoCapture, n: int = 10) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames


# ─────────────────────────────────────────────────────────────────────
# Phase 1：YOLO 偵測 + P25 過濾
# ─────────────────────────────────────────────────────────────────────

def _infer_all_frames(
    model,
    cap: cv2.VideoCapture,
    crop_roi: tuple[int, int] | None,
    total: int,
) -> tuple[dict, dict]:
    """逐幀 YOLO 推論。消耗並釋放 cap，回傳 (bat_head_dets, bat_body_dets)（各 best-per-frame）。

    Bat（整體棒子）跟 Bat-top 是同一次推論的兩個類別輸出，Bat 用來在
    Bat-top 缺偵測（動態模糊）時當補洞的備援資料源。
    """
    bat_head_dets: dict = {}
    bat_body_dets: dict = {}
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fd = run_inference(model, frame, frame_index, crop_roi)
        bat_tops = fd.get_by_class(constants.CLASS_BAT_HEAD)
        if bat_tops:
            bat_head_dets[frame_index] = max(bat_tops, key=lambda d: d.confidence)
        bat_bodies = fd.get_by_class(constants.CLASS_BAT)
        if bat_bodies:
            bat_body_dets[frame_index] = max(bat_bodies, key=lambda d: d.confidence)

        frame_index += 1
        if frame_index % 100 == 0:
            print(f"[bat-cloud] phase1  {frame_index}/{total} frames...")

    cap.release()
    print(f"[bat-cloud] YOLO Bat-top detections: {len(bat_head_dets)}/{total} "
          f"({len(bat_head_dets)/total*100:.1f}%)")
    print(f"[bat-cloud] YOLO Bat (whole) detections: {len(bat_body_dets)}/{total} "
          f"({len(bat_body_dets)/total*100:.1f}%)")
    return bat_head_dets, bat_body_dets


def _phase1_collect(
    model,
    video_path: str,
    conf_threshold: float,
) -> tuple[dict, int, float, int, int]:
    """回傳 (bat_head_dets, total_frames, fps, width, height)。"""
    cap    = cv2.VideoCapture(video_path)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[bat-cloud] {width}x{height} @ {fps:.0f}fps  {total} frames")

    seed_frames = _read_seed_frames(cap)
    crop_roi = detect_content_roi(seed_frames) if seed_frames else None
    if crop_roi:
        print(f"[bat-cloud] Content ROI (y): {crop_roi[0]} – {crop_roi[1]}")
    else:
        print("[bat-cloud] Content ROI: not detected, using full frame")

    constants.apply_fps_scaling(fps)
    constants.YOLO_CONFIDENCE_THRESHOLD = conf_threshold

    bat_head_dets, bat_body_dets = _infer_all_frames(model, cap, crop_roi, total)

    print("[bat-cloud] Applying P25 bbox filter...")
    bat_head_dets = filter_dets_by_p25(bat_head_dets)

    bat_head_dets = fill_bat_top_gaps_with_body(bat_head_dets, bat_body_dets)

    return bat_head_dets, total, fps, width, height


# ─────────────────────────────────────────────────────────────────────
# 繪製工具
# ─────────────────────────────────────────────────────────────────────

def _draw_bbox_labeled(frame, bbox, color, label) -> None:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, max(y1 - 4, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


def _draw_legend(frame) -> None:
    x, y = 10, 55
    for color, label in _LEGEND:
        cv2.rectangle(frame, (x, y - 9), (x + 13, y + 3), color, -1)
        cv2.putText(frame, label, (x + 17, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
        y += 17


def _draw_frame_info(frame, frame_index, total, in_swing) -> None:
    status = "SWING" if in_swing else "---"
    text = f"Frame {frame_index}/{total}  {status}"
    color = _C_SWING if in_swing else (255, 255, 255)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(frame, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


# ─────────────────────────────────────────────────────────────────────
# 點雲分布圖（PNG）— 對照 test_cloud_tracker.py 的 _build_trajectory_figure
# ─────────────────────────────────────────────────────────────────────

def _build_trajectory_figure(
    title: str,
    pts: list,
    raw: list,
    total: int,
    frame_width: int,
    frame_height: int,
    static_clusters: list | None = None,
    swing_windows: list | None = None,
    swing_pts: list | None = None,
) -> None:
    """三子圖共用繪製核心（XY / X vs Time / Y vs Time）。呼叫後需自行 savefig + close。

    XY 子圖座標軸固定為完整影格尺寸（對齊 IMG_3288_pointcloud.png 的慣例），
    而非自動縮放至資料點範圍，方便跨影片／跨版本比較同一個實際拍攝畫面。

    swing_pts（若提供）額外用紅色星形疊加標出——不管 pts 本身代表哪個
    stage 保留的點，都能一眼看出最終判定的揮棒視窗實際落在哪裡，不用
    跨圖比對 X/Y vs Time 子圖的黃色色塊。
    """
    static_clusters = static_clusters or []
    swing_windows = swing_windows or []
    swing_pts = swing_pts or []

    all_fi = [p.frame_index for p in raw]
    all_cx = [p.x for p in raw]
    all_cy = [p.y for p in raw]
    pt_fi  = [p.frame_index for p in pts]
    pt_cx  = [p.x for p in pts]
    pt_cy  = [p.y for p in pts]

    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    fig.suptitle(title, fontsize=11, fontweight='bold')

    ax1 = axes[0]
    if all_cx:
        sc_plot = ax1.scatter(all_cx, all_cy, c=all_fi, cmap='plasma',
                               s=8, alpha=0.35, label='all Bat-top detections (raw)')
        plt.colorbar(sc_plot, ax=ax1, label='frame index')
    if pt_cx:
        ax1.scatter(pt_cx, pt_cy, c='lime', s=34, alpha=0.9,
                    zorder=3, label=f'pts ({len(pts)})')
    if swing_pts:
        sw_cx = [p.x for p in swing_pts]
        sw_cy = [p.y for p in swing_pts]
        ax1.scatter(sw_cx, sw_cy, c='red', s=90, marker='*',
                    edgecolors='black', linewidths=0.5, zorder=5,
                    label=f'swing window pts ({len(swing_pts)})')
    for sc in static_clusters:
        ax1.plot(sc.cx, sc.cy, 'rx', markersize=14, markeredgewidth=2.5)
    ax1.set_xlabel('X (px)')
    ax1.set_ylabel('Y (px)')
    ax1.set_title('XY  (red×=static cluster center | lime=kept points | red star=swing window)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.2)
    ax1.set_xlim(0, frame_width)
    ax1.set_ylim(frame_height, 0)

    def _draw_time_axis(ax, all_y, pt_y):
        if all_fi:
            ax.scatter(all_fi, all_y, c=all_fi, cmap='plasma', s=5, alpha=0.25)
        if pt_fi:
            ax.scatter(pt_fi, pt_y, c='lime', s=22, alpha=0.9, zorder=3)
        for s, e in swing_windows:
            ax.axvspan(s, e, color='yellow', alpha=0.25)
        for sc in static_clusters:
            ax.axvspan(sc.start_frame, sc.end_frame, color='red', alpha=0.08)
        ax.set_xlabel('Frame Index')
        ax.set_xlim(0, total)
        ax.grid(True, alpha=0.2)

    ax2 = axes[1]
    _draw_time_axis(ax2, all_cx, pt_cx)
    ax2.set_ylabel('X (px)')
    ax2.set_ylim(0, frame_width)
    ax2.set_title('X vs Time  (yellow=swing window | red band=static cluster span)')

    ax3 = axes[2]
    _draw_time_axis(ax3, all_cy, pt_cy)
    ax3.set_ylabel('Y (px)')
    ax3.set_ylim(frame_height, 0)
    ax3.set_title('Y vs Time')

    plt.tight_layout()


def _swing_window_points(result, swing_windows: list) -> list:
    """從點序列中抽出落在任一揮棒視窗範圍內的點——這才是最終輸出的揮棒段，
    不是整支影片濾完的全部乾淨軌跡。

    刻意從 result.positions + result.removed_outlier（精確重建出 Stage 5'
    之前的 flight_pts）取點，不是只用 result.positions——Stage 5' 的 MAD
    離群值濾除本來就可能誤刪揮棒本身的加速段（揮棒視窗偵測本身也是刻意
    用 Stage 5' 之前的資料當輸入，理由相同），若只用 result.positions 篩，
    揮棒視窗內的點會因為被 Stage 5' 砍掉而所剩無幾。
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


def _save_pointcloud(
    raw: list,
    result,
    total: int,
    fps: float,
    frame_width: int,
    frame_height: int,
    out_path: str,
) -> None:
    if not _HAS_MATPLOTLIB:
        print("[bat-cloud] matplotlib not installed — skipping plot")
        return

    swing_pts = _swing_window_points(result, result.swing_windows)
    title = (
        f"BatCloudTracker  ({len(raw)} raw Bat-top dets, {total} frames @ {fps:.0f}fps)\n"
        f"static={len(result.static_clusters)} clusters  "
        f"clean={len(result.positions)}  "
        f"swing_pts={len(swing_pts)}  "
        f"swing_windows={result.swing_windows}"
    )
    _build_trajectory_figure(
        title, swing_pts, raw, total, frame_width, frame_height,
        static_clusters=result.static_clusters,
        swing_windows=result.swing_windows,
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[bat-cloud] pointcloud PNG → {out_path}")


# ─────────────────────────────────────────────────────────────────────
# 逐 Stage 診斷輸出（--stages）— 對照 test_cloud_tracker.py 的 _run_stages
# ─────────────────────────────────────────────────────────────────────

def _run_stages(
    raw: list, result, total: int, fps: float,
    frame_width: int, frame_height: int, base: str,
) -> None:
    if not _HAS_MATPLOTLIB:
        print("[bat-cloud] matplotlib not installed — skipping stage PNGs")
        return

    swing_pts = _swing_window_points(result, result.swing_windows)

    def save(label: str, pts: list, **kw) -> None:
        if not _HAS_MATPLOTLIB:
            return
        title = (
            f"[{label}]  pts={len(pts)}  raw={len(raw)}  "
            f"swing_pts={len(swing_pts)}  {total}f@{fps:.0f}fps"
        )
        _build_trajectory_figure(
            title, pts, raw, total, frame_width, frame_height,
            swing_pts=swing_pts, **kw,
        )
        out_path = f"{base}_bat_{label}.png"
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[stages] → {out_path}")

    # Stage 0: P25 過濾後的原始點（沿用 --stages 呼叫前已完成的 P25 過濾）
    save("s0_p25", raw)

    # Stage 1: 靜止叢集識別（顯示叢集但不移除點）
    save("s1_static_identified", raw, static_clusters=result.static_clusters)

    # Stage 2: 靜止叢集移除
    removed_static_fi = {p.frame_index for p in result.removed_static}
    after_static = [p for p in raw if p.frame_index not in removed_static_fi]
    save("s2_static_removed", after_static, static_clusters=result.static_clusters)

    # Stage 3: 時序雜訊過濾
    removed_noise_fi = {p.frame_index for p in result.removed_noise}
    after_noise = [p for p in after_static if p.frame_index not in removed_noise_fi]
    save("s3_noise_filtered", after_noise, static_clusters=result.static_clusters)

    # Stage 5': 速率離群值移除（最終乾淨軌跡）+ 揮棒視窗
    save("s5_speed_filtered", result.positions,
         static_clusters=result.static_clusters,
         swing_windows=result.swing_windows)

    print(f"[stages] all stage PNGs saved to: {Path(base).parent}")


# ─────────────────────────────────────────────────────────────────────
# Phase 2：寫出影片
# ─────────────────────────────────────────────────────────────────────

def _phase2_render(
    video_path: str,
    output_path: str,
    bat_head_dets: dict,
    result,
    total: int,
    fps: float,
    width: int,
    height: int,
) -> None:
    # 累積軌跡線只畫揮棒視窗內的點——這才是最終輸出的揮棒段，不是整支
    # 影片濾完的全部乾淨軌跡
    swing_pts = _swing_window_points(result, result.swing_windows)
    smoothed = smooth_trajectory(swing_pts, constants.BAT_TRAJECTORY_SMOOTH_WINDOW)
    traj_map = {p.frame_index: p for p in smoothed}
    clean_frames    = frozenset(p.frame_index for p in result.positions)
    static_frames   = frozenset(p.frame_index for p in result.removed_static)
    noise_frames    = frozenset(p.frame_index for p in result.removed_noise)
    outlier_frames  = frozenset(p.frame_index for p in result.removed_outlier)

    def _in_any_window(fi: int) -> bool:
        return any(s <= fi <= e for s, e in result.swing_windows)

    cap    = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    drawn: list = []
    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index in traj_map:
            drawn.append(traj_map[frame_index])

        out = draw_bat_trajectory(frame, drawn)

        if fi_bbox := bat_head_dets.get(frame_index):
            if frame_index in clean_frames:
                _draw_bbox_labeled(out, fi_bbox.bbox, _C_CLEAN, "CLEAN")
            elif frame_index in static_frames:
                _draw_bbox_labeled(out, fi_bbox.bbox, _C_STATIC, "STATIC")
            elif frame_index in noise_frames:
                _draw_bbox_labeled(out, fi_bbox.bbox, _C_NOISE, "NOISE")
            elif frame_index in outlier_frames:
                _draw_bbox_labeled(out, fi_bbox.bbox, _C_OUTLIER, "OUTLIER")

        for sc in result.static_clusters:
            cv2.drawMarker(out, (int(sc.cx), int(sc.cy)), _C_CLUSTER,
                            cv2.MARKER_CROSS, 22, 2)

        in_swing = _in_any_window(frame_index)
        if in_swing:
            cv2.rectangle(out, (4, 4), (width - 4, height - 4), _C_SWING, 3)

        _draw_legend(out)
        _draw_frame_info(out, frame_index, total, in_swing)
        writer.write(out)

        frame_index += 1
        if frame_index % 100 == 0:
            print(f"[bat-cloud] phase2  {frame_index}/{total} frames...")

    cap.release()
    writer.release()


# ─────────────────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────────────────

def run_bat_trail_cloud_test(video_path: str, conf_threshold: float, stages: bool = False) -> None:
    print(f"[bat-cloud] Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    bat_head_dets, total, fps, width, height = _phase1_collect(
        model, video_path, conf_threshold,
    )

    raw = extract_bat_swing_trajectory(bat_head_dets)
    print("\n" + "=" * 65)
    print("  BAT CLOUD PIPELINE DIAGNOSTIC")
    print("=" * 65)
    print(f"\n[0] extract_bat_swing_trajectory : {len(raw)} pts (Bat-top, P25 filtered)")

    if not raw:
        print("[bat-cloud] No bat-top trajectory — no output")
        return

    result = BatCloudTracker().process(raw, fps, width)

    print(f"\n[1-2] static cluster removal      : "
          f"{len(result.static_clusters)} cluster(s), "
          f"-{len(result.removed_static)} pts removed")
    after_static = len(raw) - len(result.removed_static)
    print(f"\n[3]   noise segment filter        : "
          f"-{len(result.removed_noise)} pts removed")
    after_noise = after_static - len(result.removed_noise)
    print(f"\n[5']  speed outlier filter         : "
          f"-{len(result.removed_outlier)} pts removed")
    print(f"\n[final] clean trajectory          : {len(result.positions)} pts")
    print(f"\n[swing windows]                   : {result.swing_windows}")
    print("\n" + "=" * 65)
    print(f"  SUMMARY: raw={len(raw)}  after_static={after_static}  "
          f"after_noise={after_noise}  final={len(result.positions)}")
    print("=" * 65 + "\n")

    # --stages 模式：逐 Stage 輸出 PNG，不跑完整分析／不輸出影片
    if stages:
        print("\n[bat-cloud] Stages mode: generating per-stage PNGs…")
        base = str(Path(video_path).with_suffix(""))
        _run_stages(raw, result, total, fps, width, height, base)
        return

    png_path = str(Path(video_path).with_suffix("")) + "_bat_cloud.png"
    _save_pointcloud(raw, result, total, fps, width, height, png_path)

    if not result.positions:
        print("[bat-cloud] No clean trajectory after pipeline — no output")
        return

    output_path = str(Path(video_path).with_suffix("")) + "_bat_trail_cloud.mp4"
    print(f"[bat-cloud] Rendering output: {output_path}")
    _phase2_render(
        video_path, output_path, bat_head_dets, result,
        total, fps, width, height,
    )
    print("[bat-cloud] Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseball Hitting Tracker - Bat-top Cloud Pipeline Diag"
    )
    parser.add_argument(
        "video_path",
        help=f"Input video path\nExample: {EXAMPLE_VIDEO}",
    )
    parser.add_argument(
        "--conf", type=float,
        default=constants.YOLO_CONFIDENCE_THRESHOLD,
        help=f"YOLO confidence threshold (default: {constants.YOLO_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--stages", action="store_true",
        help="每個 Stage 各輸出一張 PNG，不產生影片",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_bat_trail_cloud_test(args.video_path, args.conf, stages=args.stages)
