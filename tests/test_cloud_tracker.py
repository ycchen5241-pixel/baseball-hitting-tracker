# tests/test_cloud_tracker.py
# CloudBallTracker 點雲式追蹤器測試腳本
#
# Phase 1 — YOLO 推論，收集所有 ball 偵測（非 best-per-frame）
# Phase 2 — CloudBallTracker 六階段管線
# Phase 3 — 輸出點雲圖（PNG）+ 標注影片（MP4）
#
# 使用方式：
#   py -3.12 tests/test_cloud_tracker.py <video_path> [--conf 0.25]

import argparse
import sys
from dataclasses import dataclass, field
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

from config import constants
from detector.ball_tracker_cloud import (
    CloudBallTracker, CloudTrackResult,
    _p25_filter, _build_static_clusters, _remove_static,
    _filter_noise_segments, _detect_scenario, _identify_tstand,
    _solve_tstand_contact, _filter_axis_outliers, _truncate_at_impact,
    _merge_flight_extension_segment,
)
from detector.bat_tracker_cloud import BatCloudTracker
from detector.barrel_tracker import BarrelTracker
from detector.centroid import BallPosition, BatPositionsInput
from detector.trajectory_fitter import fit_trajectory_segments
from detector.yolo_tracker import load_model, run_inference, detect_content_roi
from physics.pipeline import compute_ball_physics, BallPhysicsResult
from physics.trajectory import (
    extract_bat_swing_trajectory, filter_dets_by_p25, fill_bat_top_gaps_with_body,
)

MODEL_PATH = "Training/best_v7.pt"
HP_LOCK_MIN_CONF = 0.5

# OpenCV BGR 格式
_C_CLOUD   = (0, 200, 255)   # 橙黃
_C_STATIC  = (100, 100, 255) # 粉紅
_C_TSTAND  = (0, 255, 128)   # 黃綠
_C_CONTACT = (255, 100, 0)   # 藍


# ══════════════════════════════════════════════════════════════════════
# Phase 1：YOLO 推論 + 資料收集
# ══════════════════════════════════════════════════════════════════════

@dataclass
class _RawData:
    all_ball_dets:  list  = field(default_factory=list)  # list[(fi, Detection)]
    bat_head_dets:  dict  = field(default_factory=dict)  # {fi: Detection}
    bat_body_dets:  dict  = field(default_factory=dict)
    hp_locked_bbox: tuple | None = None
    barrel_pos:     list  = field(default_factory=list)
    frame_width:    int   = 0
    frame_height:   int   = 0


def _collect_raw(model, cap, total: int, crop_roi) -> _RawData:
    raw     = _RawData()
    barrel  = BarrelTracker()
    thr     = constants.YOLO_CONFIDENCE_THRESHOLD
    max_s   = constants.BALL_MAX_BBOX_SIDE_PX

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if fi == 0:
            raw.frame_height, raw.frame_width = frame.shape[:2]

        fd = run_inference(model, frame, fi, crop_roi)

        # HP 鎖定
        if raw.hp_locked_bbox is None:
            hp = fd.best_by_class(constants.CLASS_HOME_PLATE)
            if hp and hp.confidence >= HP_LOCK_MIN_CONF:
                raw.hp_locked_bbox = hp.bbox
                print(f"  [collect] HP locked @ f{fi} conf={hp.confidence:.2f}")

        # 收集所有 ball 偵測（不只是 best）
        for det in fd.get_by_class(constants.CLASS_BALL):
            if det.confidence >= thr and (det.bbox[2] - det.bbox[0]) <= max_s:
                raw.all_ball_dets.append((fi, det))

        # Bat / Barrel
        bh = fd.best_by_class(constants.CLASS_BAT_HEAD)
        if bh:
            raw.bat_head_dets[fi] = bh
        bb = fd.best_by_class(constants.CLASS_BAT)
        if bb:
            raw.bat_body_dets[fi] = bb
        bp = barrel.update(fd, fi)
        if bp is not None:
            raw.barrel_pos.append(bp)

        if fi % 300 == 0:
            print(f"  [collect] f{fi}/{total}"
                  f"  all_ball={len(raw.all_ball_dets)}"
                  f"  bat={len(raw.bat_head_dets)}")

    return raw


def _build_bat_positions(raw: _RawData, fps: float) -> list:
    """Bat trail pipeline：P25 過濾 → gap-fill → BatCloudTracker，供接觸幀
    判斷用清洗過的球棒軌跡（見 PHYSICS_VALIDATION_PRINCIPLES.md B2「球棒的
    角色」——信心度佐證用途，資料品質不好會讓佐證本身不可靠）。

    Returns:
        落在揮棒視窗內的乾淨球棒軌跡點（list[BallPosition]）；無 Bat-top
        偵測時回傳空清單。
    """
    bat_head_f = filter_dets_by_p25(raw.bat_head_dets)
    bat_head_filled = fill_bat_top_gaps_with_body(bat_head_f, raw.bat_body_dets)
    bat_raw = extract_bat_swing_trajectory(bat_head_filled)
    if not bat_raw:
        return []

    bat_result = BatCloudTracker().process(bat_raw, fps, raw.frame_width)
    flight_pts = sorted(
        bat_result.positions + bat_result.removed_outlier,
        key=lambda p: p.frame_index,
    )
    return [
        p for p in flight_pts
        if any(lo <= p.frame_index <= hi for lo, hi in bat_result.swing_windows)
    ]


# ══════════════════════════════════════════════════════════════════════
# 物理計算
# ══════════════════════════════════════════════════════════════════════

def _compute_physics(
    ball_result: CloudTrackResult,
    bat_head_dets: dict,
    barrel_pos: list,
    hp_locked_bbox: tuple | None,
    fps: float,
    label: str,
) -> BallPhysicsResult:
    """Ball Physics 計算，實作見 physics/pipeline.py::compute_ball_physics()
    （2026-07-26 抽出，main.py／test_side_view_ev.py 共用同一份實作，見
    CONTEXT.md）。此前這裡呼叫 find_contact_frame() 時漏傳 barrel_positions，
    Path A（距離最小值）備援訊號實際上從未生效；改用共用實作後一併修正。
    """
    res = compute_ball_physics(
        ball_result, bat_head_dets, barrel_pos, hp_locked_bbox, fps,
    )
    print(f"[{label}] contact frame = {res.contact}")
    if res.contact is None:
        return res

    print(f"[{label}] velocity window: {len(res.velocity_post)} pts")
    if res.launch_angle is not None:
        print(f"[{label}] LA = {res.launch_angle:.2f} deg")
    else:
        print(f"[{label}] LA error: 資料品質不足")

    if hp_locked_bbox is not None and res.exit_velocity is not None:
        print(f"[{label}] EV = {res.exit_velocity:.1f} km/h")
    elif hp_locked_bbox is not None:
        print(f"[{label}] EV error: 資料品質不足")

    return res

    return res


# ══════════════════════════════════════════════════════════════════════
# Stage 逐步診斷輸出
# ══════════════════════════════════════════════════════════════════════

def _build_trajectory_figure(
    title: str,
    pts: list,
    raw: _RawData,
    total: int,
    static_clusters: list | None = None,
    tstand=None,
    contact_frame: int | None = None,
    impact_frame: int | None = None,
    impact_xy: tuple | None = None,
) -> None:
    """三子圖共用繪製核心（XY / X vs Time / Y vs Time）。呼叫後需自行 savefig + close。"""
    sc_list = static_clusters or []
    all_fi  = [fi for fi, _ in raw.all_ball_dets]
    all_cx  = [(d.bbox[0] + d.bbox[2]) / 2 for _, d in raw.all_ball_dets]
    all_cy  = [(d.bbox[1] + d.bbox[3]) / 2 for _, d in raw.all_ball_dets]
    pt_fi   = [p.frame_index for p in pts]
    pt_cx   = [p.x for p in pts]
    pt_cy   = [p.y for p in pts]

    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    ax1 = axes[0]
    if all_cx:
        sc_plot = ax1.scatter(all_cx, all_cy, c=all_fi, cmap='plasma',
                              s=6, alpha=0.3, label='all YOLO')
        plt.colorbar(sc_plot, ax=ax1, label='frame index')
    if pt_cx:
        ax1.scatter(pt_cx, pt_cy, c='lime', s=30, alpha=0.9,
                    zorder=3, label=f'pts ({len(pts)})')
    for scc in sc_list:
        ax1.plot(scc.cx, scc.cy, 'rx', markersize=14, markeredgewidth=2.5)
    if tstand:
        ax1.plot(tstand.cx, tstand.cy, 'g*', markersize=18, label='T-stand')

    # 接觸幀標記：pts 裡離 contact_frame 最近的那個點，用顯眼顏色單獨標出
    contact_pt = None
    if contact_frame is not None and pts:
        contact_pt = min(pts, key=lambda p: abs(p.frame_index - contact_frame))
        ax1.plot(contact_pt.x, contact_pt.y, marker='*', color='red',
                  markersize=22, markeredgecolor='black', markeredgewidth=1,
                  label=f'contact f{contact_frame}', zorder=6)

    if impact_xy is not None:
        ax1.plot(impact_xy[0], impact_xy[1], 'c^', markersize=16, markeredgewidth=2,
                 label=f'impact f{impact_frame}', zorder=5)
    ax1.set_xlim(0, raw.frame_width)
    ax1.set_ylim(raw.frame_height, 0)
    ax1.set_xlabel('X (px)')
    ax1.set_ylabel('Y (px)')
    ax1.set_title('XY  (red×=static | lime=pts | red★=contact | cyan▲=impact)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.2)

    ax2 = axes[1]
    if all_fi:
        ax2.scatter(all_fi, all_cx, c=all_fi, cmap='plasma', s=4, alpha=0.2)
    if pt_fi:
        ax2.scatter(pt_fi, pt_cx, c='lime', s=20, alpha=0.9, zorder=3)
    if contact_frame:
        ax2.axvline(contact_frame, color='orange', linestyle='--',
                    alpha=0.8, label=f'contact f{contact_frame}')
    if contact_pt is not None:
        ax2.plot(contact_pt.frame_index, contact_pt.x, marker='*', color='red',
                  markersize=18, markeredgecolor='black', markeredgewidth=1, zorder=6)
    if impact_frame:
        ax2.axvline(impact_frame, color='cyan', linestyle='--',
                    alpha=0.8, label=f'impact f{impact_frame}')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('X (px)')
    ax2.set_title('X vs Time')
    ax2.set_xlim(0, total)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    ax3 = axes[2]
    if all_fi:
        ax3.scatter(all_fi, all_cy, c=all_fi, cmap='plasma', s=4, alpha=0.2)
    if pt_fi:
        ax3.scatter(pt_fi, pt_cy, c='lime', s=20, alpha=0.9, zorder=3)
    if contact_frame:
        ax3.axvline(contact_frame, color='orange', linestyle='--', alpha=0.8)
    if contact_pt is not None:
        ax3.plot(contact_pt.frame_index, contact_pt.y, marker='*', color='red',
                  markersize=18, markeredgecolor='black', markeredgewidth=1, zorder=6)
    if impact_frame:
        ax3.axvline(impact_frame, color='cyan', linestyle='--', alpha=0.8)
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Y (px)')
    ax3.invert_yaxis()
    ax3.set_title('Y vs Time')
    ax3.set_xlim(0, total)
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()


def _save_stage_png(
    stage_label: str,
    pts: list,
    raw: _RawData,
    fps: float,
    total: int,
    out_path: str,
    static_clusters: list | None = None,
    tstand=None,
    contact_frame: int | None = None,
    scenario: str = '',
) -> None:
    if not _HAS_MATPLOTLIB:
        print(f"[stages] matplotlib not installed — skipping {stage_label}")
        return
    title = (
        f"[{stage_label}]  pts={len(pts)}  scenario={scenario or '?'}  "
        f"raw={len(raw.all_ball_dets)}  {total}f@{fps:.0f}fps"
        + (f"  contact=f{contact_frame}" if contact_frame else "")
    )
    _build_trajectory_figure(
        title, pts, raw, total,
        static_clusters=static_clusters,
        tstand=tstand,
        contact_frame=contact_frame,
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[stages] → {out_path}")


_ROLE_COLOR = {
    "incoming": "dodgerblue",
    "flight": "lime",
    "bounce": "orange",
    "unclassified": "gray",
}
_ROLE_LABEL = {
    "incoming": "來球段",
    "flight": "飛行段",
    "bounce": "反彈段",
    "unclassified": "未分類",
}


def _save_trajfit_png(
    fit_result, raw: _RawData, fps: float, total: int, scenario: str,
    out_path: str,
) -> None:
    """TrajectoryFitter 結果視覺化：每個線段一種顏色，標示角色與交界幀。"""
    if not _HAS_MATPLOTLIB:
        print(f"[trajfit] matplotlib not installed — skipping")
        return

    all_fi = [fi for fi, _ in raw.all_ball_dets]
    all_cx = [(d.bbox[0] + d.bbox[2]) / 2 for _, d in raw.all_ball_dets]
    all_cy = [(d.bbox[1] + d.bbox[3]) / 2 for _, d in raw.all_ball_dets]
    boundary = fit_result.boundary_frames

    title = (
        f"[trajfit] segments={len(fit_result.segments)}  "
        f"discarded={len(fit_result.discarded)}  scenario={scenario}  "
        f"contact={boundary['contact']}  impact={boundary['impact']}  "
        f"{total}f@{fps:.0f}fps"
    )

    fig, axes = plt.subplots(3, 1, figsize=(16, 18))
    fig.suptitle(title, fontsize=12, fontweight='bold')

    ax1, ax2, ax3 = axes
    if all_cx:
        ax1.scatter(all_cx, all_cy, c='lightgray', s=6, alpha=0.4, label='all YOLO')
        ax2.scatter(all_fi, all_cx, c='lightgray', s=4, alpha=0.3)
        ax3.scatter(all_fi, all_cy, c='lightgray', s=4, alpha=0.3)

    if fit_result.discarded:
        dfi = [p.frame_index for p in fit_result.discarded]
        dcx = [p.x for p in fit_result.discarded]
        dcy = [p.y for p in fit_result.discarded]
        ax1.scatter(dcx, dcy, c='black', marker='x', s=30, label='discarded')
        ax2.scatter(dfi, dcx, c='black', marker='x', s=20)
        ax3.scatter(dfi, dcy, c='black', marker='x', s=20)

    for seg in fit_result.segments:
        color = _ROLE_COLOR.get(seg.role, 'gray')
        label = f"{_ROLE_LABEL.get(seg.role, seg.role)} (order={seg.order}, n={len(seg.points)})"
        pfi = [p.frame_index for p in seg.points]
        pcx = [p.x for p in seg.points]
        pcy = [p.y for p in seg.points]
        ax1.scatter(pcx, pcy, c=color, s=40, zorder=3, label=label)
        ax2.scatter(pfi, pcx, c=color, s=25, zorder=3)
        ax3.scatter(pfi, pcy, c=color, s=25, zorder=3)

    for b_label, frame in boundary.items():
        if frame is None:
            continue
        for ax in (ax2, ax3):
            ax.axvline(frame, color='red', linestyle='--', alpha=0.7,
                       label=f'{b_label} f{frame}')

    ax1.set_xlim(0, raw.frame_width)
    ax1.set_ylim(raw.frame_height, 0)
    ax1.set_xlabel('X (px)')
    ax1.set_ylabel('Y (px)')
    ax1.set_title('XY')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.2)

    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('X (px)')
    ax2.set_title('X vs Time')
    ax2.set_xlim(0, total)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)

    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Y (px)')
    ax3.invert_yaxis()
    ax3.set_title('Y vs Time')
    ax3.set_xlim(0, total)
    ax3.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[trajfit] → {out_path}")


def _run_stages(raw: _RawData, bat_positions: list, fps: float, total: int, base: str) -> None:
    def save(label: str, pts: list, **kw) -> None:
        _save_stage_png(label, pts, raw, fps, total,
                        f"{base}_{label}.png", **kw)

    # Stage 0: P25 大框過濾
    filtered = _p25_filter(raw.all_ball_dets)
    candidates = sorted([
        BallPosition(
            x=(d.bbox[0] + d.bbox[2]) / 2,
            y=(d.bbox[1] + d.bbox[3]) / 2,
            frame_index=fi,
            confidence=d.confidence,
        )
        for fi, d in filtered
    ], key=lambda p: p.frame_index)
    save("s0_p25", candidates)

    # Stage 1: 靜止叢集識別（顯示叢集但不移除點）
    static_clusters = _build_static_clusters(candidates)
    save("s1_static_identified", candidates, static_clusters=static_clusters)

    # Stage 2: 靜止點移除
    sparse = _remove_static(candidates, static_clusters)
    save("s2_static_removed", sparse, static_clusters=static_clusters)

    # Stage 3: 時序雜訊過濾
    flight_pts = _filter_noise_segments(sparse)
    save("s3_noise_filtered", flight_pts, static_clusters=static_clusters)

    # Stage 4: 情況判斷（不移除點，僅標注 scenario）
    scenario = _detect_scenario(flight_pts, static_clusters, total)
    save("s4_scenario", flight_pts, static_clusters=static_clusters,
         scenario=scenario)

    # Stage 4b: 第二輪靜止叢集移除（門檻收緊，攔截第一輪漏網的靜止殘留）
    round2_static = _build_static_clusters(
        flight_pts, max_spread=constants.CLOUD_STATIC_ROUND2_MAX_SPREAD_PX,
    )
    flight_pts = _remove_static(flight_pts, round2_static)
    save("s4b_static_round2", flight_pts, static_clusters=round2_static,
         scenario=scenario)

    # DEBUG: 暫存 flight_pts，供除錯腳本重複使用，避免每次重跑 YOLO
    import pickle
    with open(r"C:\Users\akk10\.claude\jobs\4940137d\tmp\flight_pts_debug.pkl", "wb") as _f:
        pickle.dump({"flight_pts": flight_pts, "scenario": scenario}, _f)

    # TrajectoryFitter：Sequential RANSAC 遞迴擬合（取代 Stage5+impact 的新方案）
    # 直接吃 Stage 3 輸出的 flight_pts，不依賴 contact_frame
    fit_result = fit_trajectory_segments(flight_pts, scenario)
    _save_trajfit_png(fit_result, raw, fps, total, scenario,
                       f"{base}_s_trajfit.png")
    for seg in fit_result.segments:
        print(f"[trajfit] {seg.role:12s} order={seg.order} "
              f"n={len(seg.points)} f{seg.start_frame}-f{seg.end_frame}")
    print(f"[trajfit] discarded={len(fit_result.discarded)} "
          f"boundary={fit_result.boundary_frames}")

    # Stage 6（前移）: T座身份辨識 + 接觸幀判斷（見 PHYSICS_VALIDATION_PRINCIPLES.md
    # B2/C2/C4/C5：轉折點/T座位置本身即為擊球點，球棒不參與此數值決定）
    flight_seg = next((s for s in fit_result.segments if s.role == "flight"), None)

    # 單幀異常跳點誤切飛行段的合併修復——與 CloudBallTracker.process()
    # 生產路徑同步（見與使用者討論 2026-07-28/29），須在 T 座身份辨識
    # 與後續 contact/impact 判斷之前執行，讓診斷路徑跟生產路徑用同一份
    # 已合併的 flight_seg。
    _merge_flight_extension_segment(fit_result, flight_seg)

    tstand = _identify_tstand(static_clusters, bat_positions, flight_seg, total)
    if scenario == "tstand" and tstand is not None and flight_seg is not None:
        contact = _solve_tstand_contact(flight_seg, tstand)
        if contact is None:
            contact = tstand.end_frame
    else:
        contact = fit_result.boundary_frames["contact"]
    save("s6_contact", flight_pts, static_clusters=static_clusters,
         tstand=tstand, contact_frame=contact, scenario=scenario)

    # 截斷（撞網偵測）：先於 MAD，移除滾動段避免汙染中位數
    post_pts = ([p for p in flight_pts if p.frame_index > contact]
                if contact is not None else flight_pts)
    flight_only, impact_frame = _truncate_at_impact(post_pts)
    impact_tag = f"f{impact_frame}" if impact_frame else "none"
    save(f"s7_truncated_impact={impact_tag}", flight_only,
         static_clusters=static_clusters, tstand=tstand,
         contact_frame=contact, scenario=scenario)

    # Stage 5: MAD 離群值移除（輸入已是純飛行段）
    clean_pts = _filter_axis_outliers(flight_only, scenario)
    save("s5_mad_filtered", clean_pts, static_clusters=static_clusters,
         tstand=tstand, contact_frame=contact, scenario=scenario)

    print(f"[stages] all stage PNGs saved to: {Path(base).parent}")


def _find_impact_xy(
    impact_frame: int, all_ball_dets: list
) -> tuple[float, float] | None:
    """從 raw YOLO 偵測中找最接近 impact_frame 的球中心座標。"""
    det_map = {fi: det for fi, det in all_ball_dets}
    for delta in range(6):
        for fi in (impact_frame + delta, impact_frame - delta):
            if fi in det_map:
                d = det_map[fi]
                return (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
    return None


# ══════════════════════════════════════════════════════════════════════
# 輸出 1：點雲圖（PNG）
# ══════════════════════════════════════════════════════════════════════

def _save_pointcloud(
    raw: _RawData,
    result: CloudTrackResult,
    fps: float,
    total: int,
    out_path: str,
) -> None:
    if not _HAS_MATPLOTLIB:
        print("[cloud] matplotlib not installed — skipping plot")
        return
    impact_tag = f"f{result.impact_frame}" if result.impact_frame else "none"
    title = (
        f"CloudBallTracker  ({len(raw.all_ball_dets)} raw dets, "
        f"{total} frames @ {fps:.0f}fps)\n"
        f"scenario={result.scenario}  "
        f"static={len(result.static_blacklist)}  "
        f"clean={len(result.positions)}  "
        f"contact=f{result.contact_frame}  "
        f"impact={impact_tag}"
    )
    impact_xy = (
        _find_impact_xy(result.impact_frame, raw.all_ball_dets)
        if result.impact_frame else None
    )
    _build_trajectory_figure(
        title, result.positions, raw, total,
        static_clusters=result.static_blacklist,
        tstand=result.tstand_cluster,
        contact_frame=result.contact_frame,
        impact_frame=result.impact_frame,
        impact_xy=impact_xy,
    )
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[cloud] pointcloud PNG → {out_path}")


# ══════════════════════════════════════════════════════════════════════
# 輸出 2：標注影片
# ══════════════════════════════════════════════════════════════════════

def _render_video(
    video_path: str,
    raw: _RawData,
    result: CloudTrackResult,
    phys: BallPhysicsResult,
    fps: float,
    out_path: str,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[cloud] Cannot reopen video: {video_path}")
        return

    w, h = raw.frame_width, raw.frame_height
    writer = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*'mp4v'), round(fps), (w, h)
    )

    clean_map  = {p.frame_index: p for p in result.positions}
    raw_map    = {}
    for fi, det in raw.all_ball_dets:
        raw_map.setdefault(fi, []).append(det)

    trail: list[tuple[int, int]] = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    font  = cv2.FONT_HERSHEY_SIMPLEX

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        out = frame.copy()

        # 所有 raw YOLO ball bbox（細框）
        for det in raw_map.get(fi, []):
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 180, 255), 1)

        # 靜止黑名單（常駐 ×）
        for sc in result.static_blacklist:
            cv2.drawMarker(out, (int(sc.cx), int(sc.cy)), _C_STATIC,
                           cv2.MARKER_CROSS, 22, 2)

        # T 座球（常駐 ★）
        if result.tstand_cluster:
            ts = result.tstand_cluster
            cv2.drawMarker(out, (int(ts.cx), int(ts.cy)), _C_TSTAND,
                           cv2.MARKER_STAR, 24, 2)

        # Cloud 追蹤點
        if fi in clean_map:
            cx, cy = int(clean_map[fi].x), int(clean_map[fi].y)
            cv2.circle(out, (cx, cy), 8, _C_CLOUD, -1)
            cv2.circle(out, (cx, cy), 9, (0, 0, 0), 1)

        # 完整飛行軌跡線（跟 cloud.png 散點圖同一個資料來源 result.positions，
        # 含契合點/反彈點；物理計算用的速度視窗 phys.velocity_post 只用於
        # LA/EV 數值運算，不影響這裡的視覺化畫線——見與使用者討論 2026-07-24）
        if fi in clean_map:
            trail.append((int(clean_map[fi].x), int(clean_map[fi].y)))
        for k in range(1, len(trail)):
            cv2.line(out, trail[k - 1], trail[k], _C_CLOUD, 2)

        # 接觸幀高亮
        if fi == result.contact_frame:
            cv2.rectangle(out, (4, 4), (w - 4, h - 4), _C_CONTACT, 4)
            cv2.putText(out, "CONTACT [Cloud]", (20, 60), font, 1.2, _C_CONTACT, 2)

        # HUD
        cv2.putText(out, f"F:{fi:04d}", (12, 28), font, 0.55, (180, 180, 180), 1)
        info = (
            f"[Cloud] {len(result.positions)}pts  "
            f"scenario:{result.scenario}  "
            + (f"LA:{phys.launch_angle:.1f}d" if phys.launch_angle else "LA:N/A") + "  "
            + (f"EV:{phys.exit_velocity:.0f}km/h" if phys.exit_velocity else "EV:N/A")
        )
        (tw, th), bl = cv2.getTextSize(info, font, 0.60, 2)
        y0 = h - 18
        cv2.rectangle(out, (10, y0 - th - 4), (10 + tw + 6, y0 + bl + 2), (0, 0, 0), -1)
        cv2.putText(out, info, (13, y0), font, 0.60, _C_CLOUD, 2)

        writer.write(out)

        if fi % 500 == 0:
            print(f"  [render] f{fi}/{total}")

    cap.release()
    writer.release()
    print(f"[cloud] video → {out_path}")


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════

def run(video_path: str, conf: float, no_trunc: bool = False,
        stages: bool = False) -> None:
    print(f"[cloud] Model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[cloud] {width}x{height} @ {fps:.1f}fps  {total} frames")

    constants.apply_fps_scaling(fps)
    constants.YOLO_CONFIDENCE_THRESHOLD = conf

    seed_frames = []
    for _ in range(min(10, total)):
        ret, f = cap.read()
        if ret:
            seed_frames.append(f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    crop_roi = detect_content_roi(seed_frames) if seed_frames else None

    # Phase 1
    print("\n[cloud] Phase 1: YOLO inference…")
    raw = _collect_raw(model, cap, total, crop_roi)
    cap.release()
    print(f"[cloud] collected {len(raw.all_ball_dets)} ball dets "
          f"| bat={len(raw.bat_head_dets)} "
          f"| hp={'LOCKED' if raw.hp_locked_bbox else 'NOT FOUND'}")

    # Phase 1b: BatCloudTracker（清洗過的球棒軌跡，供 T座身份辨識 + 接觸幀
    # 信心度佐證使用，見 PHYSICS_VALIDATION_PRINCIPLES.md B2「球棒的角色」）
    bat_positions = _build_bat_positions(raw, fps)
    print(f"[cloud] bat trail (cleaned): {len(bat_positions)} pts")

    # --stages モード：逐 Stage 輸出 PNG，不跑完整分析
    if stages:
        print("\n[cloud] Stages mode: generating per-stage PNGs…")
        base = str(Path(video_path).with_suffix(""))
        _run_stages(raw, bat_positions, fps, total, base)
        return

    # Phase 2: CloudBallTracker
    print("\n[cloud] Phase 2: CloudBallTracker…")
    result = CloudBallTracker().process(
        raw.all_ball_dets, BatPositionsInput.cleaned(bat_positions), total,
        skip_truncation=no_trunc,
    )

    # Phase 2b: Physics
    print("\n[cloud] Phase 2b: Physics…")
    phys = _compute_physics(
        result, raw.bat_head_dets, raw.barrel_pos,
        raw.hp_locked_bbox, fps, "Cloud",
    )

    # 摘要
    print("\n" + "=" * 55)
    print("  CLOUD TRACKER RESULT")
    print("=" * 55)
    rows = [
        ("Raw YOLO ball detections",      len(raw.all_ball_dets)),
        ("After P25 filter",              "—"),
        ("Static clusters (blacklist)",   len(result.static_blacklist)),
        ("T-stand cluster found",         "YES" if result.tstand_cluster else "NO"),
        ("Scenario",                      result.scenario),
        ("Clean trajectory pts",          len(result.positions)),
        ("Contact frame (cloud)",         result.contact_frame),
        ("Bat confidence (diagnostic)",   result.contact_bat_confidence),
        ("Impact frame (truncation)",     result.impact_frame if result.impact_frame else "none"),
        ("Contact frame (physics)",       phys.contact),
        ("Post-contact pts",              len(phys.velocity_post)),
        ("Launch Angle",
         f"{phys.launch_angle:.2f}°" if phys.launch_angle is not None else "N/A"),
        ("Exit Velocity",
         f"{phys.exit_velocity:.1f} km/h" if phys.exit_velocity is not None else "N/A"),
    ]
    for name, val in rows:
        print(f"  {name:<38} {str(val):>12}")
    print("=" * 55)

    # Phase 3: 輸出
    base = str(Path(video_path).with_suffix(""))
    suffix = "_cloud_no_trunc" if no_trunc else "_cloud"
    _save_pointcloud(raw, result, fps, total, base + suffix + ".png")

    print("\n[cloud] Phase 3: Rendering video…")
    _render_video(video_path, raw, result, phys, fps, base + suffix + ".mp4")

    print(f"\n[cloud] Done.")
    print(f"  Plot  : {base}{suffix}.png")
    print(f"  Video : {base}{suffix}.mp4")


def _parse_args():
    parser = argparse.ArgumentParser(description="CloudBallTracker test")
    parser.add_argument("video_path")
    parser.add_argument("--conf", type=float,
                        default=constants.YOLO_CONFIDENCE_THRESHOLD)
    parser.add_argument("--no-trunc", action="store_true",
                        help="跳過飛行截斷（Stage 1-5 only），輸出 _no_trunc.png")
    parser.add_argument("--stages", action="store_true",
                        help="每個 Stage 各輸出一張 PNG，不產生影片")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.video_path, args.conf, no_trunc=args.no_trunc, stages=args.stages)
