# tests/test_homography.py
# Homography 偵測工具 — 跳幀早退版
#
# 核心流程：跳幀取樣 → YOLO 找 HP（信心度門檻）→ KPT5 / Hough / 加權累積 Canny
#           首幀通過幾何驗證即鎖定 H 矩陣，最多容忍 3 次連續無偵測
#
# 使用方式：
#   py -3.12 tests/test_homography.py <video_path> [--conf 0.25]
#
# 輸出：
#   - <video_name>_homography.jpg : 鎖定幀角點疊加截圖

import argparse
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))  # pylint: disable=wrong-import-position

from config import constants  # noqa: E402
from detector.yolo_tracker import load_model, run_inference, detect_content_roi  # noqa: E402
from tools.homography import (  # noqa: E402
    extract_corners_from_keypoints,
    compute_homography,
    validate_homography,
    pixels_to_metric,
    sort_corners_polar,
    _estimate_tip_pixel,
    _otsu_canny_thresh,
    _select_best_4_corners,
    _select_best_5_corners,
)

MODEL_PATH = "Training/best_v7.pt"
EXAMPLE_VIDEO = "Test_vod/793888764.103208.mp4"

# ── 取樣策略 ──────────────────────────────────────────────
_SAMPLE_INTERVAL_SEC = 0.5    # 每 0.5 秒跳幀取一幀
_MAX_FAILURES = 5             # 連續無偵測次數上限，超過即放棄

# ── Home-plate 信心度門檻 ─────────────────────────────────
HP_MIN_CONFIDENCE = 0.5       # 低於此值視為遮擋，計入失敗次數

# ── 加權 Canny 候選點 ─────────────────────────────────────
_MAX_CANDIDATE_CENTROIDS = 8  # 聚類後最多保留幾個 centroid（C(8,5)=56，避免組合爆炸）
_CLUSTER_RADIUS_PX = 8.0      # 候選點聚類半徑（原始像素）

# ── Hough ROI ─────────────────────────────────────────────
_ROI_SCALE = 4
_RATIO_TOLERANCE = 0.20

# ── 角點顏色 / 標籤（順序：FL FR BR TIP BL）──────────────
CORNER_COLORS_5PT = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255),
]
CORNER_LABELS_5PT = ["FL", "FR", "BR", "TIP", "BL"]
CORNER_COLORS = CORNER_COLORS_5PT[:4]
CORNER_LABELS = CORNER_LABELS_5PT[:4]
CIRCLE_RADIUS = 6


# ──────────────────────────────────────────────────────
# Drawing helpers
# ──────────────────────────────────────────────────────

def _put_label(frame, text, x, y, color):
    size, baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    ty = max(y - 4, size[1] + baseline)
    cv2.rectangle(frame, (x, ty - size[1] - baseline), (x + size[0], ty + baseline), color, -1)
    cv2.putText(frame, text, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)


def draw_corners(frame, corners, h_mat):
    n = len(corners)
    colors = CORNER_COLORS_5PT[:n] if n == 5 else CORNER_COLORS
    labels = CORNER_LABELS_5PT[:n] if n == 5 else CORNER_LABELS
    if n == 5:
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(200, 200, 200), thickness=1)
    for i, (cx, cy) in enumerate(corners):
        px, py = int(cx), int(cy)
        cv2.circle(frame, (px, py), CIRCLE_RADIUS + 2, (0, 0, 0), -1)
        cv2.circle(frame, (px, py), CIRCLE_RADIUS, colors[i], -1)
        x_cm, y_cm = pixels_to_metric((cx, cy), h_mat)
        _put_label(frame, f"{labels[i]} ({x_cm:.1f},{y_cm:.1f})", px + 10, py, colors[i])
    return frame


def draw_frame_info(frame, frame_index, status):
    info = f"Frame {frame_index}  {status}"
    cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return frame


def draw_legend(frame):
    x0 = frame.shape[1] - 185
    y0, lh = 15, 20
    for i, (name, color) in enumerate(zip(CORNER_LABELS_5PT, CORNER_COLORS_5PT)):
        y = y0 + i * lh
        cv2.circle(frame, (x0 + 6, y - 4), 6, color, -1)
        cv2.putText(frame, name, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
        cv2.putText(frame, name, (x0 + 18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    return frame


# ──────────────────────────────────────────────────────
# Geometry validation
# ──────────────────────────────────────────────────────

def _ratio_valid(corners: np.ndarray) -> bool:
    if corners is None or len(corners) < 4:
        return False
    try:
        h = compute_homography(corners)
    except ValueError:
        return False
    if not validate_homography(h, corners, constants.HOMOGRAPHY_VALIDATE_MAX_ERROR_CM):
        return False
    n = len(corners)
    bl_idx = 4 if n == 5 else 3
    pts_m = [
        pixels_to_metric((float(corners[i][0]), float(corners[i][1])), h)
        for i in (0, 1, 2, bl_idx)
    ]

    def _d(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    w = constants.HOME_PLATE_FRONT_WIDTH_CM
    s = constants.HOME_PLATE_SIDE_LENGTH_CM
    t = _RATIO_TOLERANCE
    return (
        w * (1 - t) <= _d(pts_m[0], pts_m[1]) <= w * (1 + t)
        and s * (1 - t) <= _d(pts_m[0], pts_m[3]) <= s * (1 + t)
        and s * (1 - t) <= _d(pts_m[1], pts_m[2]) <= s * (1 + t)
    )


# ──────────────────────────────────────────────────────
# 5pt helper
# ──────────────────────────────────────────────────────

def _to_5pt(c: np.ndarray) -> np.ndarray:
    if len(c) == 5:
        return c.astype(np.float32)
    tip = _estimate_tip_pixel(c)
    return np.array([c[0], c[1], c[2], tip, c[3]], dtype=np.float32)


# ──────────────────────────────────────────────────────
# Fast pentagon sort（不呼叫 findHomography）
# ──────────────────────────────────────────────────────

def _fast_sort_pentagon(pts: np.ndarray) -> np.ndarray:
    """5 角點排序為 [FL, FR, BR, TIP, BL]，不呼叫 findHomography。

    對每個點嘗試作為 TIP，其餘 4 點用 sort_corners_polar 排序。
    TIP 須同時滿足兩個幾何條件：
      1. 比 BL/BR 中點更深（沿前→後方向的投影更大）
      2. 橫向位置靠近 BL/BR 的中線（TIP 在五邊形的對稱軸上）
    評分取最小橫向偏移的排列為最終結果。
    """
    pts_f = pts.astype(np.float32)
    best_score = float('inf')
    best_arr = pts_f

    for tip_idx in range(5):
        others = pts_f[[i for i in range(5) if i != tip_idx]]
        tip_pt = pts_f[tip_idx]
        quad = sort_corners_polar(others)  # [FL, FR, BR, BL]

        m_front = (quad[0] + quad[1]) / 2.0
        m_back = (quad[2] + quad[3]) / 2.0
        depth_vec = m_back - m_front
        depth_len = float(np.linalg.norm(depth_vec))
        if depth_len < 1e-6:
            continue
        depth_unit = depth_vec / depth_len

        back_proj = float(np.dot(m_back - m_front, depth_unit))
        tip_proj = float(np.dot(tip_pt - m_front, depth_unit))

        if back_proj < 1e-6 or tip_proj <= back_proj:
            continue  # TIP 必須比 BL/BR 更深

        # 橫向偏移：TIP 應在 BL/BR 中線上（偏移越小越好）
        lateral = tip_pt - (m_front + tip_proj * depth_unit)
        front_half_w = float(np.linalg.norm(quad[1] - quad[0])) / 2.0
        score = float(np.linalg.norm(lateral)) / (front_half_w + 1e-6)

        if score < best_score:
            best_score = score
            best_arr = np.array(
                [quad[0], quad[1], quad[2], tip_pt, quad[3]], dtype=np.float32
            )

    return best_arr


# ──────────────────────────────────────────────────────
# Weighted cumulative Canny corner extraction
# ──────────────────────────────────────────────────────

def _weighted_cluster_pts(pts: np.ndarray, weights: np.ndarray, radius: float) -> np.ndarray:
    used = [False] * len(pts)
    clusters = []
    for i in range(len(pts)):
        if used[i]:
            continue
        grp_pts, grp_w = [pts[i]], [weights[i]]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if not used[j] and float(np.linalg.norm(pts[j] - pts[i])) < radius:
                grp_pts.append(pts[j])
                grp_w.append(weights[j])
                used[j] = True
        total_w = float(sum(grp_w))
        centroid = np.average(grp_pts, axis=0, weights=grp_w).astype(np.float32)
        clusters.append((centroid, total_w))
    clusters.sort(key=lambda x: x[1], reverse=True)
    return np.array([c[0] for c in clusters], dtype=np.float32) if clusters else np.empty((0, 2), dtype=np.float32)


def _weighted_canny_corners(
    frame: np.ndarray, bbox: tuple
) -> tuple[np.ndarray | None, str]:
    x1, y1, x2, y2 = (int(v) for v in bbox)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, "FAIL"

    roi_h, roi_w = roi.shape[:2]
    if roi_h < constants.HOME_PLATE_CANNY_MIN_ROI_HEIGHT_PX:
        return None, "FAIL"

    k = constants.HOME_PLATE_BLUR_KERNEL
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    min_area = roi_w * roi_h * constants.HOME_PLATE_MIN_CONTOUR_RATIO
    high_thresh = _otsu_canny_thresh(blurred)

    all_pts: list[np.ndarray] = []
    all_weights: list[float] = []
    n_levels = constants.HOME_PLATE_CANNY_MAX_RETRIES + 1

    for i in range(n_levels):
        t = high_thresh * (constants.HOME_PLATE_CANNY_RETRY_FACTOR ** i)
        low = t * constants.HOME_PLATE_CANNY_LOW_HIGH_RATIO
        edges = cv2.Canny(blurred, int(low), int(t))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        weight = 1.0 / (i + 1)
        top_contours = sorted(contours, key=cv2.contourArea, reverse=True)[
            :constants.HOME_PLATE_MULTI_CONTOUR_TOP_N
        ]
        for c in top_contours:
            if cv2.contourArea(c) < min_area:
                continue
            arc = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, constants.HOME_PLATE_APPROX_EPSILON * arc, True)
            for pt in approx.reshape(-1, 2).astype(np.float32):
                all_pts.append(pt + np.array([x1, y1], dtype=np.float32))
                all_weights.append(weight)

    if len(all_pts) < 4:
        return None, "FAIL"

    centroids = _weighted_cluster_pts(
        np.array(all_pts, dtype=np.float32),
        np.array(all_weights, dtype=np.float32),
        radius=_CLUSTER_RADIUS_PX,
    )
    centroids = centroids[:_MAX_CANDIDATE_CENTROIDS]

    if len(centroids) >= 5:
        c5 = _select_best_5_corners(centroids, sort_fn=_fast_sort_pentagon)
        if c5 is not None:
            return c5, "WCANNY5"

    if len(centroids) >= 4:
        return _select_best_4_corners(centroids, sort_fn=sort_corners_polar), "WCANNY4"

    return None, "FAIL"


# ──────────────────────────────────────────────────────
# Hough Lines corner detection (secondary path)
# ──────────────────────────────────────────────────────

def _enhance_roi(frame: np.ndarray, bbox, scale: int = _ROI_SCALE):
    x1, y1, x2, y2 = (int(v) for v in bbox)
    pad = max(5, (y2 - y1) // 4)
    h, w = frame.shape[:2]
    x1p = max(0, x1 - pad)
    y1p = max(0, y1 - pad)
    x2p = min(w, x2 + pad)
    y2p = min(h, y2 + pad)
    roi = frame[y1p:y2p, x1p:x2p]
    if roi.size == 0:
        return None, x1p, y1p, scale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    up = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale),
                    interpolation=cv2.INTER_LANCZOS4)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    return clahe.apply(up), x1p, y1p, scale


def _seg_intersect(s1, s2):
    def _coeffs(x1, y1, x2, y2):
        a, b = float(y2 - y1), float(x1 - x2)
        return a, b, float(x2 * y1 - x1 * y2)
    a1, b1, c1 = _coeffs(*s1)
    a2, b2, c2 = _coeffs(*s2)
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-10:
        return None
    return (b1 * c2 - b2 * c1) / det, (a2 * c1 - a1 * c2) / det


def _cluster_pts(pts: np.ndarray, radius: float) -> np.ndarray:
    used = [False] * len(pts)
    centroids = []
    for i, p in enumerate(pts):
        if used[i]:
            continue
        grp = [p]
        used[i] = True
        for j in range(i + 1, len(pts)):
            if not used[j] and float(np.linalg.norm(pts[j] - p)) < radius:
                grp.append(pts[j])
                used[j] = True
        centroids.append(np.mean(grp, axis=0))
    return np.array(centroids, dtype=np.float32)


def _hough_corners(gray, x_off, y_off, scale):
    roi_h, roi_w = gray.shape[:2]
    edges = cv2.Canny(gray, 30, 90)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=20,
        minLineLength=roi_w // 6, maxLineGap=roi_w // 8,
    )
    if lines is None or len(lines) < 4:
        return None, "HOUGH: too few lines"
    segs = [tuple(ln[0]) for ln in lines]
    pts = []
    for i in range(len(segs)):
        for j in range(i + 1, len(segs)):
            p = _seg_intersect(segs[i], segs[j])
            if p and 0 <= p[0] <= roi_w and 0 <= p[1] <= roi_h:
                pts.append(list(p))
    if len(pts) < 4:
        return None, "HOUGH: too few intersections"
    clusters = _cluster_pts(np.array(pts, dtype=np.float32), radius=scale * 3.0)
    if len(clusters) < 4:
        return None, "HOUGH: too few clusters"
    orig = clusters / scale + np.array([x_off, y_off], dtype=np.float32)
    if len(orig) >= 5:
        c5 = _select_best_5_corners(orig, sort_fn=_fast_sort_pentagon)
        if c5 is not None:
            return c5, "HOUGH5"
    return _select_best_4_corners(orig, sort_fn=sort_corners_polar), "HOUGH4"


# ──────────────────────────────────────────────────────
# Frame-level corner extraction
# ──────────────────────────────────────────────────────

def _try_frame_corners(
    frame: np.ndarray, hp_det
) -> tuple[np.ndarray | None, str]:
    """KPT5 → Hough → Weighted Canny. Returns (corners, method)."""
    # 1. KPT5
    if hp_det.keypoints is not None:
        try:
            c = extract_corners_from_keypoints(hp_det.keypoints)
            if _ratio_valid(c):
                return _to_5pt(c), "KPT5"
        except ValueError:
            pass

    # 2. Hough (4x upscale + CLAHE)
    enhanced, x_off, y_off, scale = _enhance_roi(frame, hp_det.bbox)
    if enhanced is not None:
        hcorners, hmeth = _hough_corners(enhanced, x_off, y_off, scale)
        if hcorners is not None and _ratio_valid(hcorners[:4]):
            return _to_5pt(hcorners[:4]) if len(hcorners) < 5 else hcorners, hmeth

    # 3. Weighted cumulative Canny
    corners, method = _weighted_canny_corners(frame, hp_det.bbox)
    if corners is not None and _ratio_valid(corners[:4]):
        return _to_5pt(corners[:4]) if len(corners) < 5 else corners, method

    return None, "FAIL"


# ──────────────────────────────────────────────────────
# Seed frames helper
# ──────────────────────────────────────────────────────

def _read_seed_frames(cap: cv2.VideoCapture, n: int = 10) -> list[np.ndarray]:
    frames = []
    while len(frames) < n:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames


# ──────────────────────────────────────────────────────
# Main: jump-frame early-exit homography detection
# ──────────────────────────────────────────────────────

def run_homography_test(video_path: str, conf_threshold: float) -> None:
    """跳幀早退：首幀通過幾何驗證即鎖定，最多容忍 _MAX_FAILURES 次連續無偵測。
    輸出：_homography.jpg（鎖定幀）+ _test.mp4（所有取樣幀標註影片）。
    """
    print(f"[homography] Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[homography] {width}x{height} @ {fps:.1f}fps  ({total_frames} frames)")

    constants.apply_fps_scaling(fps)
    constants.YOLO_CONFIDENCE_THRESHOLD = conf_threshold

    seed_frames = _read_seed_frames(cap)
    crop_roi = detect_content_roi(seed_frames) if seed_frames else None
    if crop_roi:
        print(f"[homography] Content ROI (y): {crop_roi[0]} – {crop_roi[1]}")

    interval = max(1, round(fps * _SAMPLE_INTERVAL_SEC))
    out_fps = max(1.0, fps / interval)
    print(f"[homography] Sampling every {interval} frames ({_SAMPLE_INTERVAL_SEC}s)")

    # 取樣幀標註影片
    video_out_path = str(Path(video_path).with_suffix("")) + "_test.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_out_path, fourcc, out_fps, (width, height))

    method_counts: dict[str, int] = {
        "KPT5": 0, "HOUGH5": 0, "HOUGH4": 0,
        "WCANNY5": 0, "WCANNY4": 0, "FAIL": 0, "NO_HP": 0,
    }

    failure_count = 0
    locked = False
    lock_frame = -1
    lock_frame_img: np.ndarray | None = None
    stable_corners: np.ndarray | None = None
    lock_method = "FAIL"

    fi = round(fps)  # 從第 1 秒開始偵測，跳過影片開頭遮擋幀
    while fi < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        fd = run_inference(model, frame, fi, crop_roi)
        hp = fd.best_by_class(constants.CLASS_HOME_PLATE)
        out = frame.copy()

        if hp is None or hp.confidence < HP_MIN_CONFIDENCE:
            failure_count += 1
            method_counts["NO_HP"] += 1
            conf_str = f"{hp.confidence:.2f}" if hp is not None else "none"
            status = f"NO_HP conf={conf_str} failures={failure_count}/{_MAX_FAILURES}"
            print(f"[homography] frame {fi}: {status}")
            draw_frame_info(out, fi, status)
            # 若 HP 被偵測但信心度不足，仍畫出 bbox 供參考
            if hp is not None:
                x1, y1, x2, y2 = (int(v) for v in hp.bbox)
                cv2.rectangle(out, (x1, y1), (x2, y2), (80, 80, 80), 1)
                cv2.putText(out, f"HP {hp.confidence:.2f}", (x1, max(y1 - 4, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
            writer.write(out)
            if failure_count >= _MAX_FAILURES:
                print("[homography] Max consecutive failures reached. Cannot detect home plate.")
                break
            fi += interval
            continue

        # YOLO 偵測到本壘板，重置失敗計數
        failure_count = 0

        corners, method = _try_frame_corners(frame, hp)
        method_counts[method if method in method_counts else "FAIL"] += 1
        print(f"[homography] frame {fi}: HP conf={hp.confidence:.2f}  method={method}")

        # 畫 YOLO bbox
        x1, y1, x2, y2 = (int(v) for v in hp.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (200, 200, 200), 1)

        if corners is not None:
            try:
                h_tmp = compute_homography(corners)
                draw_corners(out, corners, h_tmp)
            except ValueError:
                pass
            cv2.putText(out, method, (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            draw_frame_info(out, fi, f"HP conf={hp.confidence:.2f}  {method}")
            locked = True
            lock_frame = fi
            lock_frame_img = frame.copy()
            stable_corners = corners
            lock_method = method
        else:
            cv2.putText(out, f"FAIL conf={hp.confidence:.2f}", (x1, max(y1 - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 200), 1)
            draw_frame_info(out, fi, f"HP conf={hp.confidence:.2f}  FAIL")

        draw_legend(out)
        writer.write(out)

        if locked:
            print(f"[homography] Locked! method={lock_method}  frame={lock_frame}")
            break

        fi += interval

    cap.release()
    writer.release()
    print(f"[homography] Test video saved → {video_out_path}")

    if not locked:
        print("[homography] Failed to lock homography. Exiting.")
        _print_report(method_counts, locked, lock_method, lock_frame, video_path)
        return

    try:
        h_mat = compute_homography(stable_corners)
    except ValueError as e:
        print(f"[homography] compute_homography failed: {e}")
        return

    _print_report(method_counts, locked, lock_method, lock_frame, video_path)

    annotated = lock_frame_img.copy()
    draw_corners(annotated, stable_corners, h_mat)
    draw_frame_info(annotated, lock_frame, f"LOCKED [{lock_method}]")
    draw_legend(annotated)
    out_path = str(Path(video_path).with_suffix("")) + "_homography.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"[homography] Overlay saved → {out_path}")


def _print_report(method_counts, locked, lock_method, lock_frame, video_path):
    print("\n" + "=" * 60)
    print("  HOMOGRAPHY REPORT")
    print("=" * 60)
    print(f"  Input       : {video_path}")
    print(f"  Status      : {'LOCKED' if locked else 'FAILED'}")
    print(f"  Lock method : {lock_method}")
    print(f"  Lock frame  : {lock_frame}")
    print("-" * 60)
    for k, v in method_counts.items():
        print(f"  {k:<12}: {v}")
    print("=" * 60 + "\n")


# ──────────────────────────────────────────────────────
# Raw YOLO detection mode
# ──────────────────────────────────────────────────────

# 各類別固定顏色（BGR）
_CLASS_COLORS = {
    "Ball":       (0, 255, 255),
    "Bat":        (255, 128, 0),
    "Bat-top":    (0, 128, 255),
    "Bat-buttom": (0, 64, 255),
    "Home-plate": (0, 255, 0),
}
_DEFAULT_COLOR = (200, 200, 200)


def run_raw_yolo(
    video_path: str,
    conf_threshold: float,
    start_sec: float | None = None,
    end_sec: float | None = None,
) -> None:
    """標出所有 YOLO 偵測的 bbox + 類別 + 信心度，輸出 _raw_yolo.mp4。
    指定 start_sec/end_sec 時改為逐幀處理該時間段，否則跳幀取樣。"""
    print(f"[raw_yolo] Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[raw_yolo] {width}x{height} @ {fps:.0f}fps  {total} frames")

    constants.YOLO_CONFIDENCE_THRESHOLD = conf_threshold

    seed_frames = _read_seed_frames(cap)
    crop_roi = detect_content_roi(seed_frames) if seed_frames else None
    if crop_roi:
        print(f"[raw_yolo] Content ROI (y): {crop_roi[0]} – {crop_roi[1]}")

    # 時間段逐幀 vs. 全段跳幀
    if start_sec is not None or end_sec is not None:
        fi_start = int((start_sec or 0.0) * fps)
        fi_end   = min(total, int((end_sec * fps) + 1) if end_sec is not None else total)
        interval = 1
        out_fps  = fps
        suffix   = f"_raw_yolo_{int(start_sec or 0)}s_{int(end_sec or (total/fps))}s.mp4"
        print(f"[raw_yolo] Every-frame mode: frames {fi_start}–{fi_end}  ({fi_end-fi_start} frames)")
    else:
        fi_start = 0
        fi_end   = total
        interval = max(1, round(fps * _SAMPLE_INTERVAL_SEC))
        out_fps  = max(1.0, fps / interval)
        suffix   = "_raw_yolo.mp4"

    output_path = str(Path(video_path).with_suffix("")) + suffix
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    fi = fi_start
    while fi < fi_end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            break

        fd = run_inference(model, frame, fi, crop_roi)
        out = frame.copy()

        for det in fd.detections:
            x1, y1, x2, y2 = (int(v) for v in det.bbox)
            color = _CLASS_COLORS.get(det.class_name, _DEFAULT_COLOR)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} {det.confidence:.2f}"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(out, (x1, y1 - th - bl - 2), (x1 + tw, y1), color, -1)
            cv2.putText(out, label, (x1, y1 - bl - 1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # 顯示幀號與對應時間
        time_str = f"{fi/fps:.2f}s"
        draw_frame_info(out, fi, f"{time_str}  conf≥{conf_threshold:.2f}  {len(fd.detections)} det")
        writer.write(out)

        if interval > 1 and (fi // interval + 1) % 20 == 0:
            print(f"[raw_yolo] {fi}/{fi_end} frames processed...")
        elif interval == 1 and (fi - fi_start + 1) % 30 == 0:
            print(f"[raw_yolo] {fi - fi_start + 1}/{fi_end - fi_start} frames processed...")

        fi += interval

    cap.release()
    writer.release()
    print(f"[raw_yolo] Output → {output_path}")


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseball Hitting Tracker - Homography Detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("video_path", help=f"Input video\nExample: {EXAMPLE_VIDEO}")
    parser.add_argument(
        "--conf", type=float, default=constants.YOLO_CONFIDENCE_THRESHOLD,
        help=f"YOLO confidence threshold (default: {constants.YOLO_CONFIDENCE_THRESHOLD})",
    )
    parser.add_argument(
        "--raw-yolo", action="store_true",
        help="Raw YOLO detection mode: draw all bbox + confidence, output _raw_yolo.mp4",
    )
    parser.add_argument(
        "--start-sec", type=float, default=None,
        help="(raw-yolo) 逐幀分析起始秒數，需搭配 --end-sec",
    )
    parser.add_argument(
        "--end-sec", type=float, default=None,
        help="(raw-yolo) 逐幀分析結束秒數，需搭配 --start-sec",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.raw_yolo:
        run_raw_yolo(args.video_path, args.conf, args.start_sec, args.end_sec)
    else:
        run_homography_test(args.video_path, args.conf)
