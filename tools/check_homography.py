# check_homography.py
# Verify Homography scale: corner points projected to metric coords vs actual dimensions

import math
import sys
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import constants
from detector.yolo_tracker import load_model, run_inference
from tools.homography import (
    extract_home_plate_corners,
    extract_corners_from_keypoints,
    bbox_corners,
    compute_homography,
    validate_homography,
    pixels_to_metric,
    sort_corners,
    HOME_PLATE_REAL_COORDS_5PT,
)

MODEL_PATH = "Training/best_v7.pt"

# MLB standard dimensions (cm)
EXPECTED = {
    "front  (FL-FR)": constants.HOME_PLATE_FRONT_WIDTH_CM,   # 43.18
    "left   (FL-BL)": constants.HOME_PLATE_SIDE_LENGTH_CM,   # 21.59
    "right  (FR-BR)": constants.HOME_PLATE_SIDE_LENGTH_CM,   # 21.59
    "diagR  (BR-TIP)": constants.HOME_PLATE_REAR_DIAGONAL_CM, # 30.48
    "diagL  (BL-TIP)": constants.HOME_PLATE_REAR_DIAGONAL_CM, # 30.48
}

# Segment definitions for 5-point [FL, FR, BR, TIP, BL]
_SEGMENTS_5PT = [
    ("front  (FL-FR)", 0, 1),
    ("left   (FL-BL)", 0, 4),
    ("right  (FR-BR)", 1, 2),
    ("diagR  (BR-TIP)", 2, 3),
    ("diagL  (BL-TIP)", 4, 3),
]

# Segment definitions for 4-point [FL, FR, BR, BL]
_SEGMENTS_4PT = [
    ("front  (FL-FR)", 0, 1),
    ("left   (FL-BL)", 0, 3),
    ("right  (FR-BR)", 1, 2),
]


def _dist(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def _draw_corners(frame, corners_px, method):
    """Draw corner points and labels onto frame."""
    out = frame.copy()
    labels = (
        ["FL", "FR", "BR", "TIP", "BL"]
        if len(corners_px) == 5
        else ["FL", "FR", "BR", "BL"]
    )
    colors = [
        (0, 255, 0), (0, 200, 255), (255, 100, 0),
        (255, 0, 200), (200, 0, 255),
    ]
    for i, (pt, label) in enumerate(zip(corners_px, labels)):
        x, y = int(pt[0]), int(pt[1])
        color = colors[i % len(colors)]
        cv2.circle(out, (x, y), 8, color, -1)
        cv2.circle(out, (x, y), 8, (255, 255, 255), 1)
        cv2.putText(out, label, (x + 10, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    pts = corners_px.astype(np.int32)
    if len(pts) == 5:
        order = [0, 1, 2, 3, 4, 0]
    else:
        order = [0, 1, 2, 3, 0]
    for a, b in zip(order[:-1], order[1:]):
        cv2.line(out, tuple(pts[a]), tuple(pts[b]), (200, 200, 200), 1)
    cv2.putText(out, f"Method: {method}", (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 100), 2)
    return out


def check(video_path: str) -> None:
    print(f"\n[check] Loading model: {MODEL_PATH}")
    model = load_model(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    constants.apply_fps_scaling(fps)
    print(f"[check] {total} frames @ {fps:.0f} fps")

    # Scan all frames, keep best home_plate detection
    best_fi, best_conf, best_det, best_frame = -1, 0.0, None, None
    fi = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        fd = run_inference(model, frame, fi)
        hp_list = fd.get_by_class(constants.CLASS_HOME_PLATE)
        if hp_list:
            det = max(hp_list, key=lambda d: d.confidence)
            if det.confidence > best_conf:
                best_conf, best_det, best_frame, best_fi = (
                    det.confidence, det, frame.copy(), fi
                )
        fi += 1
    cap.release()

    if best_det is None:
        print("[check] No home plate detected — check model path / video content")
        return

    print(f"\n[check] Best frame: #{best_fi}  (YOLO conf={best_conf:.3f})")
    print(f"[check] bbox px: {[f'{v:.1f}' for v in best_det.bbox]}")
    bw = best_det.bbox[2] - best_det.bbox[0]
    bh = best_det.bbox[3] - best_det.bbox[1]
    print(f"[check] bbox size: {bw:.0f}x{bh:.0f} px")

    # Corner extraction (mirrors _build_homography in main.py)
    corners, method = None, None
    if best_det.keypoints is not None:
        try:
            corners = extract_corners_from_keypoints(best_det.keypoints)
            method = "KPT5"
        except ValueError as e:
            print(f"[check] Keypoints failed: {e} — trying Canny")

    if corners is None:
        try:
            corners = extract_home_plate_corners(best_frame, best_det.bbox)
            method = "CANNY5" if len(corners) == 5 else "CANNY4"
        except ValueError as e:
            print(f"[check] Canny failed: {e} — using bbox fallback")
            corners = bbox_corners(best_det.bbox)
            method = "BBOX4"

    print(f"[check] Corner method: {method}  ({len(corners)} pts)")
    labels = ["FL", "FR", "BR", "TIP", "BL"] if len(corners) == 5 else ["FL", "FR", "BR", "BL"]
    print("[check] Corner pixel coords:")
    for label, pt in zip(labels, corners):
        print(f"  {label}: ({pt[0]:.1f}, {pt[1]:.1f}) px")

    # Least-squares H for diagnostic reprojection error
    if len(corners) == 5:
        h_ls, _ = cv2.findHomography(corners, HOME_PLATE_REAL_COORDS_5PT, 0)
        if h_ls is not None:
            print("\n[check] Per-corner reprojection error (least-squares H):")
            for label, pt, (rx, ry) in zip(
                labels, corners, HOME_PLATE_REAL_COORDS_5PT
            ):
                mx, my = pixels_to_metric(
                    (float(pt[0]), float(pt[1])), h_ls
                )
                err = math.sqrt((mx - rx) ** 2 + (my - ry) ** 2)
                print(
                    f"  {label}: proj({mx:.2f}, {my:.2f}) "
                    f"real({rx:.2f}, {ry:.2f}) err={err:.2f}cm"
                )

    # Compute RANSAC homography
    try:
        h_mat = compute_homography(corners)
    except ValueError as e:
        print(f"[check] Homography failed: {e}")
        return

    valid = validate_homography(h_mat, corners, constants.HOMOGRAPHY_VALIDATE_MAX_ERROR_CM)
    print(f"\n[check] Homography validation: {'PASS' if valid else 'FAIL'}")

    # Project corners to metric
    print("[check] Corner metric coords (cm):")
    metric_pts = []
    for label, pt in zip(labels, corners):
        mx, my = pixels_to_metric((float(pt[0]), float(pt[1])), h_mat)
        metric_pts.append((mx, my))
        print(f"  {label}: ({mx:.2f}, {my:.2f}) cm")

    # Edge comparison table
    print(f"\n{'='*62}")
    print(f"  {'Segment':<20}  {'Measured':>10}  {'Expected':>10}  {'Err(cm)':>9}  {'Err%':>6}")
    print(f"{'='*62}")

    segments = _SEGMENTS_5PT if len(corners) == 5 else _SEGMENTS_4PT
    all_pass = True
    for name, i, j in segments:
        if i >= len(metric_pts) or j >= len(metric_pts):
            continue
        measured = _dist(metric_pts[i], metric_pts[j])
        expected = EXPECTED.get(name)
        if expected is not None:
            err = measured - expected
            pct = err / expected * 100
            flag = "  !" if abs(pct) > 20 else ""
            status = "OK" if abs(err) < 3.0 else "FAIL"
            if abs(err) >= 3.0:
                all_pass = False
            print(
                f"  {name:<20}  {measured:>10.2f}  {expected:>10.2f}"
                f"  {err:>+8.2f}  {pct:>+5.1f}%  {status}{flag}"
            )
        else:
            print(f"  {name:<20}  {measured:>10.2f}")

    print(f"{'='*62}")
    if all_pass:
        print("[check] Result: Homography scale OK (all errors < 3cm)")
    else:
        print("[check] Result: Significant scale errors detected")
        print("        Check plate dimension constants and corner ordering")

    # Save debug image
    debug_img = _draw_corners(best_frame, corners, method)
    out_path = video_path.rsplit(".", 1)[0] + "_hp_debug.jpg"
    cv2.imwrite(out_path, debug_img)
    print(f"\n[check] Debug image saved: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.12 tools/check_homography.py <video_path>")
        sys.exit(1)
    check(sys.argv[1])
