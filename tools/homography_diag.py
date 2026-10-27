"""Diagnose home plate homography for 793471920.220478.mp4.

Shows:
  - Detected pixel corners (raw & sorted)
  - Projected edge lengths (px → cm via H)
  - Comparison to actual plate dimensions
  - Visualised frame saved to Test_vod/
"""
import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import constants
from detector.yolo_tracker import (
    detect_content_roi, load_model, run_inference,
)
from tools.homography import (
    bbox_corners, compute_homography, extract_corners_from_keypoints,
    extract_home_plate_corners, pixels_to_metric, sort_corners,
    validate_homography,
)

VIDEO_PATH = "Test_vod/793471920.220478.mp4"
OUT_IMG    = "Test_vod/793471920_homography_diag.jpg"
TARGET_FRAME = 128          # frame where best H was found


# ── Actual home plate dimensions (MLB standard, cm) ──────────────────
ACTUAL = {
    "front  (17 in)":   constants.HOME_PLATE_FRONT_WIDTH_CM,   # 43.18
    "side L (8.5 in)":  constants.HOME_PLATE_SIDE_LENGTH_CM,   # 21.59
    "side R (8.5 in)":  constants.HOME_PLATE_SIDE_LENGTH_CM,
    "diag L (12 in)":   constants.HOME_PLATE_REAR_DIAGONAL_CM, # 30.48
    "diag R (12 in)":   constants.HOME_PLATE_REAR_DIAGONAL_CM,
}


def _edge_cm(p1_cm, p2_cm) -> float:
    return math.sqrt((p1_cm[0]-p2_cm[0])**2 + (p1_cm[1]-p2_cm[1])**2)


def _edge_px(p1, p2) -> float:
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


def main() -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    constants.apply_fps_scaling(fps)

    # ROI detection (same as main pipeline)
    seed = []
    for _ in range(10):
        ret, f = cap.read()
        if ret:
            seed.append(f)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    crop_roi = detect_content_roi(seed) if seed else None
    if crop_roi:
        fh = seed[0].shape[0]
        if (crop_roi[1] - crop_roi[0] + 1) >= fh * 0.85:
            crop_roi = None
        else:
            print(f"Content ROI: y={crop_roi[0]}-{crop_roi[1]}")

    model = load_model("Training/best_v7.pt")

    # Seek to target frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, TARGET_FRAME)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"Cannot read frame {TARGET_FRAME}")
        return

    dets = run_inference(model, frame, TARGET_FRAME, crop_roi)
    hp = dets.best_by_class(constants.CLASS_HOME_PLATE)
    if hp is None:
        print(f"No home plate detected at frame {TARGET_FRAME}")
        return

    print(f"\nHome plate YOLO bbox: {[f'{v:.1f}' for v in hp.bbox]}")
    print(f"YOLO confidence     : {hp.confidence:.3f}")
    bbox_px = hp.bbox

    # ── Try corner extraction ─────────────────────────────────────────
    method = "?"
    corners5 = None

    # 1. Keypoints from model
    if hp.keypoints is not None and len(hp.keypoints) >= 5:
        kp = hp.keypoints[:5]
        if np.any(kp > 0):
            corners5 = kp.astype(np.float32)
            method = "KEYPOINTS"

    # 2. Canny
    if corners5 is None:
        try:
            corners5 = extract_home_plate_corners(frame, bbox_px)
            method = f"CANNY ({len(corners5)} pts)"
        except ValueError as e:
            print(f"Canny failed: {e}")

    # 3. BBOX fallback
    if corners5 is None:
        corners5 = bbox_corners(bbox_px)
        method = "BBOX fallback"

    print(f"Corner method       : {method}")

    # Sort corners: [TL, TR, BR, BL] or 5-point
    if len(corners5) == 5:
        sorted4 = sort_corners(corners5[:4])   # use top-4 for homography
        tip = corners5[4]                       # back tip
        print("\n5 corners (raw px):")
        labels = ["TL", "TR", "BR", "BL", "TIP"]
        for lbl, pt in zip(labels, list(corners5)):
            print(f"  {lbl:4s}: ({pt[0]:7.1f}, {pt[1]:7.1f})")
    else:
        sorted4 = sort_corners(np.array(corners5, dtype=np.float32))
        tip = None
        print("\n4 corners (raw px):")
        for lbl, pt in zip(["TL","TR","BR","BL"], sorted4):
            print(f"  {lbl:4s}: ({pt[0]:7.1f}, {pt[1]:7.1f})")

    # Pixel edge lengths
    c = sorted4  # [TL, TR, BR, BL]
    print("\nEdge lengths in pixels:")
    print(f"  front  TL→TR : {_edge_px(c[0],c[1]):.1f} px")
    print(f"  right  TR→BR : {_edge_px(c[1],c[2]):.1f} px")
    print(f"  back   BR→BL : {_edge_px(c[2],c[3]):.1f} px")
    print(f"  left   BL→TL : {_edge_px(c[3],c[0]):.1f} px")
    if tip is not None:
        print(f"  diag   BR→TIP: {_edge_px(c[2], tip):.1f} px")
        print(f"  diag   BL→TIP: {_edge_px(c[3], tip):.1f} px")

    # Homography
    try:
        h_mat = compute_homography(sorted4)
    except ValueError as e:
        print(f"\nHomography failed: {e}")
        return

    valid = validate_homography(
        h_mat, sorted4,
        constants.HOMOGRAPHY_VALIDATE_MAX_ERROR_CM,
    )
    print(f"\nHomography validation: {'OK' if valid else 'FAIL'}")

    # Project corners to cm
    corners_cm = [pixels_to_metric((pt[0], pt[1]), h_mat) for pt in sorted4]
    print("\nCorners in cm:")
    for lbl, pt_cm in zip(["TL","TR","BR","BL"], corners_cm):
        print(f"  {lbl:4s}: ({pt_cm[0]:8.2f}, {pt_cm[1]:8.2f}) cm")

    # Edge lengths in cm
    print("\n── Edge comparison (projected vs actual) ───────────────────")
    print(f"  {'Edge':<18} {'Projected':>10} {'Actual':>10} {'Error':>10} {'Error%':>8}")
    print(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*8}")

    edges_projected = {
        "front  (17 in)":  _edge_cm(corners_cm[0], corners_cm[1]),
        "side L (8.5 in)": _edge_cm(corners_cm[3], corners_cm[0]),
        "side R (8.5 in)": _edge_cm(corners_cm[1], corners_cm[2]),
    }
    if tip is not None:
        tip_cm = pixels_to_metric((tip[0], tip[1]), h_mat)
        edges_projected["diag L (12 in)"] = _edge_cm(corners_cm[3], tip_cm)
        edges_projected["diag R (12 in)"] = _edge_cm(corners_cm[2], tip_cm)

    for name, proj in edges_projected.items():
        actual = ACTUAL.get(name, 0.0)
        err = proj - actual
        pct = err / actual * 100 if actual else 0.0
        flag = "  ← !" if abs(pct) > 20 else ""
        print(f"  {name:<18} {proj:>9.2f}  {actual:>9.2f}  "
              f"{err:>+9.2f}  {pct:>+7.1f}%{flag}")

    # ── Visualise ─────────────────────────────────────────────────────
    vis = frame.copy()
    x1, y1, x2, y2 = [int(v) for v in bbox_px]
    cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 200, 0), 1)

    pts_draw = list(sorted4)
    if tip is not None:
        pts_draw.append(tip)
    colors = [(0,255,0),(0,200,255),(0,0,255),(255,0,0),(255,0,255)]
    lbls   = ["TL","TR","BR","BL","TIP"]
    for i, pt in enumerate(pts_draw):
        ix, iy = int(pt[0]), int(pt[1])
        cv2.circle(vis, (ix, iy), 5, colors[i % len(colors)], -1)
        cv2.putText(vis, lbls[i], (ix+6, iy-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i % len(colors)], 1)

    for i in range(4):
        p1 = (int(sorted4[i][0]), int(sorted4[i][1]))
        p2 = (int(sorted4[(i+1)%4][0]), int(sorted4[(i+1)%4][1]))
        cv2.line(vis, p1, p2, (0, 255, 200), 1)
    if tip is not None:
        for side in [sorted4[2], sorted4[3]]:
            cv2.line(vis,
                     (int(side[0]), int(side[1])),
                     (int(tip[0]), int(tip[1])),
                     (200, 0, 255), 1)

    cv2.putText(vis, f"f{TARGET_FRAME}  method={method}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.imwrite(OUT_IMG, vis)
    print(f"\nSaved: {OUT_IMG}")


if __name__ == "__main__":
    main()
