# physics/trajectory.py
# 球棒偵測前處理（P25 bbox 過濾、缺幀內插）與圓弧 RANSAC 擬合工具

import math

import numpy as np

from detector.centroid import BallPosition, ball_position_from_bbox
from detector.yolo_tracker import Detection
from config import constants


def filter_dets_by_p25(
    dets: dict,
    ratio: float | None = None,
) -> dict:
    """以 P25 × 倍率自適應過濾 {frame_index: Detection} 中的異常大 bbox。

    計算所有偵測的 bbox 最大邊長第 25 百分位數作為正常尺寸基準，
    超過 基準 × ratio 的偵測視為異常（背景誤偵測、運動模糊殘影）移除。
    自動適應不同拍攝距離與解析度，不依賴固定像素值。

    Args:
        dets: {frame_index: Detection} 偵測字典。
        ratio: P25 倍率；None 時使用 constants.BAT_HEAD_BBOX_P25_RATIO。

    Returns:
        過濾後的偵測字典（不含異常大框）。
    """
    if not dets:
        return dets

    if ratio is None:
        ratio = constants.BAT_HEAD_BBOX_P25_RATIO

    sides = [
        max(det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1])
        for det in dets.values()
    ]
    sorted_sides = sorted(sides)
    p25_idx = max(0, int(len(sorted_sides) * 0.25) - 1)
    p25 = sorted_sides[p25_idx]
    threshold = p25 * ratio

    result = {
        fi: det for fi, det in dets.items()
        if max(det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1]) <= threshold
    }
    rejected = len(dets) - len(result)
    if rejected:
        print(f"[bbox-p25] P25={p25:.0f}px  threshold={threshold:.0f}px  "
              f"rejected {rejected}/{len(dets)} oversized bbox(es)")
    return result


def fill_bat_top_gaps_with_body(
    bat_head_dets: dict,
    bat_body_dets: dict,
    max_gap_frames: int | None = None,
) -> dict:
    """用時間加權線性內插補 Bat-top 的短缺幀（動態模糊造成的缺偵測）。

    只在缺口前後都有真實 Bat-top 偵測、且缺幀當下 Bat（整體棒子）也有
    偵測到時才補——Bat 偵測到只當作「球棒確實還在畫面中、合理位置」的
    佐證，不用它的 bbox 幾何去推算代理座標的位置。

    原本設計是用缺口前後真實點的位移方向，去 Bat bbox 四個角裡選最前端
    的角當代理點；實測（IMG_3294 frame 983）發現這個做法會嚴重失真：
    Bat bbox 涵蓋整支棒子（握把到棒頭），棒子指向（bbox 長軸）跟棒頭
    瞬時移動方向是兩回事，尤其在揮棒轉折點附近——選到的角跳動幅度跟
    bbox 大小成正比、不是跟真實移動距離成正比，曾經造成單幀速度暴衝到
    158px/frame（正常揮棒峰值約 20-30px/frame），完全不可靠。

    改用線性內插後，因為真正能補到的缺口幾乎都只有 1-2 幀（Bat 也偵測
    不到的缺口本來就補不了，無論用哪種方法），球棒在 1-2 幀內的位移量
    很小，線性內插已經足夠準確，不需要更複雜的 bbox 幾何推論。

    補進去的點以較低信心度標記（原始兩側信心度平均值的一半），供下游
    視需要區分真實偵測與代理點。

    Args:
        bat_head_dets: {frame_index: Detection}，Bat-top 偵測（建議已過
            P25 過濾，用乾淨的真實點當內插端點）。
        bat_body_dets: {frame_index: Detection}，Bat（整體）偵測，同一次
            YOLO 推論的另一個類別輸出，只用其「有無偵測」當佐證。
        max_gap_frames: 缺口幀數上限；None 時使用
            constants.BAT_CLOUD_GAP_FILL_MAX_FRAMES。

    Returns:
        補洞後的 {frame_index: Detection}，未補到的缺幀維持缺席。
    """
    if max_gap_frames is None:
        max_gap_frames = constants.BAT_CLOUD_GAP_FILL_MAX_FRAMES

    sorted_frames = sorted(bat_head_dets.keys())
    filled = dict(bat_head_dets)
    n_filled = 0

    for f_before, f_after in zip(sorted_frames, sorted_frames[1:]):
        gap = f_after - f_before
        if gap <= 1 or gap > max_gap_frames:
            continue

        det_before = bat_head_dets[f_before]
        det_after = bat_head_dets[f_after]
        pos_before = ball_position_from_bbox(det_before.bbox, f_before, det_before.confidence)
        pos_after = ball_position_from_bbox(det_after.bbox, f_after, det_after.confidence)
        proxy_conf = (det_before.confidence + det_after.confidence) / 2.0 * 0.5

        for fi in range(f_before + 1, f_after):
            if fi not in bat_body_dets:
                continue
            t = (fi - f_before) / gap
            px = pos_before.x + (pos_after.x - pos_before.x) * t
            py = pos_before.y + (pos_after.y - pos_before.y) * t
            half = 2.0
            filled[fi] = Detection(
                class_name=constants.CLASS_BAT_HEAD,
                bbox=(px - half, py - half, px + half, py + half),
                confidence=proxy_conf,
                frame_index=fi,
            )
            n_filled += 1

    if n_filled:
        print(f"[bat-gap-fill] filled {n_filled} Bat-top gap(s) using Bat bbox")
    return filled


def extract_bat_swing_trajectory(
    bat_head_dets: dict,
) -> list[BallPosition]:
    """提取 bat_head 全程偵測位置（原始，無過濾）。

    Args:
        bat_head_dets: {frame_index: Detection} 的偵測結果字典。

    Returns:
        按 frame_index 排序的 BallPosition 序列。
    """
    return sorted(
        [
            ball_position_from_bbox(det.bbox, fi, det.confidence)
            for fi, det in bat_head_dets.items()
        ],
        key=lambda p: p.frame_index,
    )


def _fit_circle_3pts(
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
) -> tuple[float, float, float] | None:
    """三點確定圓，回傳 (cx, cy, r)；三點共線時回傳 None。"""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-10:
        return None
    sq_a = ax ** 2 + ay ** 2
    sq_b = bx ** 2 + by ** 2
    sq_c = cx ** 2 + cy ** 2
    ux = (sq_a * (by - cy) + sq_b * (cy - ay) + sq_c * (ay - by)) / d
    uy = (sq_a * (cx - bx) + sq_b * (ax - cx) + sq_c * (bx - ax)) / d
    r = math.sqrt((ax - ux) ** 2 + (ay - uy) ** 2)
    return ux, uy, r


def _ransac_circle_fit(
    positions: list[BallPosition],
    frame_width: int,
    n_iter: int | None = None,
) -> tuple[float, float, float, float]:
    """RANSAC 圓擬合，回傳 (cx, cy, r, inlier_ratio)。"""
    if n_iter is None:
        n_iter = constants.BAT_ARC_RANSAC_ITERATIONS
    pts = np.array([[p.x, p.y] for p in positions], dtype=np.float64)
    n = len(pts)
    if n < 3:
        return 0.0, 0.0, 1.0, 0.0
    tol = frame_width * constants.BAT_ARC_TOL_FRAME_WIDTH_RATIO
    best_inliers = 0
    best_cx, best_cy, best_r = 0.0, 0.0, 1.0
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.choice(n, 3, replace=False)
        res = _fit_circle_3pts(
            (pts[idx[0], 0], pts[idx[0], 1]),
            (pts[idx[1], 0], pts[idx[1], 1]),
            (pts[idx[2], 0], pts[idx[2], 1]),
        )
        if res is None:
            continue
        cx, cy, r = res
        if r < 1.0:
            continue
        dists = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        n_in = int(np.sum(np.abs(dists - r) <= tol))
        if n_in > best_inliers:
            best_inliers = n_in
            best_cx, best_cy, best_r = cx, cy, r
    return best_cx, best_cy, best_r, best_inliers / n
