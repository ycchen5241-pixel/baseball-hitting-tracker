# geometry/homography.py
# 本壘板角點提取與 Homography 矩陣計算（像素 → 公制 cm）

import itertools
import math

import cv2
import numpy as np

from config import constants


# 本壘板 4 個角點的標準公制座標（cm）
# 使用前端左、前端右、後端右、後端左的順序
HOME_PLATE_REAL_COORDS = np.array([
    [0.0, 0.0],
    [constants.HOME_PLATE_FRONT_WIDTH_CM, 0.0],
    [constants.HOME_PLATE_FRONT_WIDTH_CM, constants.HOME_PLATE_SIDE_LENGTH_CM],
    [0.0, constants.HOME_PLATE_SIDE_LENGTH_CM],
], dtype=np.float32)

# 本壘板五邊形後尖端深度：sqrt(斜邊² - 半寬²)
# 30.48² - (43.18/2)² = 929.03 - 466.13 = 462.90 → sqrt ≈ 21.51 cm
HOME_PLATE_TIP_DEPTH_CM: float = math.sqrt(
    constants.HOME_PLATE_REAR_DIAGONAL_CM ** 2
    - (constants.HOME_PLATE_FRONT_WIDTH_CM / 2.0) ** 2
)

# 本壘板五邊形完整 5 個角點的標準公制座標（cm）
# 順序：前左、前右、後右、後尖端、後左
HOME_PLATE_REAL_COORDS_5PT = np.array([
    [0.0, 0.0],
    [constants.HOME_PLATE_FRONT_WIDTH_CM, 0.0],
    [constants.HOME_PLATE_FRONT_WIDTH_CM, constants.HOME_PLATE_SIDE_LENGTH_CM],
    [constants.HOME_PLATE_FRONT_WIDTH_CM / 2.0,
     constants.HOME_PLATE_SIDE_LENGTH_CM + HOME_PLATE_TIP_DEPTH_CM],
    [0.0, constants.HOME_PLATE_SIDE_LENGTH_CM],
], dtype=np.float32)


def _quad_area(pts: np.ndarray) -> float:
    """Shoelace 公式計算四邊形面積。

    Args:
        pts: shape (4, 2) 的頂點陣列，需已排序為一致的繞行方向。

    Returns:
        四邊形面積（pixels²）。
    """
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(
        float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    )


def _select_best_4_corners(pts: np.ndarray, sort_fn=None) -> np.ndarray:
    """從 N 個多邊形頂點中選出面積最大的 4 點組合。

    面積最大代表選到「最外框」的 4 個角點，可自然排除本壘板五邊形
    的後方尖角。對 N=4 時直接排序回傳，不做組合搜索。

    Args:
        pts: shape (N, 2) 的頂點陣列，N >= 4。
        sort_fn: 4 角點排序函式，預設 sort_corners。

    Returns:
        shape (4, 2) 的最佳角點陣列（已排序）。
    """
    if sort_fn is None:
        sort_fn = sort_corners

    if len(pts) == 4:
        return sort_fn(pts.astype(np.float32))

    best_corners: np.ndarray | None = None
    best_area = -1.0
    for indices in itertools.combinations(range(len(pts)), 4):
        candidate = pts[list(indices)].astype(np.float32)
        ordered = sort_fn(candidate)
        area = _quad_area(ordered)
        if area > best_area:
            best_area = area
            best_corners = ordered
    return best_corners  # type: ignore[return-value]  # N>=5 時一定有值


def sort_corners(corners: np.ndarray) -> np.ndarray:
    """將 4 個角點排序為 [前左, 前右, 後右, 後左] 順序（側面視角）。

    側面拍攝：image-x 對應本壘板深度（大 x = 前端/投手側，小 x = 後端/捕手側）。
    image-y 對應本壘板寬度（小 y = 近攝影機側/板左，大 y = 遠攝影機側/板右）。

    Args:
        corners: shape (4, 2) 的角點座標陣列。

    Returns:
        排序後 shape (4, 2) 的角點陣列：[前左 FL, 前右 FR, 後右 BR, 後左 BL]。
    """
    x_order = np.argsort(corners[:, 0])
    front_indices = x_order[2:]   # image-x 最大的 2 點 = 前端（投手側）
    rear_indices  = x_order[:2]   # image-x 最小的 2 點 = 後端（捕手側）

    front = corners[front_indices]
    front_sorted = front[np.argsort(front[:, 1])]   # y 升序: [FL, FR]

    rear = corners[rear_indices]
    rear_sorted = rear[np.argsort(rear[:, 1])]      # y 升序: [BL, BR]

    return np.array(
        [front_sorted[0], front_sorted[1], rear_sorted[1], rear_sorted[0]],
        dtype=np.float32,
    )


def sort_pentagon_corners(pts: np.ndarray) -> np.ndarray:
    """將 5 個角點排序為 [前左, 前右, 後右, 後尖端, 後左]。

    窮舉測試 5 次：每次假設其中一個點為後尖端（TIP），其餘 4 點
    以 sort_corners 排序為四邊形並計算 4 點 Homography，再將 TIP
    候選點投影至公制座標，選出投影結果最接近標準尖端位置的排列。

    此方式以獨立的 4 點 H 驗證第 5 點（非循環評分），對排列
    辨別力遠優於以同一批 5 點計算 H 後再量測邊長的方式。

    Args:
        pts: shape (5, 2) 的角點陣列（任意順序）。

    Returns:
        shape (5, 2) 的排序後角點陣列，對應 HOME_PLATE_REAL_COORDS_5PT 順序：
        [前左 FL, 前右 FR, 後右 BR, 後尖端 TIP, 後左 BL]。
    """
    pts_f = pts.astype(np.float32)
    tip_x = float(HOME_PLATE_REAL_COORDS_5PT[3, 0])
    tip_y = float(HOME_PLATE_REAL_COORDS_5PT[3, 1])

    best_score = float('inf')
    best_arr = pts_f

    for tip_idx in range(5):
        other_pts = pts_f[[i for i in range(5) if i != tip_idx]]
        tip_pt = pts_f[tip_idx]

        quad = sort_corners(other_pts)   # [FL, FR, BR, BL]
        h, _ = cv2.findHomography(quad, HOME_PLATE_REAL_COORDS, 0)
        if h is None:
            continue

        cx, cy = pixels_to_metric((float(tip_pt[0]), float(tip_pt[1])), h)
        score = (cx - tip_x) ** 2 + (cy - tip_y) ** 2

        if score < best_score:
            best_score = score
            best_arr = np.array(
                [quad[0], quad[1], quad[2], tip_pt, quad[3]],
                dtype=np.float32,
            )

    return best_arr


def sort_corners_polar(corners: np.ndarray) -> np.ndarray:
    """4 角點排序（polar sort 版）：不依賴 image-x = 深度方向假設。

    以極角排序獲得穩定循環序，用邊長和辨識前後邊對（前+後 ≈ 2× 側邊），
    再以各邊中點 image-x 確認哪端朝向投手（大 x）；左右以 image-y 區分。
    """
    pts = corners.astype(np.float32)
    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    cyclic = pts[np.argsort(angles)]

    def _elen(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    e = [_elen(cyclic[i], cyclic[(i + 1) % 4]) for i in range(4)]

    if e[0] + e[2] >= e[1] + e[3]:
        mid_a = (cyclic[0] + cyclic[1]) / 2.0
        mid_b = (cyclic[2] + cyclic[3]) / 2.0
        front, back = (
            (cyclic[[0, 1]], cyclic[[2, 3]]) if mid_a[0] >= mid_b[0]
            else (cyclic[[2, 3]], cyclic[[0, 1]])
        )
    else:
        mid_a = (cyclic[1] + cyclic[2]) / 2.0
        mid_b = (cyclic[3] + cyclic[0]) / 2.0
        front, back = (
            (cyclic[[1, 2]], cyclic[[3, 0]]) if mid_a[0] >= mid_b[0]
            else (cyclic[[3, 0]], cyclic[[1, 2]])
        )

    fl, fr = front[np.argsort(front[:, 1])]
    bl, br = back[np.argsort(back[:, 1])]
    return np.array([fl, fr, br, bl], dtype=np.float32)


def sort_pentagon_corners_polar(pts: np.ndarray) -> np.ndarray:
    """5 角點排序（polar sort 版）：5 TIP × 4 循環旋轉共 20 組合，以 H 投影評分選最佳。

    極角排序確保循環序對 ±1–2px 抖動穩定；窮舉旋轉消除 sort_corners image-x 依賴。
    """
    pts_f = pts.astype(np.float32)
    tip_x = float(HOME_PLATE_REAL_COORDS_5PT[3, 0])
    tip_y = float(HOME_PLATE_REAL_COORDS_5PT[3, 1])

    centroid = pts_f.mean(axis=0)
    angles = np.arctan2(pts_f[:, 1] - centroid[1], pts_f[:, 0] - centroid[0])
    cyclic = pts_f[np.argsort(angles)]

    best_score = float('inf')
    best_arr = pts_f

    for tip_i in range(5):
        tip_pt = cyclic[tip_i]
        others = np.array([cyclic[j] for j in range(5) if j != tip_i], dtype=np.float32)
        for rot in range(4):
            quad = np.roll(others, rot, axis=0)
            h, _ = cv2.findHomography(quad, HOME_PLATE_REAL_COORDS, 0)
            if h is None:
                continue
            cx, cy = pixels_to_metric((float(tip_pt[0]), float(tip_pt[1])), h)
            score = (cx - tip_x) ** 2 + (cy - tip_y) ** 2
            if score < best_score:
                best_score = score
                best_arr = np.array(
                    [quad[0], quad[1], quad[2], tip_pt, quad[3]], dtype=np.float32
                )

    return best_arr


def _select_best_5_corners(pts: np.ndarray, sort_fn=None) -> np.ndarray | None:
    """從 N 個多邊形頂點中選出面積最大的 5 點組合（五邊形）。

    N=5 時直接排序回傳；N>5 時以 Shoelace 面積評分選出最佳 5 點。

    Args:
        pts: shape (N, 2) 的頂點陣列，N >= 5。
        sort_fn: 5 角點排序函式，預設 sort_pentagon_corners。

    Returns:
        shape (5, 2) 的最佳角點陣列（已排序），或 N < 5 時回傳 None。
    """
    if sort_fn is None:
        sort_fn = sort_pentagon_corners

    if len(pts) < 5:
        return None
    if len(pts) == 5:
        return sort_fn(pts.astype(np.float32))

    best_corners: np.ndarray | None = None
    best_area = -1.0
    for indices in itertools.combinations(range(len(pts)), 5):
        candidate = pts[list(indices)].astype(np.float32)
        ordered = sort_fn(candidate)
        area = _quad_area(ordered)
        if area > best_area:
            best_area = area
            best_corners = ordered
    return best_corners


def _otsu_canny_thresh(blurred: np.ndarray) -> float:
    """從梯度幅值的 Otsu 閾值自動決定 Canny 高閾值。

    使用 Sobel 梯度幅值代替像素灰度值執行 Otsu，能反映真實邊緣強度分布，
    在本壘板因光線、土塵或草地而對比度變化時仍能自適應調整。

    Args:
        blurred: 已高斯模糊的灰階影像。

    Returns:
        Canny 高閾值（float），至少為 HOME_PLATE_CANNY_MIN_HIGH_THRESH。
    """
    gx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.clip(np.sqrt(gx ** 2 + gy ** 2), 0, 255).astype(np.uint8)
    high, _ = cv2.threshold(mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return max(float(high), constants.HOME_PLATE_CANNY_MIN_HIGH_THRESH)


def _passes_contour_area_filter(
    contour: np.ndarray, roi_w: int, roi_h: int,
) -> bool:
    """面積過濾：輪廓面積必須超過 ROI 面積 × 最小比例。"""
    min_area = roi_w * roi_h * constants.HOME_PLATE_MIN_CONTOUR_RATIO
    return cv2.contourArea(contour) >= min_area


def _try_single_contour_corners(
    contour: np.ndarray,
    roi_w: int,
    roi_h: int,
    sort_fn_4=None,
) -> np.ndarray | None:
    """嘗試從單一輪廓提取 4 個有效角點。

    執行面積過濾 → approxPolyDP → _select_best_4_corners → 退化檢查。

    Args:
        contour: OpenCV 輪廓（shape (N,1,2) 或 (N,2)）。
        roi_w: ROI 寬度（pixels），用於計算最小輪廓面積。
        roi_h: ROI 高度（pixels）。
        sort_fn_4: 4 角點排序函式，None 使用預設 sort_corners。

    Returns:
        shape (4, 2) 的角點陣列（ROI 局部座標），失敗回傳 None。
    """
    if not _passes_contour_area_filter(contour, roi_w, roi_h):
        return None

    arc_len = cv2.arcLength(contour, closed=True)
    eps = constants.HOME_PLATE_APPROX_EPSILON * arc_len
    approx = cv2.approxPolyDP(contour, eps, closed=True)

    if len(approx) < 4:
        return None

    corners = _select_best_4_corners(approx.reshape(-1, 2), sort_fn_4)
    if _quad_area(corners) < constants.HOME_PLATE_MIN_CORNER_AREA_PX:
        return None

    return corners


def _try_5corner_contour_corners(
    contour: np.ndarray,
    roi_w: int,
    roi_h: int,
    sort_fn_5=None,
) -> np.ndarray | None:
    """嘗試從單一輪廓提取 5 個五邊形角點（含後尖端）。

    使用較小的 approxPolyDP epsilon（×0.7）以保留本壘板後方的尖角頂點。

    Args:
        contour: OpenCV 輪廓（shape (N,1,2) 或 (N,2)）。
        roi_w: ROI 寬度（pixels）。
        roi_h: ROI 高度（pixels）。
        sort_fn_5: 5 角點排序函式，None 使用預設 sort_pentagon_corners。

    Returns:
        shape (5, 2) 的角點陣列（ROI 局部座標），失敗回傳 None。
    """
    if not _passes_contour_area_filter(contour, roi_w, roi_h):
        return None

    arc_len = cv2.arcLength(contour, closed=True)
    eps = (
        constants.HOME_PLATE_APPROX_EPSILON
        * constants.FIVE_CORNER_EPSILON_FACTOR
        * arc_len
    )
    approx = cv2.approxPolyDP(contour, eps, closed=True)
    pts = approx.reshape(-1, 2)

    if len(pts) < 5:
        return None

    corners = _select_best_5_corners(pts, sort_fn_5)
    if corners is None:
        return None
    if _quad_area(corners) < constants.HOME_PLATE_MIN_CORNER_AREA_PX:
        return None

    return corners


def _try_merged_contours_corners(
    contours: list,
    roi_w: int,
    roi_h: int,
    sort_fn_4=None,
    sort_fn_5=None,
) -> np.ndarray | None:
    """策略 B：合併前 N 大輪廓取凸包後提取角點（遮擋導致輪廓破碎時的保底）。

    Args:
        contours: 所有已偵測輪廓列表。
        roi_w: ROI 寬度（pixels）。
        roi_h: ROI 高度（pixels）。
        sort_fn_4: 4 角點排序函式。
        sort_fn_5: 5 角點排序函式。

    Returns:
        shape (5, 2) 或 (4, 2) 的角點陣列；失敗回傳 None。
    """
    if len(contours) < 2:
        return None
    top_n = sorted(contours, key=cv2.contourArea, reverse=True)[
        :constants.HOME_PLATE_MULTI_CONTOUR_TOP_N
    ]
    merged = np.concatenate([c.reshape(-1, 2) for c in top_n])
    hull = cv2.convexHull(merged.reshape(-1, 1, 2))  # pylint: disable=too-many-function-args
    corners = _try_5corner_contour_corners(hull, roi_w, roi_h, sort_fn_5)
    if corners is not None:
        return corners
    return _try_single_contour_corners(hull, roi_w, roi_h, sort_fn_4)


def _try_corners_from_edges(
    edges: np.ndarray,
    roi_w: int,
    roi_h: int,
    sort_fn_4=None,
    sort_fn_5=None,
) -> np.ndarray | None:
    """從 Canny 邊緣圖嘗試提取角點，優先 5 角點，退回 4 角點。

    策略 A1：最大輪廓 → 5 角點（五邊形）。
    策略 A2：最大輪廓 → 4 角點（矩形近似）。
    策略 B：合併前 N 大輪廓取凸包再提取（輪廓破碎時的保底）。

    Args:
        edges: Canny 邊緣二值圖（ROI 空間）。
        roi_w: ROI 寬度（pixels）。
        roi_h: ROI 高度（pixels）。
        sort_fn_4: 4 角點排序函式。
        sort_fn_5: 5 角點排序函式。

    Returns:
        shape (5, 2) 或 (4, 2) 的角點陣列（ROI 局部座標），均失敗回傳 None。
    """
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    corners = _try_5corner_contour_corners(largest, roi_w, roi_h, sort_fn_5)
    if corners is not None:
        return corners

    corners = _try_single_contour_corners(largest, roi_w, roi_h, sort_fn_4)
    if corners is not None:
        return corners

    return _try_merged_contours_corners(contours, roi_w, roi_h, sort_fn_4, sort_fn_5)


def _auto_canny_with_retries(
    blurred: np.ndarray,
    roi_w: int,
    roi_h: int,
    sort_fn_4=None,
    sort_fn_5=None,
) -> np.ndarray | None:
    """階梯式降低 Canny 閾值重試，直到找到有效角點或達重試上限。

    初次使用 Otsu 最優高閾值；每次重試乘以 RETRY_FACTOR 降低閾值，
    使邊緣偵測更寬鬆，有助於應對遮擋或低對比度場景。

    Args:
        blurred: 已做高斯模糊的灰階 ROI 影像。
        roi_w: ROI 寬度（pixels）。
        roi_h: ROI 高度（pixels）。
        sort_fn_4: 4 角點排序函式。
        sort_fn_5: 5 角點排序函式。

    Returns:
        shape (4, 2) 或 (5, 2) 的角點陣列（ROI 相對座標）；全部失敗則 None。
    """
    high_thresh = _otsu_canny_thresh(blurred)
    for retry in range(constants.HOME_PLATE_CANNY_MAX_RETRIES + 1):
        t = high_thresh * (constants.HOME_PLATE_CANNY_RETRY_FACTOR ** retry)
        low = t * constants.HOME_PLATE_CANNY_LOW_HIGH_RATIO
        edges = cv2.Canny(blurred, int(low), int(t))
        corners = _try_corners_from_edges(edges, roi_w, roi_h, sort_fn_4, sort_fn_5)
        if corners is not None:
            return corners
    return None


def extract_home_plate_corners(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    min_roi_height: int | None = None,
    polar: bool = False,
) -> np.ndarray:
    """從本壘板 bbox 區域提取角點（像素座標）。

    優先嘗試提取 5 個角點（五邊形含後尖端），退回 4 個角點（矩形近似）。
    流程：裁切 → 灰階 → 高斯模糊 → Auto-Canny（梯度 Otsu）→ 角點提取。
    當固定閾值 Canny 失敗時透過階梯式降低閾值重試，並在單一輪廓不足時
    自動合併多輪廓補救破碎輪廓。

    Args:
        frame: BGR 影像陣列。
        bbox: (x1, y1, x2, y2) home_plate bounding box。

    Returns:
        shape (5, 2) 或 (4, 2) 的角點像素座標陣列（絕對座標）。
        5 點：五邊形完整形狀 [前左, 前右, 後右, 後尖端, 後左]。
        4 點：矩形近似 [左上, 右上, 右下, 左下]。

    Raises:
        ValueError: 所有重試策略均無法找到有效角點。
    """
    x1, y1, x2, y2 = (int(v) for v in bbox)
    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        raise ValueError(f"Empty ROI for home_plate bbox {bbox}")

    roi_h, roi_w = roi.shape[:2]

    _min_h = (
        min_roi_height if min_roi_height is not None
        else constants.HOME_PLATE_CANNY_MIN_ROI_HEIGHT_PX
    )
    if roi_h < _min_h:
        raise ValueError(
            f"ROI too short for reliable corner detection "
            f"({roi_w}x{roi_h} < min height {_min_h}px)"
        )

    k = constants.HOME_PLATE_BLUR_KERNEL
    blurred = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (k, k), 0)
    offset = np.array([x1, y1], dtype=np.float32)

    sort_fn_4 = sort_corners_polar if polar else None
    sort_fn_5 = sort_pentagon_corners_polar if polar else None
    corners = _auto_canny_with_retries(blurred, roi_w, roi_h, sort_fn_4, sort_fn_5)
    if corners is None:
        high_thresh = _otsu_canny_thresh(blurred)
        raise ValueError(
            f"No valid home plate corners after "
            f"{constants.HOME_PLATE_CANNY_MAX_RETRIES + 1} threshold attempts"
            f" (Otsu high={high_thresh:.1f}, ROI {roi_w}x{roi_h})"
        )
    return corners + offset


def extract_corners_from_keypoints(keypoints: np.ndarray) -> np.ndarray:
    """從模型輸出的關鍵點直接取得本壘板 5 個角點（像素座標）。

    適用於 best_v7.pt 等關鍵點模型，直接輸出五邊形角點，精度高於 Canny。

    Args:
        keypoints: shape (N, 2) 的關鍵點像素座標陣列，N >= 5。

    Returns:
        shape (5, 2) 的角點陣列（已排序為 [前左, 前右, 後右, 後尖端, 後左]）。

    Raises:
        ValueError: 關鍵點不足或有無效（零座標）關鍵點。
    """
    if keypoints is None or len(keypoints) < 5:
        raise ValueError(
            f"Need >= 5 keypoints, got "
            f"{len(keypoints) if keypoints is not None else 0}"
        )

    pts = keypoints[:5, :2].astype(np.float32)

    # 過濾無效（零座標）關鍵點：模型未偵測到時輸出 (0, 0)
    zero_mask = np.all(pts == 0, axis=1)
    if zero_mask.any():
        raise ValueError(
            "Undetected keypoints at indices: "
            f"{np.where(zero_mask)[0].tolist()}"
        )

    return sort_pentagon_corners(pts)


def _estimate_tip_pixel(corners_4: np.ndarray) -> np.ndarray:
    """從 4 個矩形角點幾何估算後尖端的像素座標。

    原理：後尖端位於後端中點沿「前→後」方向再延伸一段距離。
    延伸比例 = HOME_PLATE_TIP_DEPTH_CM / HOME_PLATE_SIDE_LENGTH_CM
             = sqrt(30.48²-(43.18/2)²) / 21.59 ≈ 21.52/21.59 ≈ 0.9965

    Args:
        corners_4: sort_corners 排序後的 (4,2) 角點，
                   順序為 [前左, 前右, 後右, 後左]。

    Returns:
        估算後尖端的像素座標，shape (2,)。
    """
    m_front = (corners_4[0] + corners_4[1]) / 2.0   # 前端中點
    m_back  = (corners_4[2] + corners_4[3]) / 2.0   # 後端中點
    direction = m_back - m_front                     # 前→後方向向量
    ratio = HOME_PLATE_TIP_DEPTH_CM / constants.HOME_PLATE_SIDE_LENGTH_CM
    return (m_back + direction * ratio).astype(np.float32)


def validate_homography(
    h_matrix: np.ndarray | None,
    corners_pixels: np.ndarray,
    max_error_cm: float,
) -> bool:
    """驗證 Homography 是否退化：將輸入角點反投影至 cm 並比對預期真實座標。

    4 點輸入對比 HOME_PLATE_REAL_COORDS（矩形），
    5 點輸入對比 HOME_PLATE_REAL_COORDS_5PT（五邊形）。
    任一角點誤差超過 max_error_cm 即視為退化。

    Args:
        h_matrix: 3x3 Homography 矩陣；None 直接回傳 False。
        corners_pixels: shape (4,2) 或 (5,2) 的像素角點。
        max_error_cm: 每個角點允許的最大投影誤差（cm）。

    Returns:
        True 表示 Homography 有效；False 表示退化或超出誤差。
    """
    if h_matrix is None:
        return False

    if corners_pixels.shape[0] == 4:
        expected = HOME_PLATE_REAL_COORDS
    elif corners_pixels.shape[0] == 5:
        expected = HOME_PLATE_REAL_COORDS_5PT
    else:
        return False

    for (px, py), (ex, ey) in zip(corners_pixels, expected):
        cx, cy = pixels_to_metric((float(px), float(py)), h_matrix)
        if math.sqrt((cx - ex) ** 2 + (cy - ey) ** 2) > max_error_cm:
            return False
    return True


def compute_homography(corners_pixels: np.ndarray) -> np.ndarray:
    """根據 4 或 5 個像素角點計算 Homography 矩陣（像素 → 公制 cm）。

    4 個角點（CANNY4）：幾何估算後尖端像素座標，升級為 5 點 RANSAC 計算，
    與 CANNY5 使用相同的五邊形真實座標，消除矩形近似導致的退化問題。
    5 個角點（CANNY5/KPT5）：RANSAC 過度確定解，最高精度。

    兩種路徑均使用 RANSAC，避免先做全點 least-squares 再手動移除 outlier
    的循環邏輯缺陷（H_initial 被 outlier 扭曲後可能誤判正確角點為外點）。
    RANSAC 直接以 4 點子集枚舉所有可能組合，以最大 inlier 數選出最穩健的 H。

    Args:
        corners_pixels: shape (4, 2) 或 (5, 2) 的角點像素座標。
                        4 點需已由 sort_corners 排序為 [前左,前右,後右,後左]。
                        5 點需已由 sort_pentagon_corners 排序。

    Returns:
        3x3 Homography 矩陣 H。

    Raises:
        ValueError: 輸入角點數量不為 4 或 5。
    """
    thresh = constants.HOMOGRAPHY_VALIDATE_MAX_ERROR_CM

    if corners_pixels.shape == (4, 2):
        # CANNY4+1：從 4 個矩形角點估算後尖端，升級為 5 點 RANSAC
        tip = _estimate_tip_pixel(corners_pixels)
        corners_5 = np.array([
            corners_pixels[0],   # 前左 → real (0, 0)
            corners_pixels[1],   # 前右 → real (43.18, 0)
            corners_pixels[2],   # 後右 → real (43.18, 21.59)
            tip,                 # 後尖端 → real (21.59, 43.11)
            corners_pixels[3],   # 後左 → real (0, 21.59)
        ], dtype=np.float32)
        homography_matrix, _ = cv2.findHomography(
            corners_5,
            HOME_PLATE_REAL_COORDS_5PT,
            cv2.RANSAC,
            thresh,
        )
    elif corners_pixels.shape == (5, 2):
        homography_matrix, _ = cv2.findHomography(
            corners_pixels,
            HOME_PLATE_REAL_COORDS_5PT,
            cv2.RANSAC,
            thresh,
        )
    else:
        raise ValueError(
            f"Expected (4,2) or (5,2) corners, got {corners_pixels.shape}"
        )
    return homography_matrix


def bbox_corners(
    bbox: tuple[float, float, float, float],
) -> np.ndarray:
    """bbox 4 角點直接作為本壘板角點（Canny 失敗時的 fallback）。

    YOLO 偵測的 bbox 通常緊貼本壘板，四角近似板的四個頂點。
    此方法精度低於 Canny 角點提取，但對任何 bbox 均有效。

    Args:
        bbox: (x1, y1, x2, y2) home_plate bounding box。

    Returns:
        shape (4, 2) 的角點像素座標陣列：[左上, 右上, 右下, 左下]。
    """
    x1, y1, x2, y2 = bbox
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def pixels_to_metric(
    point_pixels: tuple[float, float],
    homography_matrix: np.ndarray,
) -> tuple[float, float]:
    """將單一像素座標轉換為公制座標（cm）。

    Args:
        point_pixels: (x, y) 像素座標。
        homography_matrix: 3x3 Homography 矩陣。

    Returns:
        (X, Y) 公制座標（cm）。
    """
    pt = np.array([[[point_pixels[0], point_pixels[1]]]], dtype=np.float32)
    transformed = cv2.perspectiveTransform(pt, homography_matrix)
    x_cm, y_cm = transformed[0][0]
    return float(x_cm), float(y_cm)
