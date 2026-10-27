# detector/ball_tracker_cloud.py
# 點雲式球體軌跡過濾器
#
# 原理：收集所有幀的 YOLO ball 偵測（非 best-per-frame），
# 在 XY 空間做全局密度分析，將靜止背景球識別為密集叢集後移除，
# 剩餘稀疏點交給 TrajectoryFitter 做穩健擬合，重建飛行軌跡。
#
# 管線：
#   Stage 0  P25 大框過濾（移除異常大 bbox 誤偵測）
#   Stage 1  XY 密度聚類 → 靜止叢集識別
#   Stage 2  靜止叢集移除
#   Stage 3  時序段落過濾（移除孤立雜訊點群）
#   Stage 4  情況判斷（T座 / 投球 / 側拋）
#   TrajectoryFitter  Sequential RANSAC 遞迴擬合 + 段落角色辨識
#                      （來球段/飛行段/反彈段，取代舊版 Stage 5 MAD
#                      與 _truncate_at_impact；不依賴 contact_frame，
#                      段落邊界由擬合過程本身推導）
#   T 座球身份辨識（在 TrajectoryFitter 之後執行，只回答「哪個靜止叢集
#                      是 T 座球」；優先用飛行段軌跡線的幾何吻合度判斷，
#                      球棒軌跡佐證與時間中點退回皆降級為輔助/最後備援）
#   接觸幀判斷（見 PHYSICS_VALIDATION_PRINCIPLES.md B2/C2/C4/C5）：
#                      pitch/sidetoss 情境為來球段/飛行段擬合曲線的外插
#                      交點；T座情境為把 T座位置當已知起點併入飛行段重新
#                      擬合、比較與原曲線的一致性。球棒不參與此數值決定，
#                      僅事後附註信心度標籤（confirmed/unconfirmed/unknown）
#                      供診斷參考
#
# 輸入：list[tuple[frame_index, Detection]]（所有幀的全部 ball 偵測）
# 輸出：CloudTrackResult

import math
from dataclasses import dataclass, field

import numpy as np

from config import constants
from detector.centroid import BallPosition, BatPositionsInput
from detector.clustering import dist, make_xy_clusters
from detector.trajectory_fitter import fit_trajectory_segments, _fit_exact
from detector.yolo_tracker import Detection


@dataclass
class StaticCluster:
    cx: float
    cy: float
    start_frame: int
    end_frame: int
    n: int
    spread: float


@dataclass
class CloudTrackResult:
    positions: list           # list[BallPosition]：乾淨飛行軌跡（截斷後）
    static_blacklist: list    # list[StaticCluster]：靜止背景球
    tstand_cluster: object    # StaticCluster | None：T 座球（若識別到）
    contact_frame: int | None
    scenario: str             # 'tstand' / 'pitch' / 'sidetoss' / 'unknown'
    impact_frame: int | None = None  # 撞擊點幀（反彈截斷位置；無撞擊為 None）
    flight_pts: list = field(default_factory=list)  # Stage 3 輸出：接觸前後皆含，供呼叫端自行切分 pre/post
    contact_bat_confidence: str = "unknown"  # 'confirmed'/'unconfirmed'/'unknown'：球棒信心度佐證，僅供診斷，不影響 contact_frame


# ─── 工具函式 ──────────────────────────────────────────────────────────


def _bbox_max_side(det: Detection) -> float:
    return max(det.bbox[2] - det.bbox[0], det.bbox[3] - det.bbox[1])


# ─── Stage 0：P25 大框過濾 ────────────────────────────────────────────


def _p25_filter(dets: list) -> list:
    """以 P25 × BALL_BBOX_P25_RATIO 移除異常大 bbox 的 ball 偵測。"""
    if not dets:
        return dets
    sides = sorted(_bbox_max_side(d) for _, d in dets)
    p25_idx = max(0, int(len(sides) * 0.25) - 1)
    threshold = sides[p25_idx] * constants.BALL_BBOX_P25_RATIO
    filtered = [(fi, d) for fi, d in dets if _bbox_max_side(d) <= threshold]
    removed = len(dets) - len(filtered)
    if removed:
        print(f"[cloud] p25 filter: removed {removed} large-bbox detections "
              f"(threshold={threshold:.1f}px)")
    return filtered


# ─── Stage 1：靜止叢集識別 ───────────────────────────────────────────


def _split_by_relative_gap(cluster: list, ratio: float, min_gap_frames: float) -> list:
    """依時間間隔的相對倍數切段，不用純絕對幀數。

    以叢集自身相鄰偵測間隔的中位數為基準，門檻取
    max(中位數 × ratio, min_gap_frames)：偵測極密集（中位數趨近 1 幀）
    的物體改吃絕對下限，避免正常的幾幀空窗換算倍數後被誤判為異常；
    偵測本來就較稀疏的物體則用相對倍數，避免固定絕對值跨拍攝條件失效。
    點數不足 3（無法定義有意義的中位數基準）時整團當一段處理，不切。
    """
    sorted_pts = sorted(cluster, key=lambda p: p.frame_index)
    if len(sorted_pts) < 3:
        return [sorted_pts]

    gaps = [
        sorted_pts[i + 1].frame_index - sorted_pts[i].frame_index
        for i in range(len(sorted_pts) - 1)
    ]
    median_gap = sorted(gaps)[len(gaps) // 2]
    threshold = max(median_gap * ratio, min_gap_frames)

    segments: list[list] = []
    current = [sorted_pts[0]]
    for i in range(1, len(sorted_pts)):
        if gaps[i - 1] > threshold:
            segments.append(current)
            current = []
        current.append(sorted_pts[i])
    segments.append(current)
    return segments


def _build_static_clusters(
    candidates: list,
    max_spread: float = constants.CLOUD_STATIC_MAX_SPREAD_PX,
) -> list:
    """識別靜止背景球叢集。

    使用 CLOUD_STATIC_CLUSTER_RADIUS_PX（80px）作為空間聚類半徑，
    以涵蓋靜止球整片影片中的 YOLO 定位抖動（Legacy 的 25px 太小）。

    空間聚類後，依時間相對間隔（CLOUD_STATIC_SEGMENT_GAP_RATIO）切成
    子段落，每段各自獨立判斷是否為靜止叢集，段落之間不合併——避免單一
    長遮擋（如揮棒經過）拖垮全域密度導致漏判，也避免把物體真正離開後
    殘留的少量點誤併入原叢集邊界，汙染下游的接觸幀判定。

    每段條件（AND）：
      n >= BATCH_STATIC_MIN_FRAMES（fps 縮放後）
      spread < max_spread（第一輪呼叫預設 CLOUD_STATIC_MAX_SPREAD_PX＝150px；
        第二輪呼叫改傳 CLOUD_STATIC_ROUND2_MAX_SPREAD_PX 收緊門檻）
      detection_rate > CLOUD_STATIC_MIN_RATE（僅計算段落自身跨度）
    """
    radius = constants.CLOUD_STATIC_CLUSTER_RADIUS_PX
    raw = make_xy_clusters(candidates, radius)
    static: list[StaticCluster] = []

    for cluster in raw:
        segments = _split_by_relative_gap(
            cluster,
            constants.CLOUD_STATIC_SEGMENT_GAP_RATIO,
            constants.CLOUD_STATIC_SEGMENT_GAP_MIN_FRAMES,
        )
        for segment in segments:
            n = len(segment)
            if n < constants.BATCH_STATIC_MIN_FRAMES:
                continue
            start = min(p.frame_index for p in segment)
            end = max(p.frame_index for p in segment)
            span = max(1, end - start)
            rate = n / span
            if rate < constants.CLOUD_STATIC_MIN_RATE:
                continue
            xs = [p.x for p in segment]
            ys = [p.y for p in segment]
            spread = max(max(xs) - min(xs), max(ys) - min(ys))
            if spread > max_spread:
                continue
            cx = sum(xs) / n
            cy = sum(ys) / n
            static.append(StaticCluster(
                cx=cx, cy=cy, start_frame=start, end_frame=end, n=n, spread=spread,
            ))
            print(f"[cloud] static cluster ({cx:.0f},{cy:.0f}) "
                  f"n={n} rate={rate:.2f} f{start}-f{end} spread={spread:.0f}px")

    return static


# ─── Stage 2：靜止移除 ───────────────────────────────────────────────


def _remove_static(candidates: list, static_clusters: list) -> list:
    """移除落在靜止叢集空間範圍內且時間重疊的偵測點。

    移除條件（AND）：
      1. 距叢集中心 <= 移除半徑（與 Stage 1 聚類半徑一致，80px）
      2. 偵測幀落在叢集的 start_frame ~ end_frame 之間

    加入時間邊界是為了避免 T 座叢集（end_frame ≈ 接觸幀）把
    飛行球起飛後在同位置的偵測也一起移除。
    """
    def _removal_r(sc: StaticCluster) -> float:
        return max(
            sc.spread / 2 * 1.2,
            constants.CLOUD_STATIC_CLUSTER_RADIUS_PX,
        )

    return [
        p for p in candidates
        if not any(
            dist(p.x, p.y, sc.cx, sc.cy) <= _removal_r(sc)
            and sc.start_frame <= p.frame_index <= sc.end_frame
            for sc in static_clusters
        )
    ]


# ─── Stage 3：時序段落過濾 ───────────────────────────────────────────


def _segment_by_gap(pts: list) -> list:
    """按時間間距切段；超過 CLOUD_GAP_THRESHOLD_FRAMES 則開新段落。"""
    if not pts:
        return []
    gap_thr = constants.CLOUD_GAP_THRESHOLD_FRAMES
    segments: list[list] = []
    current = [pts[0]]
    for i in range(1, len(pts)):
        if pts[i].frame_index - pts[i - 1].frame_index > gap_thr:
            segments.append(current)
            current = []
        current.append(pts[i])
    segments.append(current)
    return segments


def _filter_noise_segments(pts: list) -> list:
    """移除點數不足 CLOUD_MIN_SEGMENT_POINTS 的孤立小段落。"""
    sorted_pts = sorted(pts, key=lambda p: p.frame_index)
    segments = _segment_by_gap(sorted_pts)
    min_pts = constants.CLOUD_MIN_SEGMENT_POINTS
    kept = [seg for seg in segments if len(seg) >= min_pts]
    removed_segs = len(segments) - len(kept)
    if removed_segs:
        print(f"[cloud] stage3: dropped {removed_segs} tiny segment(s)")
    result = []
    for seg in kept:
        result.extend(seg)
    return sorted(result, key=lambda p: p.frame_index)


# ─── Stage 4：情況判斷 ───────────────────────────────────────────────


def _detect_scenario(pts: list, static_clusters: list, total_frames: int) -> str:
    """判斷來球情況：tstand / pitch / sidetoss / unknown。

    T 座：存在在影片中段消失（非末尾）的靜止叢集。
    投球：flight_candidates 的 X 方向出現符號反轉（球飛來再被打出）。
    側拋：X 方向一致但不是 T 座。
    """
    for sc in static_clusters:
        if total_frames * 0.1 < sc.end_frame < total_frames * 0.9:
            return 'tstand'

    if len(pts) < 4:
        return 'unknown'

    vx_signs = []
    for i in range(len(pts) - 1):
        dframe = max(1, pts[i + 1].frame_index - pts[i].frame_index)
        vx = (pts[i + 1].x - pts[i].x) / dframe
        if abs(vx) > 2.0:
            vx_signs.append(1 if vx > 0 else -1)

    if len(vx_signs) < 2:
        return 'unknown'

    sign_changes = sum(
        1 for i in range(len(vx_signs) - 1) if vx_signs[i] != vx_signs[i + 1]
    )
    return 'pitch' if sign_changes >= 1 else 'sidetoss'


# ─── Stage 5：X/Y 軸離群值移除 ──────────────────────────────────────


def _mad_thresholds(clean: list) -> tuple[float, float]:
    """計算 MAD 閾值（X 和 Y 各自獨立）。

    rate = |Δpos / Δframe|（絕對值位移率）
    threshold = median_rate + K × MAD
    回傳 (threshold_x, threshold_y)。

    只納入幀距 <= CLOUD_GAP_THRESHOLD_FRAMES 的相鄰對（同段落內）。
    跨段落的對幀距極大、rate ≈ 0，會把 median 壓近 0 讓 threshold 失準。
    """
    if len(clean) < 2:
        return (float('inf'), float('inf'))

    gap_thr = constants.CLOUD_GAP_THRESHOLD_FRAMES
    rx, ry = [], []
    for i in range(len(clean) - 1):
        df = max(1, clean[i + 1].frame_index - clean[i].frame_index)
        if df > gap_thr:
            continue  # 跨段落，排除在 MAD 計算外
        rx.append(abs(clean[i + 1].x - clean[i].x) / df)
        ry.append(abs(clean[i + 1].y - clean[i].y) / df)

    def _mad(rates: list) -> tuple[float, float]:
        s = sorted(rates)
        med = s[len(s) // 2]
        mad = sorted(abs(r - med) for r in rates)[len(rates) // 2]
        return med, mad

    k = constants.CLOUD_MAD_K
    med_x, mad_x = _mad(rx)
    med_y, mad_y = _mad(ry)
    thr_x = max(2.0, med_x + k * mad_x)
    thr_y = max(2.0, med_y + k * mad_y)
    return thr_x, thr_y


def _filter_axis_outliers(pts: list, scenario: str) -> list:
    """MAD 基礎的孤立離群值移除，迭代直到收斂。

    孤立離群點的條件（前後兩側皆超標才移除，避免誤殺段落端點）：

    大小離群（X 或 Y）：
        某點的前向和後向位移率都超過 MAD 閾值
        → 孤立跳躍點，非連續飛行軌跡的一部分

    方向反轉（X，T座/側拋模式）：
        vx 符號前後反轉 AND 兩側速率都超過 threshold × DIR_MIN_RATIO
        → 排除 YOLO bbox jitter（1-3px/frame）的小波動誤殺
    """
    dir_ratio = constants.CLOUD_DIR_REVERSAL_MIN_RATIO
    clean = list(pts)

    for iteration in range(10):
        if len(clean) < 3:
            break

        thr_x, thr_y = _mad_thresholds(clean)

        to_remove: set[int] = set()
        for i in range(1, len(clean) - 1):
            df_b = max(1, clean[i].frame_index - clean[i - 1].frame_index)
            df_a = max(1, clean[i + 1].frame_index - clean[i].frame_index)

            vx_b = (clean[i].x - clean[i - 1].x) / df_b
            vx_a = (clean[i + 1].x - clean[i].x) / df_a
            vy_b = (clean[i].y - clean[i - 1].y) / df_b
            vy_a = (clean[i + 1].y - clean[i].y) / df_a

            rx_b, rx_a = abs(vx_b), abs(vx_a)
            ry_b, ry_a = abs(vy_b), abs(vy_a)

            # 大小離群：前後兩側都超過閾值才標記（孤立點才有雙側超標）
            mag_outlier = (
                (rx_b > thr_x and rx_a > thr_x) or
                (ry_b > thr_y and ry_a > thr_y)
            )

            # 方向反轉：T座/側拋才檢查，且兩側速率都需超過最低幅度（排除 jitter）
            dir_outlier = (
                scenario in ('tstand', 'sidetoss')
                and vx_b * vx_a < 0
                and rx_b > thr_x * dir_ratio
                and rx_a > thr_x * dir_ratio
            )

            if mag_outlier or dir_outlier:
                to_remove.add(i)

        if not to_remove:
            break
        new_clean = [p for i, p in enumerate(clean) if i not in to_remove]
        print(f"[cloud] stage5 iter{iteration+1}: "
              f"removed {len(to_remove)} outlier(s), {len(new_clean)} remain"
              f"  thr_x={thr_x:.1f} thr_y={thr_y:.1f}")
        clean = new_clean

    return clean


# ─── Stage 6：T 座球鎖定 + 接觸幀確認 ───────────────────────────────


def _identify_tstand(
    static_clusters: list,
    bat_positions: list,
    flight_seg,
    total_frames: int,
) -> StaticCluster | None:
    """識別哪個靜止叢集是 T 座球。

    T 座球的特徵（依優先序判斷）：
      0. 靜止叢集在影片中段消失（不是全片存在的背景球）——前提篩選
      1. 把這個候選位置當飛行段的已知起點併入現有偵測點重新擬合，曲線
         是否仍與原本（僅用真實偵測點擬合）的曲線吻合（見 _solve_tstand_
         contact）——不依賴球棒偵測品質，直接用這支影片自己的軌跡幾何
         證據，優先度最高
      2. 消失幀附近（±CLOUD_TSTAND_TIME_TOL）球棒曾掃過其位置——軌跡
         幾何無法唯一判斷、或沒有飛行段可用時的輔助佐證
      3. 消失幀離影片正中點最近——以上皆無法判斷時的最後備援，純屬猜測

    Args:
        bat_positions: list[BallPosition]，已清洗過的球棒軌跡點
            （呼叫端應餵入 BatCloudTracker.process() 的乾淨輸出，而非
            原始逐幀 YOLO 偵測——原始偵測含雜訊，會讓鄰近度判斷不穩定）。
        flight_seg: TrajectoryFitter 擬合出的飛行段（FittedSegment），
            無法擬合時傳 None，此時跳過軌跡幾何判斷直接用球棒/時間佐證。
    """
    terminating = [
        sc for sc in static_clusters
        if total_frames * 0.1 < sc.end_frame < total_frames * 0.9
    ]
    if not terminating:
        return None

    if flight_seg is not None:
        geo_candidates = [
            sc for sc in terminating
            if _solve_tstand_contact(flight_seg, sc) is not None
        ]
        if len(geo_candidates) == 1:
            best = geo_candidates[0]
            print(f"[cloud] tstand (trajectory-consistent): "
                  f"({best.cx:.0f},{best.cy:.0f}) end=f{best.end_frame}")
            return best
        if geo_candidates:
            # 多個候選都跟飛行軌跡線吻合，縮小範圍後續用球棒/時間佐證排序
            terminating = geo_candidates

    time_tol = constants.CLOUD_TSTAND_TIME_TOL
    bat_r = constants.CLOUD_TSTAND_SPACE_TOL_PX

    candidates = []
    for sc in terminating:
        best_bat_d = float('inf')
        for bp in bat_positions:
            if abs(bp.frame_index - sc.end_frame) > time_tol:
                continue
            d = dist(sc.cx, sc.cy, bp.x, bp.y)
            if d < best_bat_d:
                best_bat_d = d
        if best_bat_d <= bat_r:
            candidates.append((best_bat_d, sc))

    if not candidates:
        # 無軌跡幾何/球棒佐證時退回：僅依消失幀選取（最靠近影片中點者）
        mid = total_frames / 2
        terminating.sort(key=lambda sc: abs(sc.end_frame - mid))
        best = terminating[0]
        print(f"[cloud] tstand (fallback, no bat confirm): "
              f"({best.cx:.0f},{best.cy:.0f}) end=f{best.end_frame}")
        return best

    candidates.sort(key=lambda x: x[0])
    best = candidates[0][1]
    print(f"[cloud] tstand (bat confirmed): ({best.cx:.0f},{best.cy:.0f}) "
          f"end=f{best.end_frame} bat_dist={candidates[0][0]:.0f}px")
    return best


def _validate_curve_extension(
    flight_seg,
    candidate_points: list[BallPosition],
    angle_tol_deg: float | None = None,
) -> bool:
    """把候選點集（座標已知的邊界點，如 T座起點、反彈終點；或緊接在後面
    被誤切的整段點集）併入飛行段既有真實偵測點重新擬合，比較候選點那一
    側邊界的速度向量方向在加入前後變化多少，藉此判斷候選跟現有飛行段
    是不是同一條軌跡（見與使用者討論 2026-07-23：以方向角度為主，先不
    加入速度量級的判斷——「是不是同一條線」是純幾何問題，由方向決定；
    速度快慢是另一個額外的物理假設（飛行速度需穩定），不必然成立，不該
    用來否決一個方向明明吻合的候選點）。

    候選點的座標是獨立量到或直接偵測到的硬事實，不是從飛行段曲線外插
    解出來的，也不受它離現有軌跡線多遠限制。T座起點、反彈終點、整段
    合併判斷都適用同一套邏輯，共用這個函式。

    Args:
        flight_seg: 既有飛行段。
        candidate_points: 候選點集合（可以是單一點包成的長度 1 列表，
            也可以是整段點集）；一律以時間排序後併入。
        angle_tol_deg: 角度容忍值；None 時使用單點併入既有校準的
            TRAJFIT_EXTENSION_ANGLE_TOL_DEG。整段併入應傳入獨立的
            TRAJFIT_SEGMENT_MERGE_ANGLE_TOL_DEG——見該常數註解，
            不假設兩種情境共用同一個容忍值。
    """
    if angle_tol_deg is None:
        angle_tol_deg = constants.TRAJFIT_EXTENSION_ANGLE_TOL_DEG

    merged = sorted(flight_seg.points + candidate_points, key=lambda p: p.frame_index)
    new_coeffs_x, new_coeffs_y = _fit_exact(merged, flight_seg.order)

    # 候選點集在起點外還是終點外，決定要在原始飛行段哪一側的邊界評估
    # 速度向量方向——只在乎軌跡被候選點拉動的那一端變化多少
    cand_min = min(p.frame_index for p in candidate_points)
    cand_max = max(p.frame_index for p in candidate_points)
    at_frame = (
        flight_seg.start_frame if cand_max < flight_seg.start_frame
        else flight_seg.end_frame
    )

    deriv_x_old = np.polyder(np.array(flight_seg.coeffs_x, dtype=np.float64))
    deriv_y_old = np.polyder(np.array(flight_seg.coeffs_y, dtype=np.float64))
    deriv_x_new = np.polyder(np.array(new_coeffs_x, dtype=np.float64))
    deriv_y_new = np.polyder(np.array(new_coeffs_y, dtype=np.float64))

    v_old = np.array([np.polyval(deriv_x_old, at_frame), np.polyval(deriv_y_old, at_frame)])
    v_new = np.array([np.polyval(deriv_x_new, at_frame), np.polyval(deriv_y_new, at_frame)])

    speed_old = float(np.linalg.norm(v_old))
    speed_new = float(np.linalg.norm(v_new))
    if speed_old < 1e-6 or speed_new < 1e-6:
        return False

    cos_angle = float(np.dot(v_old, v_new) / (speed_old * speed_new))
    cos_angle = max(-1.0, min(1.0, cos_angle))
    angle_deg = math.degrees(math.acos(cos_angle))

    return angle_deg <= angle_tol_deg


def _merge_flight_extension_segment(fit_result, flight_seg) -> None:
    """檢查緊接在飛行段後面、時間序最早的下一段，是不是只是被單幀異常
    跳點誤切的同一條軌跡（見與使用者討論 2026-07-28：例如丟幀造成
    frame_index 與真實經過時間不成比例，讓 Sequential RANSAC 在那一格
    產生過大殘差而誤切成兩段——切開的兩段各自內部速率其實一致，只有
    邊界那一格異常）。只檢查緊接在後面那一段，不做鏈式掃描——避免重演
    BatCloudTracker D1 那種貪婪演算法被中間小段卡住、真正該比對的兩段
    永遠沒機會直接比對的問題。

    通過方向一致性驗證（見 _validate_curve_extension；此處用整段點集，
    容忍值改用較嚴格的 TRAJFIT_SEGMENT_MERGE_ANGLE_TOL_DEG）就合併回
    飛行段本身（更新 points/coeffs_x/coeffs_y），並把該段從
    fit_result.segments 移除；沒通過就維持原樣，不影響後續 bounce
    角色判斷（含 boundary_frames["impact"] 的段落選取）。
    """
    if flight_seg is None:
        return
    idx = fit_result.segments.index(flight_seg)
    if idx + 1 >= len(fit_result.segments):
        return
    next_seg = fit_result.segments[idx + 1]

    if not _validate_curve_extension(
        flight_seg, next_seg.points,
        angle_tol_deg=constants.TRAJFIT_SEGMENT_MERGE_ANGLE_TOL_DEG,
    ):
        return

    merged_points = sorted(flight_seg.points + next_seg.points, key=lambda p: p.frame_index)
    flight_seg.coeffs_x, flight_seg.coeffs_y = _fit_exact(merged_points, flight_seg.order)
    flight_seg.points = merged_points
    fit_result.segments.remove(next_seg)
    print(
        f"[cloud] flight extension: merged f{next_seg.start_frame}-f{next_seg.end_frame} "
        f"(was {next_seg.role}) back into flight — direction-consistent, "
        f"likely same trajectory split by a single-frame anomaly"
    )


def _solve_tstand_contact(flight_seg, tstand: StaticCluster) -> int | None:
    """T座情境：把 T座位置當成飛行段的已知起點，用 _validate_curve_extension
    驗證是否跟現有飛行段是同一條軌跡；不吻合時回傳 None 交由呼叫端退回
    既有備援序（球棒佐證／消失幀最近影片中點）。
    """
    candidate = BallPosition(
        x=tstand.cx, y=tstand.cy, frame_index=tstand.end_frame, confidence=1.0,
    )
    if not _validate_curve_extension(flight_seg, [candidate]):
        return None
    return tstand.end_frame


def _bat_confidence_tag(
    contact_frame: int | None,
    contact_xy: tuple[float, float] | None,
    bat_positions: list,
) -> str:
    """球棒信心度佐證（診斷用，不影響 contact_frame 數值本身）。

    撞擊發生時球棒理論上必定通過或極接近擊球點；吻合與否只反映球棒
    偵測本身的品質，不可用來否決一個已有物理充分理由成立的接觸幀判斷
    （見 PHYSICS_VALIDATION_PRINCIPLES.md B2「球棒的角色」）。
    """
    if contact_frame is None or contact_xy is None:
        return "unknown"
    time_tol = constants.CLOUD_TSTAND_TIME_TOL
    space_tol = constants.CLOUD_TSTAND_SPACE_TOL_PX
    cx, cy = contact_xy
    nearby = any(
        abs(bp.frame_index - contact_frame) <= time_tol
        and dist(bp.x, bp.y, cx, cy) <= space_tol
        for bp in bat_positions
    )
    return "confirmed" if nearby else "unconfirmed"


# ─── 飛行段落截斷（撞擊點偵測）────────────────────────────────────────


def _truncate_at_impact(pts: list) -> tuple[list, int | None]:
    """軌跡夾角偵測 impact 點並截斷。

    物理前提：球在自由飛行中路徑平滑；突然出現的銳利夾角 = 撞到了東西。

    判斷邏輯：
      對每個三連點 (p[i-1], p[i], p[i+1]) 計算夾角 cos θ。
      找到第一個 cos θ 低於閾值且後續 N_CONFIRM 個點持續維持新方向的點。
      孤立離群值即使造成夾角，後續方向會回復 → 不滿足持續性 → 跳過繼續掃。

    Returns:
        (valid_pts, impact_frame)
        impact_frame 為 None 表示無明顯持續夾角（無撞擊）。
    """
    n_confirm = constants.CLOUD_IMPACT_N_CONFIRM
    angle_thr = constants.CLOUD_IMPACT_ANGLE_COS_THRESHOLD

    if len(pts) < 3 + n_confirm:
        return pts, None

    for i in range(1, len(pts) - n_confirm):
        ax = pts[i].x - pts[i - 1].x
        ay = pts[i].y - pts[i - 1].y
        bx = pts[i + 1].x - pts[i].x
        by = pts[i + 1].y - pts[i].y
        len_a = math.hypot(ax, ay)
        len_b = math.hypot(bx, by)
        if len_a < 1e-6 or len_b < 1e-6:
            continue
        cos_theta = (ax * bx + ay * by) / (len_a * len_b)

        if cos_theta >= angle_thr:
            continue

        # 後續 N_CONFIRM 個連續對的位移向量與 B 方向一致 → 持續性確認
        sustained = all(
            (pts[j + 1].x - pts[j].x) * bx + (pts[j + 1].y - pts[j].y) * by > 0
            for j in range(i + 1, i + n_confirm)
            if j + 1 < len(pts)
        )
        if sustained:
            impact_frame = pts[i].frame_index
            print(f"[cloud] impact at f{impact_frame} "
                  f"(cos={cos_theta:.2f}): "
                  f"truncating {len(pts) - i} post-impact pts")
            return pts[:i], impact_frame

    return pts, None


# ─── 主類別 ──────────────────────────────────────────────────────────


class CloudBallTracker:
    """點雲式球體追蹤器。

    接受所有幀的完整 YOLO ball 偵測（非 best-per-frame），
    執行六階段管線輸出乾淨飛行軌跡與診斷資訊。

    使用方式：
        tracker = CloudBallTracker()
        result = tracker.process(all_ball_dets, bat_positions, total_frames)
    """

    def process(
        self,
        all_dets: list,        # list[tuple[int, Detection]]
        bat_positions: BatPositionsInput,  # 呼叫端須用 .cleaned()/.raw()
                               # 明確標記來源（見 BatPositionsInput 說明）；
                               # .cleaned() 較穩定，T 座辨識/信心度佐證的
                               # 鄰近度判斷用原始資料雜訊較高
        total_frames: int,
        skip_truncation: bool = False,
    ) -> CloudTrackResult:
        print(
            f"[cloud] input: {len(all_dets)} raw ball detections"
            f"  (bat_positions: {bat_positions.source})"
        )
        bat_positions = bat_positions.positions

        # Stage 0: P25 大框過濾
        filtered_dets = _p25_filter(all_dets)

        # 轉換為 BallPosition 序列
        candidates: list[BallPosition] = sorted([
            BallPosition(
                x=(d.bbox[0] + d.bbox[2]) / 2,
                y=(d.bbox[1] + d.bbox[3]) / 2,
                frame_index=fi,
                confidence=d.confidence,
            )
            for fi, d in filtered_dets
        ], key=lambda p: p.frame_index)

        print(f"[cloud] after p25: {len(candidates)} detections")

        if not candidates:
            return CloudTrackResult([], [], None, None, 'unknown')

        # Stage 1: 靜止叢集識別
        static_clusters = _build_static_clusters(candidates)
        print(f"[cloud] stage1: {len(static_clusters)} static cluster(s)")

        # Stage 2: 靜止移除
        sparse = _remove_static(candidates, static_clusters)
        print(f"[cloud] stage2: {len(sparse)} pts remain "
              f"(removed {len(candidates) - len(sparse)})")

        if not sparse:
            return CloudTrackResult([], static_clusters, None, None, 'unknown')

        # Stage 3: 時序雜訊過濾
        flight_pts = _filter_noise_segments(sparse)
        print(f"[cloud] stage3: {len(flight_pts)} pts after segment filter")

        # Stage 4: 情況判斷
        scenario = _detect_scenario(flight_pts, static_clusters, total_frames)
        print(f"[cloud] stage4: scenario = {scenario}")

        # 第二輪靜止叢集移除：對 flight_pts（Stage 3 存活點）再跑一次
        # 同樣的空間聚類判斷，但門檻收緊為 CLOUD_STATIC_ROUND2_MAX_SPREAD_PX，
        # 攔截第一輪因貪婪聚類順序（_make_xy_clusters）漏網的靜止殘留——
        # 這批殘留若直接流入 TrajectoryFitter，密度/間距/單調性三項檢查
        # 對「幾乎不動」的情況是盲的，可能被誤判為合法飛行段（見
        # REVIEW_2026-04-06.md 靜止殘留誤判紀錄）
        round2_static = _build_static_clusters(
            flight_pts, max_spread=constants.CLOUD_STATIC_ROUND2_MAX_SPREAD_PX,
        )
        flight_pts = _remove_static(flight_pts, round2_static)
        print(f"[cloud] stage4b: {len(round2_static)} round-2 static "
              f"cluster(s), {len(flight_pts)} pts remain")

        # TrajectoryFitter: Sequential RANSAC 遞迴擬合，取代 Stage 5 MAD
        # 與 _truncate_at_impact——直接吃 Stage 3 輸出，段落邊界（來球段/
        # 飛行段/反彈段）從擬合過程本身浮現，不依賴 contact_frame。
        # 需在 T 座身份辨識之前執行：_identify_tstand 優先用飛行段軌跡線
        # 的幾何吻合度判斷候選，需要這裡先算出 flight_seg。
        fit_result = fit_trajectory_segments(flight_pts, scenario)
        for seg in fit_result.segments:
            print(f"[cloud] trajfit: {seg.role:12s} order={seg.order} "
                  f"n={len(seg.points)} f{seg.start_frame}-f{seg.end_frame}")
        print(f"[cloud] trajfit: discarded={len(fit_result.discarded)}")

        flight_seg = next((s for s in fit_result.segments if s.role == "flight"), None)

        # 單幀異常跳點（可能是丟幀造成）誤切飛行段：檢查緊接在後面的
        # 下一段方向是否一致，一致就合併回飛行段本身（見與使用者討論
        # 2026-07-28）。需在 T 座身份辨識與 boundary_frames 使用之前
        # 執行，讓下游都拿到合併後的正確飛行段與段落列表。
        _merge_flight_extension_segment(fit_result, flight_seg)

        # T 座球身份辨識：優先用飛行段軌跡線的幾何吻合度判斷候選，
        # 球棒佐證與「消失幀最接近影片中點」皆降級為輔助/最後備援
        tstand = _identify_tstand(static_clusters, bat_positions, flight_seg, total_frames)

        # contact_frame 判斷（見 PHYSICS_VALIDATION_PRINCIPLES.md B2/C2/C4/C5）：
        # 轉折點/T座位置本身即為擊球點的充分證據，球棒不參與此數值的決定，
        # 只在下方另外算信心度佐證標籤。
        contact_xy: tuple[float, float] | None = None
        tstand_contact_validated = False
        if scenario == "tstand" and tstand is not None and flight_seg is not None:
            contact = _solve_tstand_contact(flight_seg, tstand)
            tstand_contact_validated = contact is not None
            if contact is None:
                contact = tstand.end_frame
            contact_xy = (tstand.cx, tstand.cy)
            print(f"[cloud] tstand contact (refit consistency): {contact}")
        else:
            contact = fit_result.boundary_frames["contact"]
            if contact is not None and flight_seg is not None:
                contact_xy = (
                    float(np.polyval(flight_seg.coeffs_x, contact)),
                    float(np.polyval(flight_seg.coeffs_y, contact)),
                )
        # 反彈點偵測：TrajectoryFitter 的 _passes_segment_checks 要求線段
        # 全程水平方向單調（見 trajectory_fitter.py），但真正的反彈本來就
        # 會反轉方向，導致這批點連不合格線段都算不上，直接被歸進
        # discarded，boundary_frames["impact"] 因此拿不到值。改用舊版
        # Stage 5 的三點夾角測試（_truncate_at_impact，仍保留在本檔案，
        # 只是 process() 先前沒有呼叫它）在 contact 之後的原始候選點上
        # 直接偵測夾角突變，作為 TrajectoryFitter 沒找到反彈段時的備援
        # （見與使用者討論 2026-07-23）。只用來還原 impact_frame 這個
        # 標記值，不影響 clean_pts／flight_seg 本身的點組成。
        impact_frame = fit_result.boundary_frames["impact"]
        if impact_frame is None and contact is not None:
            post_pts = [p for p in flight_pts if p.frame_index > contact]
            _, angle_impact_frame = _truncate_at_impact(post_pts)
            if angle_impact_frame is not None:
                impact_frame = angle_impact_frame

        # 反彈點驗證：impact_frame 對應的座標是 Stage 4b 就已經真實偵測到
        # 的點（_truncate_at_impact 挑出來的轉折幀本身），不需要外插，直接
        # 從 flight_pts 撈出來即可。跟 T座起點一樣用 _validate_curve_extension
        # 驗證是否跟現有飛行段是同一條軌跡（見與使用者討論 2026-07-23：
        # 反彈點應該視為飛行段終點，套用跟 T座起點相同的合理性驗證）。
        # 驗證要在下面併入 contact 候選點之前、對照原始（僅真實偵測點）
        # 的飛行段曲線進行，避免兩個候選點的驗證互相影響。
        #
        # 驗證沒過（或該幀根本不在 flight_pts 裡）時，impact_frame 這個
        # 對外回報的標記值本身也一併視為未知、重置為 None，不是只影響
        # 併不併入 clean_pts。原本兩者脫鉤：驗證失敗只代表不併入計算，
        # 畫面上仍會標出這個未經確認的點，可能誤導成「這裡就是撞擊點」
        # （例如球被遮擋物擋住、之後偵測到的其實是不相關的另一顆球時）。
        # 這裡只是承認「觀測不到就是不知道」，不試圖用其他方式推算真正
        # 的撞擊幀（見與使用者討論 2026-07-29）。
        impact_candidate: BallPosition | None = None
        if impact_frame is not None and flight_seg is not None:
            found = next(
                (p for p in flight_pts if p.frame_index == impact_frame), None,
            )
            if found is not None and _validate_curve_extension(flight_seg, [found]):
                impact_candidate = found
            else:
                impact_frame = None

        # 驗證通過的候選點併入飛行段軌跡，形成完整的飛行段軌跡線，供下游
        # EV/LA 計算使用（見與使用者討論 2026-07-22/24：只算出 contact_frame
        # 數值不夠，候選點還要實際成為軌跡點；擬合成功時，最終輸出的線
        # 本身也要是併入候選點後重新擬合出的那條線，不能點列表已經更新、
        # coeffs_x/coeffs_y 卻還是舊的、沒有候選點的那條）。T座情境只在
        # 通過重擬合一致性驗證時才併入；投球/側拋情境只在真的解出缺口
        # 交點（非退回飛行段起始幀）時才併入；反彈點只在通過同樣的一致性
        # 驗證時才併入。兩個候選點都通過時，用併入兩點後的完整點集一次
        # 重新擬合，讓 coeffs 跟點列表保持一致。
        if flight_seg is not None:
            new_points = list(flight_seg.points)
            if contact_xy is not None:
                should_inject_contact = (
                    (scenario == "tstand" and tstand_contact_validated)
                    or (scenario != "tstand" and contact != flight_seg.start_frame)
                )
                if should_inject_contact:
                    new_points.append(BallPosition(
                        x=contact_xy[0], y=contact_xy[1],
                        frame_index=contact, confidence=1.0,
                    ))
            if impact_candidate is not None:
                new_points.append(impact_candidate)
            if len(new_points) != len(flight_seg.points):
                new_points.sort(key=lambda p: p.frame_index)
                flight_seg.coeffs_x, flight_seg.coeffs_y = _fit_exact(
                    new_points, flight_seg.order,
                )
                flight_seg.points = new_points
                print(f"[cloud] contact/impact point(s) merged, flight segment "
                      f"refit ({len(flight_seg.points)} pts)")

        bat_confidence = _bat_confidence_tag(contact, contact_xy, bat_positions)
        print(f"[cloud] bat confidence (diagnostic only): {bat_confidence}")

        if skip_truncation:
            # 不截斷撞擊/反彈：飛行段 + 反彈段全部合併輸出
            clean_pts = sorted(
                (p for seg in fit_result.segments for p in seg.points),
                key=lambda p: p.frame_index,
            )
            impact_frame = None
            print(f"[cloud] truncation: skipped (skip_truncation=True)")
        else:
            clean_pts = fit_result.flight_positions
            if impact_frame is None:
                print(f"[cloud] truncation: no impact detected")

        print(f"[cloud] final: {len(clean_pts)} clean pts  "
              f"contact={contact}  impact={impact_frame}")

        return CloudTrackResult(
            positions=clean_pts,
            static_blacklist=static_clusters,
            tstand_cluster=tstand,
            contact_frame=contact,
            scenario=scenario,
            impact_frame=impact_frame,
            flight_pts=flight_pts,
            contact_bat_confidence=bat_confidence,
        )
