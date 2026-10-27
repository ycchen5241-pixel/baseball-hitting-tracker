# detector/bat_tracker_cloud.py
# Bat-top 點雲式軌跡過濾器（實驗性，對照 physics/trajectory.py 的 arc-chain pipeline）
#
# 沿用 CloudBallTracker（detector/ball_tracker_cloud.py）的核心手法：
# 全片段收集 bat_top 偵測，用 XY 密度聚類識別「持棒站姿 / 跟隨動作定格」
# 等靜止叢集並移除，再用純速率 MAD 過濾孤立誤偵測跳點。
#
# 與 CloudBallTracker 的差異：
#   - 只有一個追蹤類別（Bat-top），不涉及跨類別（棒頭/棒身）幾何混合
#   - 不做球專屬的 T座鎖定 / 接觸幀確認 / 撞擊截斷（Stage 4/6）
#   - Stage 5 拿掉方向反轉離群值規則——揮棒的方向反轉（後擺頂點、跟隨
#     動作轉折）是真實訊號，不是雜訊
#   - 常數獨立校準（BAT_CLOUD_*），不沿用球的 CLOUD_* 數值
#
# 管線：
#   Stage 1  XY 密度聚類 → 靜止叢集識別
#   Stage 2  靜止叢集移除
#   Stage 3  時序段落過濾（移除孤立雜訊點群）
#   ├─ 揮棒視窗偵測（轉向幅度 + 方向一致性掃描，見 _detect_swing_windows_shape）
#   │  以 Stage 3 輸出為輸入，跟 Stage 5' 平行、互不影響
#   └─ Stage 5' 純速率 MAD 離群值移除 → 乾淨點序列（result.positions）
#
# 輸入：list[BallPosition]（單一 best-per-frame bat_top，沿用現有慣例）
# 輸出：BatCloudTrackResult

import math
from dataclasses import dataclass, field

from config import constants
from detector.ball_tracker_cloud import StaticCluster
from detector.centroid import BallPosition
from detector.clustering import dist, make_xy_clusters
from physics.trajectory import _ransac_circle_fit


@dataclass
class BatCloudTrackResult:
    positions: list        = field(default_factory=list)  # list[BallPosition]：乾淨揮棒軌跡
    static_clusters: list  = field(default_factory=list)   # list[StaticCluster]
    removed_static: list   = field(default_factory=list)   # 被靜止叢集移除的點（診斷用）
    removed_noise: list    = field(default_factory=list)   # 被雜訊段落移除的點
    removed_outlier: list  = field(default_factory=list)   # 被速率離群值移除的點
    swing_windows: list    = field(default_factory=list)   # list[tuple[int, int]]


# ─── Stage 1：靜止叢集識別 ───────────────────────────────────────────


def _build_static_clusters(candidates: list) -> list:
    """識別靜止叢集（持棒站姿等待 / 跟隨動作定格）。

    條件（AND）：
      n >= BAT_CLOUD_STATIC_MIN_FRAMES（fps 縮放後）
      spread < BAT_CLOUD_STATIC_MAX_SPREAD_PX
      detection_rate > BAT_CLOUD_STATIC_MIN_RATE
    """
    radius = constants.BAT_CLOUD_STATIC_CLUSTER_RADIUS_PX
    raw = make_xy_clusters(candidates, radius)
    static: list[StaticCluster] = []

    for cluster in raw:
        n = len(cluster)
        if n < constants.BAT_CLOUD_STATIC_MIN_FRAMES:
            continue
        start = min(p.frame_index for p in cluster)
        end = max(p.frame_index for p in cluster)
        span = max(1, end - start)
        rate = n / span
        if rate < constants.BAT_CLOUD_STATIC_MIN_RATE:
            continue
        xs = [p.x for p in cluster]
        ys = [p.y for p in cluster]
        spread = max(max(xs) - min(xs), max(ys) - min(ys))
        if spread > constants.BAT_CLOUD_STATIC_MAX_SPREAD_PX:
            continue
        cx = sum(xs) / n
        cy = sum(ys) / n
        static.append(StaticCluster(
            cx=cx, cy=cy, start_frame=start, end_frame=end, n=n, spread=spread,
        ))
        print(f"[bat-cloud] static cluster ({cx:.0f},{cy:.0f}) "
              f"n={n} rate={rate:.2f} f{start}-f{end} spread={spread:.0f}px")

    return static


# ─── Stage 2：靜止移除 ───────────────────────────────────────────────


def _remove_static(candidates: list, static_clusters: list) -> tuple[list, list]:
    """移除落在靜止叢集空間範圍內且時間重疊的偵測點。回傳 (保留, 移除)。"""
    def _removal_r(sc: StaticCluster) -> float:
        return max(
            sc.spread / 2 * 1.2,
            constants.BAT_CLOUD_STATIC_CLUSTER_RADIUS_PX,
        )

    kept: list = []
    removed: list = []
    for p in candidates:
        hit = any(
            dist(p.x, p.y, sc.cx, sc.cy) <= _removal_r(sc)
            and sc.start_frame <= p.frame_index <= sc.end_frame
            for sc in static_clusters
        )
        (removed if hit else kept).append(p)
    return kept, removed


# ─── Stage 3：時序段落過濾 ───────────────────────────────────────────


def _segment_by_gap(pts: list) -> list:
    """按時間間距切段；超過 BAT_CLOUD_GAP_THRESHOLD_FRAMES 則開新段落。"""
    if not pts:
        return []
    gap_thr = constants.BAT_CLOUD_GAP_THRESHOLD_FRAMES
    segments: list[list] = []
    current = [pts[0]]
    for i in range(1, len(pts)):
        if pts[i].frame_index - pts[i - 1].frame_index > gap_thr:
            segments.append(current)
            current = []
        current.append(pts[i])
    segments.append(current)
    return segments


def _filter_noise_segments(pts: list) -> tuple[list, list]:
    """移除點數不足 BAT_CLOUD_MIN_SEGMENT_POINTS 的孤立小段落。回傳 (保留, 移除)。"""
    sorted_pts = sorted(pts, key=lambda p: p.frame_index)
    segments = _segment_by_gap(sorted_pts)
    min_pts = constants.BAT_CLOUD_MIN_SEGMENT_POINTS
    kept_segs = [seg for seg in segments if len(seg) >= min_pts]
    dropped_segs = [seg for seg in segments if len(seg) < min_pts]
    if dropped_segs:
        print(f"[bat-cloud] stage3: dropped {len(dropped_segs)} tiny segment(s)")
    kept = sorted((p for seg in kept_segs for p in seg), key=lambda p: p.frame_index)
    removed = [p for seg in dropped_segs for p in seg]
    return kept, removed


# ─── Stage 5'：純速率 MAD 離群值移除（無方向反轉規則） ─────────────


def _mad_thresholds(clean: list) -> tuple[float, float]:
    """計算 MAD 閾值（X 和 Y 各自獨立）。只納入同段落內相鄰對。"""
    if len(clean) < 2:
        return (float('inf'), float('inf'))

    gap_thr = constants.BAT_CLOUD_GAP_THRESHOLD_FRAMES
    rx, ry = [], []
    for i in range(len(clean) - 1):
        df = max(1, clean[i + 1].frame_index - clean[i].frame_index)
        if df > gap_thr:
            continue
        rx.append(abs(clean[i + 1].x - clean[i].x) / df)
        ry.append(abs(clean[i + 1].y - clean[i].y) / df)

    if not rx or not ry:
        return (float('inf'), float('inf'))

    def _mad(rates: list) -> tuple[float, float]:
        s = sorted(rates)
        med = s[len(s) // 2]
        mad = sorted(abs(r - med) for r in rates)[len(rates) // 2]
        return med, mad

    k = constants.BAT_CLOUD_MAD_K
    med_x, mad_x = _mad(rx)
    med_y, mad_y = _mad(ry)
    thr_x = max(2.0, med_x + k * mad_x)
    thr_y = max(2.0, med_y + k * mad_y)
    return thr_x, thr_y


def _filter_speed_outliers(pts: list) -> tuple[list, list]:
    """孤立速率離群值移除，迭代直到收斂。回傳 (保留, 移除)。

    只保留「大小離群」規則（前後兩側位移率皆超過 MAD 閾值才移除），
    不含 CloudBallTracker Stage 5 的方向反轉規則——揮棒的方向反轉
    （後擺頂點、跟隨動作轉折）是真實訊號，套用該規則會誤刪轉折點。
    """
    clean = list(pts)
    removed_all: list = []

    for iteration in range(10):
        if len(clean) < 3:
            break

        thr_x, thr_y = _mad_thresholds(clean)

        to_remove: set[int] = set()
        for i in range(1, len(clean) - 1):
            df_b = max(1, clean[i].frame_index - clean[i - 1].frame_index)
            df_a = max(1, clean[i + 1].frame_index - clean[i].frame_index)

            rx_b = abs(clean[i].x - clean[i - 1].x) / df_b
            rx_a = abs(clean[i + 1].x - clean[i].x) / df_a
            ry_b = abs(clean[i].y - clean[i - 1].y) / df_b
            ry_a = abs(clean[i + 1].y - clean[i].y) / df_a

            if (rx_b > thr_x and rx_a > thr_x) or (ry_b > thr_y and ry_a > thr_y):
                to_remove.add(i)

        if not to_remove:
            break
        removed_all.extend(clean[i] for i in to_remove)
        new_clean = [p for i, p in enumerate(clean) if i not in to_remove]
        print(f"[bat-cloud] stage5 iter{iteration+1}: "
              f"removed {len(to_remove)} outlier(s), {len(new_clean)} remain"
              f"  thr_x={thr_x:.1f} thr_y={thr_y:.1f}")
        clean = new_clean

    return clean, removed_all


# ─── 揮棒視窗偵測（轉向幅度 + 方向一致性掃描） ─────────────────────


def _smoothed_speeds(pts: list) -> list[tuple[int, float]]:
    """3 幀中值濾波後的逐點速度，跳過超過 gap 閾值的斷點配對。"""
    gap_thr = constants.BAT_CLOUD_GAP_THRESHOLD_FRAMES
    raw: list = [None] * len(pts)
    for i in range(1, len(pts)):
        df = pts[i].frame_index - pts[i - 1].frame_index
        if df > gap_thr:
            continue
        d = dist(pts[i - 1].x, pts[i - 1].y, pts[i].x, pts[i].y)
        raw[i] = d / max(1, df)

    smoothed: list[tuple[int, float]] = []
    for i in range(len(pts)):
        window = [raw[j] for j in range(max(0, i - 1), min(len(pts), i + 2))
                  if raw[j] is not None]
        if window:
            smoothed.append((i, sorted(window)[len(window) // 2]))
    return smoothed


def _heading(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.degrees(math.atan2(y2 - y1, x2 - x1))


def _signed_angle_diff(a: float, b: float) -> float:
    """a - b 正規化到 (-180, 180]。"""
    return (a - b + 180) % 360 - 180


def _turn_angles(pts: list) -> list[tuple[int, float]]:
    """算每個內部點的簽名轉向角（度）：用較寬的基線（跳過
    BAT_CLOUD_TURN_STRIDE_FRAMES 幀）比較 heading(p[i-stride]→p[i]) 到
    heading(p[i]→p[i+stride]) 的變化量，而非直接用相鄰幀。

    實測發現（IMG_3294）：用相鄰單幀（stride=1）算轉向角，偵測雜訊
    （bbox 中心跳動）在角度計算上會被放大成劇烈的正負號亂跳，即使在
    最快揮棒區間也一樣（逐幀算出 74.6, -89.7, 33.8... 完全不連貫）；
    改用較寬基線後，同一段真實資料算出的角度變得平滑、方向持續一致。

    跳過跨過大缺口、或前後任一段位移小於 BAT_CLOUD_TURN_MIN_STEP_PX ×
    stride 的點（位移太小時角度計算不穩定）。
    """
    gap_thr = constants.BAT_CLOUD_GAP_THRESHOLD_FRAMES
    min_step = constants.BAT_CLOUD_TURN_MIN_STEP_PX
    stride = constants.BAT_CLOUD_TURN_STRIDE_FRAMES
    turns: list[tuple[int, float]] = []
    for i in range(stride, len(pts) - stride):
        p0, p1, p2 = pts[i - stride], pts[i], pts[i + stride]
        df1 = p1.frame_index - p0.frame_index
        df2 = p2.frame_index - p1.frame_index
        if df1 > gap_thr or df2 > gap_thr:
            continue
        d1 = dist(p0.x, p0.y, p1.x, p1.y)
        d2 = dist(p1.x, p1.y, p2.x, p2.y)
        if d1 < min_step * stride or d2 < min_step * stride:
            continue
        h1 = _heading(p0.x, p0.y, p1.x, p1.y)
        h2 = _heading(p1.x, p1.y, p2.x, p2.y)
        turns.append((i, _signed_angle_diff(h2, h1)))
    return turns


def _find_all_curve_runs(pts: list, min_turn_deg: float) -> list[tuple[int, int]]:
    """找出所有「轉向幅度夠大 + 方向持續同一個正負號」的連續段（不只留
    最長的一段）——用於後續判斷時間相鄰的片段是不是同一次揮棒被誤切、
    需要合併（見 _try_merge_fragments）。

    不假設路徑是任何全域形狀（圓、橢圓）——只要求每一步轉向角度超過
    門檻（不是隨便彎一點就算）、且正負號跟前一步一致，連續段中間若
    跨過缺口/小位移被跳過的點也視為中斷。回傳依時間順序排列的
    (pts 起始索引, 結束索引) 列表。
    """
    turns = _turn_angles(pts)
    runs: list[tuple[int, int]] = []
    run: list[tuple[int, float]] = []

    def _flush() -> None:
        if len(run) >= 2:
            runs.append((run[0][0], run[-1][0]))

    for idx, turn in turns:
        if abs(turn) < min_turn_deg:
            _flush()
            run.clear()
            continue
        same_run = run and idx == run[-1][0] + 1 and (turn > 0) == (run[-1][1] > 0)
        if run and not same_run:
            _flush()
            run.clear()
        run.append((idx, turn))
    _flush()

    return runs


def _try_merge_fragments(
    seg: list, frag_a: tuple[int, int], frag_b: tuple[int, int], frame_width: int,
) -> tuple[int, int] | None:
    """檢查時間相鄰的兩個片段是不是同一段揮棒被誤切，是的話合併。

    只在片段間距在 BAT_CLOUD_MERGE_MAX_GAP_FRAMES 內、且雙方點數都達
    BAT_CLOUD_MERGE_MIN_FRAGMENT_POINTS 才嘗試——避免每組片段都硬跑一次
    RANSAC。合併驗證：對兩片段（含中間點）的聯集跑圓弧擬合，A、B 各自
    的內點率都要達標才接受合併——只看合併後整體內點率，可能被其中一段
    主導、掩蓋另一段其實不合的情況，所以要分開檢查。

    合併成功回傳擴大後的 (start_idx, end_idx)（連中間被排除的低曲率點
    也一併涵蓋進最終範圍）；不成立回傳 None，維持兩段分開。

    注意：此函式在 IMG_3294 測試中從未被觸發過（詳見
    detector/BAT_CLOUD_MERGE_NOTES.md），門檻數值未經真實資料驗證。
    """
    a_start, a_end = frag_a
    b_start, b_end = frag_b

    if seg[b_start].frame_index - seg[a_end].frame_index > constants.BAT_CLOUD_MERGE_MAX_GAP_FRAMES:
        return None
    if (a_end - a_start + 1) < constants.BAT_CLOUD_MERGE_MIN_FRAGMENT_POINTS:
        return None
    if (b_end - b_start + 1) < constants.BAT_CLOUD_MERGE_MIN_FRAGMENT_POINTS:
        return None

    combined = seg[a_start:b_end + 1]
    cx, cy, r, _ = _ransac_circle_fit(
        combined, frame_width, n_iter=constants.BAT_CLOUD_ARC_RANSAC_ITERATIONS,
    )
    tol = frame_width * constants.BAT_CLOUD_ARC_TOL_FRAME_WIDTH_RATIO

    def _inlier_ratio(lo: int, hi: int) -> float:
        pts = seg[lo:hi + 1]
        inliers = sum(1 for p in pts if abs(dist(p.x, p.y, cx, cy) - r) <= tol)
        return inliers / len(pts)

    min_inlier = constants.BAT_CLOUD_MERGE_MIN_INLIER_RATIO
    if _inlier_ratio(a_start, a_end) < min_inlier:
        return None
    if _inlier_ratio(b_start, b_end) < min_inlier:
        return None

    return a_start, b_end


def _cut_at_extremum(pts: list, start_i: int, end_i: int) -> int:
    """在 [start_i, end_i] 範圍內找離 pts[start_i] 距離最大的索引，當作
    延伸終點——揮棒延伸到底之後，收棒回拉會讓球棒開始靠近起點，離起點
    距離在延伸到底那一刻達到最大值、之後遞減，用這個全域最大值取代
    方向反轉角度門檻，不受局部雜訊誤觸發影響。
    """
    start_pt = pts[start_i]
    best_i, best_dist = end_i, -1.0
    for i in range(start_i, end_i + 1):
        d = dist(start_pt.x, start_pt.y, pts[i].x, pts[i].y)
        if d > best_dist:
            best_dist, best_i = d, i
    return best_i


def _find_swing_in_segment(
    seg: list, min_turn_deg: float, min_total_deg: float,
    min_speed: float, span_min: float, frame_width: int,
) -> tuple[float, int, int] | None:
    """在單一 Stage3 斷點區段內尋找最多一個揮棒視窗（峰值速度, 起幀, 訖幀）。

    找出所有連貫轉向片段 → 依序嘗試合併時間相鄰的片段（圓弧擬合倒推
    驗證是否為同一段揮棒被誤切，見 _try_merge_fragments）→ 每個（可能
    合併過的）候選都裁到延伸終點（自然裁掉收棒回拉）→ 累積轉角總量、
    空間跨距、峰值速度都需達門檻才接受 → 多個候選都通過時取峰值速度
    最高的一個。任一階段沒有候選存活就判定此區段沒有揮棒。
    """
    runs = _find_all_curve_runs(seg, min_turn_deg)
    if not runs:
        return None

    merged: list[tuple[int, int]] = [runs[0]]
    for frag in runs[1:]:
        result = _try_merge_fragments(seg, merged[-1], frag, frame_width)
        if result is not None:
            merged[-1] = result
        else:
            merged.append(frag)

    best_result: tuple[float, int, int] | None = None
    for start_i, raw_end_i in merged:
        extremum_i = _cut_at_extremum(seg, start_i, raw_end_i)
        window_pts = seg[start_i:extremum_i + 1]
        if len(window_pts) < 3:
            continue

        total_turn = sum(abs(t) for _, t in _turn_angles(window_pts))
        if total_turn < min_total_deg:
            continue

        xs = [p.x for p in window_pts]
        ys = [p.y for p in window_pts]
        span = max(max(xs) - min(xs), max(ys) - min(ys))
        if span < span_min:
            continue

        speeds_list = _smoothed_speeds(window_pts)
        if not speeds_list:
            continue
        peak_speed = max(s for _, s in speeds_list)
        if peak_speed < min_speed:
            continue

        if best_result is None or peak_speed > best_result[0]:
            best_result = (peak_speed, window_pts[0].frame_index, window_pts[-1].frame_index)

    return best_result


def _detect_swing_windows_shape(
    pts: list, fps: float, static_clusters: list | None = None,
    frame_width: int = 1920,
) -> list[tuple[int, int]]:
    """轉向幅度 + 方向一致性掃描 + 片段合併驗證，粗略框出揮棒區間（視覺
    化/診斷用，容許幀級誤差）。

    不假設路徑是任何全域形狀（圓、橢圓），直接用局部「每一步轉向角度」
    掃描：找出所有「轉向幅度夠大（不是隨便彎一點就算）且方向持續同一
    個正負號」的連續段（_find_all_curve_runs）。（曾經試過直接對整個
    區段做 RANSAC 圓弧擬合，實測發現對候選點集的微小變動極度敏感——
    margin 差 1 幀，coverage 從 307° 跳到 43°——根本原因是真實揮棒路徑
    是倒鉤/橢圓形，不是單一固定圓心的圓，對整個混雜區段硬套圓模型導致
    擬合本身不穩定，因此改用不假設全域形狀的局部轉向角掃描先粗略切段。）

    但單步零容忍的切段方式，可能把同一次揮棒因為單幀雜訊、偵測缺口、
    或路徑本身曲率天生較低的延伸段，誤切成好幾個獨立片段。因此在切段
    後，依序嘗試合併時間相鄰的片段（_try_merge_fragments）：對候選合併
    範圍跑 RANSAC 圓弧擬合，只有當兩個片段各自的內點率都達標，才判定
    它們其實是同一段揮棒被誤切、接受合併——這裡的圓弧擬合是「事後驗證
    一個已經先縮小範圍的具體候選」，跟之前放棄的「對整個區段擬合」是
    不同風險量級的用法。

    每個（可能經過合併的）候選，在段內找「離起點距離最大」的一幀當
    延伸終點（_cut_at_extremum），自然裁掉收棒回拉，不受局部雜訊誤
    觸發影響。

    先用 Stage 3 既有的 _segment_by_gap 把點雲切成真正的斷點區段；若有
    靜止叢集（持棒站姿等待，Stage 1）可用，只在「最後一個靜止叢集結束
    之後」的區段裡找揮棒——真正的下棒必然發生在打者定住站姿之後，站姿
    之前的區段（舉棒/走位等）直接排除候選資格。實測發現（IMG_3294）：
    打者舉棒等進站姿的準備動作，速度峰值可能跟真揮棒相差不到 5%，單靠
    速度量級無法可靠分開兩者，轉向幅度門檻本身也不一定擋得住（若準備
    動作剛好也是又快又緊的繞圈），此結構性過濾仍是必要的第一層防線。

    每個區段最多找一個候選視窗（_find_swing_in_segment：找連貫轉向段
    → 嘗試合併相鄰片段 → 裁到延伸終點 → 累積轉角總量、空間跨距、峰值
    速度都需達門檻才接受，多個候選都通過時取峰值最高的一個）。跨區段
    比較：所有區段的候選一起依峰值速度由高到低排序，第一個（全片段
    最高峰）當作主視窗；後續區段只有峰值達主視窗一定比例以上
    （BAT_CLOUD_SWING_SECONDARY_RATIO）才接受為真正的第二次揮棒——用來
    擋掉準備動作/晃動造成的假性次要視窗。

    輸入為 Stage 3 後（靜止叢集 + 孤立段落已移除）的 flight_pts，刻意
    不用 Stage 5' 之後的 clean_pts——MAD 離群值濾除可能誤刪揮棒本身的
    加速段。

    限制：
    - 僅驗證於側面攝影機視角
    - 起訖幀可能有數幀誤差，不保證精確切在下棒起點（可能含入 trigger
      準備動作），此為單點軌跡事後判別方法的固有模糊帶，非本方案獨有
      缺陷
    - 若同一斷點區段內真的連續發生兩次揮棒（中間沒有偵測缺幀或靜止
      叢集分隔），目前設計只會抓到該區段裡峰值最高的一次；此為配合
      「正常情況只會有一次揮棒」的簡化，非設計目標
    - 若打者從未進入偵測得到的靜止站姿（沒有靜止叢集）、且準備動作
      恰好又快又緊（同時滿足轉向幅度+方向一致+速度門檻），仍可能被
      誤判，此為已知殘留限制
    - 轉向角度/總量門檻、片段合併驗證的門檻數值未經大量資料校準，僅
      以現有常數量級當起點；片段合併這條路徑在 IMG_3294 測試中從未被
      觸發過，詳見 detector/BAT_CLOUD_MERGE_NOTES.md
    """
    _ = fps  # 縮放已套用至 BAT_CLOUD_TURN_* 常數
    segments = _segment_by_gap(sorted(pts, key=lambda p: p.frame_index))

    if static_clusters:
        stance_end = max(sc.end_frame for sc in static_clusters)
        segments = [seg for seg in segments if seg[-1].frame_index >= stance_end]

    min_turn_deg = constants.BAT_CLOUD_TURN_MIN_ANGLE_DEG
    min_total_deg = constants.BAT_CLOUD_TURN_MIN_TOTAL_DEG
    min_speed = constants.BAT_CLOUD_TURN_MIN_SPEED_PX
    span_min = constants.BAT_CLOUD_SWING_MIN_SPAN_PX
    secondary_ratio = constants.BAT_CLOUD_SWING_SECONDARY_RATIO

    candidates: list[tuple[float, int, int]] = []
    for seg in segments:
        if len(seg) < 5:
            continue
        result = _find_swing_in_segment(
            seg, min_turn_deg, min_total_deg, min_speed, span_min, frame_width,
        )
        if result is not None:
            candidates.append(result)

    candidates.sort(key=lambda t: -t[0])

    windows: list[tuple[int, int]] = []
    primary_peak: float | None = None
    for peak_speed, lo, hi in candidates:
        if primary_peak is not None and peak_speed < primary_peak * secondary_ratio:
            break
        windows.append((lo, hi))
        if primary_peak is None:
            primary_peak = peak_speed

    windows.sort()
    return windows


# ─── 主類別 ──────────────────────────────────────────────────────────


class BatCloudTracker:
    """Bat-top 點雲式軌跡過濾器。

    process() 需在呼叫前先對整支影片套用 constants.apply_fps_scaling(fps)，
    與 CloudBallTracker / BarrelTracker 的既有慣例一致。
    """

    def process(
        self, bat_top_positions: list, fps: float, frame_width: int,
    ) -> BatCloudTrackResult:
        candidates = sorted(bat_top_positions, key=lambda p: p.frame_index)

        static_clusters = _build_static_clusters(candidates)
        sparse, removed_static = _remove_static(candidates, static_clusters)

        flight_pts, removed_noise = _filter_noise_segments(sparse)

        # 揮棒視窗改用 flight_pts（Stage 5' 之前）——Stage 5' 的 MAD 離群值
        # 濾除可能誤刪揮棒本身的加速段，視窗偵測不該建立在被砍過的資料上
        swing_windows = _detect_swing_windows_shape(
            flight_pts, fps, static_clusters, frame_width,
        )

        clean_pts, removed_outlier = _filter_speed_outliers(flight_pts)

        return BatCloudTrackResult(
            positions=clean_pts,
            static_clusters=static_clusters,
            removed_static=removed_static,
            removed_noise=removed_noise,
            removed_outlier=removed_outlier,
            swing_windows=swing_windows,
        )
