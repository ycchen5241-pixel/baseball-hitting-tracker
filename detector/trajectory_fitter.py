# detector/trajectory_fitter.py
# 軌跡擬合：離群值篩除 + 軌跡段辨識（來球段/飛行段/反彈段）
#
# 取代 CloudBallTracker 的 Stage 5（全域速度 MAD）與飛行段落截斷
# （_truncate_at_impact 三點夾角測試）。不依賴 contact_frame——各段落
# 交界時間點由擬合過程本身推導，Stage 6 的 contact_frame 只作為事後
# 交叉驗證的獨立第二意見，不是這裡的輸入依賴。
#
# 核心方法：Sequential RANSAC 遞迴擬合
#   對候選點反覆做穩健擬合（先試直線，找不到夠緊密共識再試二次曲線）
#   → 每輪取出共識點集，組成一條「已確認線段」
#   → 候選集合扣掉共識點，繼續對剩餘點嘗試擬合下一條線段
#   → 直到候選點數低於統計可靠下限、或兩種階數都找不到共識為止
#
# 段落角色依因果時間序判斷：來球段（可選）→ 飛行段 → 反彈段（可選），
# 不使用速度量級。

import math
import statistics
from dataclasses import dataclass, field

import numpy as np

from config import constants
from detector.centroid import BallPosition

# MAD → 標準差一致性常數：在常態分布假設下 sigma ≈ 1.4826 × MAD，
# 統計慣例換算值（非本專案調校參數），讓 MAD 可以跟 K 倍標準差門檻互相比較。
_MAD_TO_SIGMA = 1.4826


@dataclass
class FittedSegment:
    """一條穩健擬合出的軌跡線段。"""

    points: list             # list[BallPosition]，依 frame_index 排序的共識點
    order: int                # 擬合階數：1=直線，2=二次曲線
    coeffs_x: tuple           # np.polyfit 係數：x(t)，t=frame_index
    coeffs_y: tuple           # np.polyfit 係數：y(t)
    role: str = "unclassified"  # 'incoming' / 'flight' / 'bounce' / 'unclassified'

    @property
    def start_frame(self) -> int:
        return self.points[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.points[-1].frame_index


@dataclass
class TrajectoryFitResult:
    """軌跡擬合的完整輸出。"""

    segments: list = field(default_factory=list)   # list[FittedSegment]，依時間排序
    discarded: list = field(default_factory=list)   # list[BallPosition]，無法分類的殘留雜訊

    @property
    def flight_positions(self) -> list:
        """飛行段的乾淨點，供下游物理計算（EV/LA）使用。"""
        for seg in self.segments:
            if seg.role == "flight":
                return seg.points
        return []

    @property
    def boundary_frames(self) -> dict:
        """段落交界時間點：{'contact': 來球→飛行, 'impact': 飛行→反彈}。

        contact 優先以來球段/飛行段兩條擬合曲線的外插交點求解（見
        solve_segment_intersection、PHYSICS_VALIDATION_PRINCIPLES.md C5），
        還原動態模糊造成的缺偵測間隙；解不出時退回飛行段起始幀。
        """
        result: dict = {"contact": None, "impact": None}
        by_role = {seg.role: seg for seg in self.segments}
        if "incoming" in by_role and "flight" in by_role:
            incoming, flight = by_role["incoming"], by_role["flight"]
            t = solve_segment_intersection(incoming, flight)
            result["contact"] = t if t is not None else flight.start_frame
        if "flight" in by_role and "bounce" in by_role:
            result["impact"] = by_role["bounce"].start_frame
        return result


# ─── 穩健擬合核心 ────────────────────────────────────────────────────


def _fit_exact(pts: list, order: int) -> tuple:
    """最小平方解出 x(t)、y(t) 的 order 階多項式係數。

    點數等於 order+1 時為精確解（RANSAC 最小子集）；
    點數更多時為最小平方擬合（收斂後的最終線段）。
    """
    t = np.array([p.frame_index for p in pts], dtype=np.float64)
    x = np.array([p.x for p in pts], dtype=np.float64)
    y = np.array([p.y for p in pts], dtype=np.float64)
    coeffs_x = np.polyfit(t, x, order)
    coeffs_y = np.polyfit(t, y, order)
    return coeffs_x, coeffs_y


def _residual_2d(p, coeffs_x, coeffs_y) -> float:
    """點到擬合曲線的 2D 歐氏距離殘差（x、y 一起算，不分軸）。"""
    pred_x = float(np.polyval(coeffs_x, p.frame_index))
    pred_y = float(np.polyval(coeffs_y, p.frame_index))
    return math.hypot(p.x - pred_x, p.y - pred_y)


def _ransac_fit(candidates: list, order: int, rng: np.random.Generator):
    """RANSAC：隨機取樣最小子集，找支持度最高的 order 階多項式模型。

    沿用專案既有慣例（見 physics/trajectory.py::_ransac_circle_fit）：
    固定種子隨機取樣、n_iter 次迭代，結果可重現。

    Returns:
        (inlier_pts, coeffs_x, coeffs_y) 或 None（找不到夠格的共識）。
    """
    n = len(candidates)
    k = order + 1
    if n < max(k, constants.TRAJFIT_MIN_CONSENSUS_POINTS):
        return None

    tol = constants.TRAJFIT_RANSAC_TOL_PX
    best_idx: list = []
    for _ in range(constants.TRAJFIT_RANSAC_ITERATIONS):
        idx = rng.choice(n, k, replace=False)
        subset = [candidates[i] for i in idx]
        if len({p.frame_index for p in subset}) < k:
            continue  # 同一幀時間重複，無法唯一解多項式
        coeffs_x, coeffs_y = _fit_exact(subset, order)
        inlier_idx = [
            i for i, p in enumerate(candidates)
            if _residual_2d(p, coeffs_x, coeffs_y) <= tol
        ]
        if len(inlier_idx) > len(best_idx):
            best_idx = inlier_idx

    if len(best_idx) < constants.TRAJFIT_MIN_CONSENSUS_POINTS:
        return None

    inlier_pts = [candidates[i] for i in best_idx]
    coeffs_x, coeffs_y = _fit_exact(inlier_pts, order)
    return inlier_pts, coeffs_x, coeffs_y


def _refine_by_mad(pts: list, order: int) -> list:
    """用這批共識點自己的 median+K×MAD 殘差門檻做最後收斂。

    殘差量的是「軌跡彎不彎、跳不跳」，不受整體移動速度快慢系統性
    影響，用全域統計量篩選不會重演 Stage 5 舊機制「快速段被當少數
    污染誤殺」的問題。
    """
    fit_order = min(order, len(pts) - 1)
    coeffs_x, coeffs_y = _fit_exact(pts, fit_order)
    residuals = [_residual_2d(p, coeffs_x, coeffs_y) for p in pts]

    med = statistics.median(residuals)
    mad = statistics.median([abs(r - med) for r in residuals])
    thr = max(
        med + constants.TRAJFIT_RESIDUAL_MAD_K * _MAD_TO_SIGMA * mad,
        constants.TRAJFIT_RESIDUAL_MIN_THRESHOLD_PX,
    )

    refined = [p for p, r in zip(pts, residuals) if r <= thr]
    if len(refined) <= fit_order:
        return pts
    return refined


# ─── 外插求交點（PHYSICS_VALIDATION_PRINCIPLES.md C5） ────────────────
#
# 動態模糊造成缺偵測間隙時，段落邊界不能直接採用缺口邊緣的偵測幀
# （誤差等於缺口長度），改用缺口兩側的擬合曲線往缺口方向外插，解交點。
# 只解 x 軸（水平方向優先，見 B3）。


def solve_poly_roots_near(coeffs, lo: int, hi: int) -> int | None:
    """解多項式零點，回傳離 [lo, hi] 這個缺口範圍最近的實根（四捨五入為
    幀號）；無實根則回傳 None。

    不設外插距離上限（與使用者討論 2026-07-23：短距離擊球軌跡近乎直線，
    多重根極少發生，不需要用距離上限過濾）。若仍出現多個實根，取離
    [lo, hi] 最近者，而非單純取最小值——避免真的出現多重根時選到離缺口
    很遠的無關解。

    兩段曲線在時間軸上完全相同（係數差全為零）時無唯一解，回傳 None。
    """
    if np.allclose(coeffs, 0):
        return None
    roots = np.roots(coeffs)
    real_roots = [r.real for r in roots if abs(r.imag) < 1e-6]
    if not real_roots:
        return None

    range_lo, range_hi = min(lo, hi), max(lo, hi)

    def _dist_to_range(t: float) -> float:
        # 呼叫端傳入的 lo/hi 在沒有缺口（兩段重疊）時可能是反過來的，
        # 這裡先排序，讓「是否落在觀測重疊範圍內」判斷不受呼叫順序影響。
        if range_lo <= t <= range_hi:
            return 0.0
        return min(abs(t - range_lo), abs(t - range_hi))

    best = min(real_roots, key=_dist_to_range)
    return round(best)


def solve_segment_intersection(seg_a: FittedSegment, seg_b: FittedSegment) -> int | None:
    """解兩段擬合曲線 x(t) 相等的時間點，供缺偵測間隙下還原真正的段落
    交界時刻，取代直接採用缺口邊緣的偵測幀。

    只解 x 軸，但用 y 軸做外插可信度驗證：解出的 t 代回兩段各自的
    y(t)，應彼此吻合（在 RANSAC 本身的內點容忍值內）——因為兩段曲線
    宣稱在這一刻交會，x、y 應同時成立，不能只挑 x 軸湊巧吻合。若 y
    差距過大，代表擬合曲線外插到這個時間點已經跟真實軌跡脫鉤，此交點
    不可採信，回傳 None。
    """
    diff = np.polysub(seg_a.coeffs_x, seg_b.coeffs_x)
    t = solve_poly_roots_near(diff, seg_a.end_frame, seg_b.start_frame)
    if t is None:
        return None
    y_a = np.polyval(seg_a.coeffs_y, t)
    y_b = np.polyval(seg_b.coeffs_y, t)
    if abs(y_a - y_b) > constants.TRAJFIT_RANSAC_TOL_PX:
        return None
    return t


# ─── 單一線段合格判斷 ────────────────────────────────────────────────


def _passes_segment_checks(pts: list) -> bool:
    """有沒有真的在動 + 密度跨度相稱 + 內部連續性 + 水平方向全程單調。

    曲率合理性（短距離場景該接近直線，明顯曲率為非自然軌跡）已經
    隱含在「先試直線、直線擬合不了才試二次曲線」的呼叫順序裡
    （見 _find_next_segment），不需要在這裡另外檢查。

    空間跨度檢查回答「這批點有沒有真的在動」，跟後面密度/間距/單調性
    三項（回答「動得合不合理」）是互補而非重複：近乎靜止的殘留點群
    抖動幅度通常不到 TRAJFIT_MONOTONIC_MIN_DX_PX，密度/間距/單調性
    三項對這種情況是盲的，必須靠總跨度直接把關。
    """
    n = len(pts)
    if n < constants.TRAJFIT_MIN_CONSENSUS_POINTS:
        return False

    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    spread = max(max(xs) - min(xs), max(ys) - min(ys))
    if spread < constants.TRAJFIT_MIN_SPATIAL_SPREAD_PX:
        return False

    span = max(1, pts[-1].frame_index - pts[0].frame_index)
    if n / span < constants.TRAJFIT_MIN_DENSITY_RATE:
        return False

    max_gap = max(
        pts[i + 1].frame_index - pts[i].frame_index for i in range(n - 1)
    )
    if max_gap > constants.CLOUD_GAP_THRESHOLD_FRAMES:
        return False

    dx_signs = set()
    for i in range(n - 1):
        dx = pts[i + 1].x - pts[i].x
        if abs(dx) > constants.TRAJFIT_MONOTONIC_MIN_DX_PX:
            dx_signs.add(1 if dx > 0 else -1)
    if len(dx_signs) > 1:
        return False

    return True


def _find_next_segment(candidates: list, rng: np.random.Generator):
    """對候選集合嘗試找出下一條線段：先試直線，找不到再試二次曲線。

    直線擬合不了、必須用明顯曲率才能解釋這批點時，才接受二次曲線
    ——這代表短距離場景下的非自然軌跡（通常是撞網反彈），仍然是
    一條合格的線段，只是不會是「飛行段」的角色（見 _assign_roles）。

    找到 RANSAC 共識、但沒通過合格判斷（例如 Stage 1 沒抓乾淨、殘留
    的靜止污染叢集——內部一致但時間跨度稀疏）時，仍然要把這批點從
    候選集合移除再回傳。這批點已經證明是一個獨立、自洽的子群，不是
    孤立雜訊；留著不移除，它會在下一輪重新贏得 RANSAC 多數決，卡住
    遞迴，讓後面真正的飛行段永遠沒機會被搜尋到。

    Returns:
        (FittedSegment | None, remaining_candidates)：
          - seg 不是 None：找到合格線段，remaining 已扣掉這批點
          - seg 是 None、remaining 不是 None：找到共識但不合格，
            remaining 已扣掉這批不合格的點，應該繼續搜尋
          - remaining 是 None：兩種階數都連 RANSAC 共識都找不到，
            候選點裡沒有任何一致子集，遞迴該真正停止
    """
    for order in (1, 2):
        result = _ransac_fit(candidates, order, rng)
        if result is None:
            continue
        inlier_pts, _, _ = result
        refined_pts = _refine_by_mad(inlier_pts, order)
        refined_keys = {(p.frame_index, p.x, p.y) for p in refined_pts}
        remaining = [
            p for p in candidates
            if (p.frame_index, p.x, p.y) not in refined_keys
        ]
        if not _passes_segment_checks(refined_pts):
            return None, remaining
        fit_order = min(order, len(refined_pts) - 1)
        coeffs_x, coeffs_y = _fit_exact(refined_pts, fit_order)
        seg = FittedSegment(
            points=sorted(refined_pts, key=lambda p: p.frame_index),
            order=order,
            coeffs_x=coeffs_x,
            coeffs_y=coeffs_y,
        )
        return seg, remaining
    return None, None


# ─── 段落角色辨識（時間序 + 因果關係，不用速度量級） ──────────────────


def _direction_sign(seg: FittedSegment) -> int:
    dx = seg.points[-1].x - seg.points[0].x
    return 1 if dx >= 0 else -1


def _assign_roles(segments: list, scenario: str) -> None:
    """依時間序 + 因果關係貼標籤：來球段（可選）→ 飛行段 → 反彈段（可選）。

    tstand 情境不會有來球段——接觸前球是靜止的，早在 Stage 1/2 就被
    移除，不會進到候選點裡，時間序最早的線段直接就是飛行段。

    pitch/sidetoss 情境下，若時間序最早的線段是直線、且跟下一段的
    水平方向相反（球飛來又被打出去），視為來球段；其餘情況時間序
    最早的線段一律視為飛行段。飛行段之後的所有線段都是反彈段（可能
    因為近距離反彈而不只一段，但都歸類為同一角色，不細分內部結構，
    呼應「撞網後行為忽略」的既有原則）。
    """
    if not segments:
        return

    start_idx = 0
    if scenario == "pitch" and len(segments) >= 2:
        first, second = segments[0], segments[1]
        if first.order == 1 and _direction_sign(first) != _direction_sign(second):
            first.role = "incoming"
            start_idx = 1

    segments[start_idx].role = "flight"
    for seg in segments[start_idx + 1:]:
        seg.role = "bounce"


# ─── 主流程 ──────────────────────────────────────────────────────────


def fit_trajectory_segments(
    flight_pts: list, scenario: str,
) -> TrajectoryFitResult:
    """對 Stage 1-3 清完的候選點做 Sequential RANSAC 遞迴擬合。

    輸出乾淨的線段列表與角色標籤（來球段/飛行段/反彈段），取代
    CloudBallTracker 的 Stage 5 MAD 與 _truncate_at_impact。

    Args:
        flight_pts: Stage 1-3 清完的候選點（list[BallPosition]）。
        scenario: Stage 4 輸出的情境判斷（'tstand'/'pitch'/'sidetoss'/'unknown'）。

    Returns:
        TrajectoryFitResult，包含依時間排序的線段列表與無法分類的殘留點。
    """
    candidates = sorted(flight_pts, key=lambda p: p.frame_index)
    rng = np.random.default_rng(constants.TRAJFIT_RANSAC_SEED)
    segments: list = []
    discarded: list = []

    while len(candidates) >= constants.TRAJFIT_MIN_CANDIDATES_FOR_FIT:
        seg, remaining = _find_next_segment(candidates, rng)
        if remaining is None:
            break  # 兩種階數都找不到任何一致子集，候選池裡沒東西可挖了
        removed_this_round = [
            p for p in candidates
            if p not in remaining
        ] if seg is None else []
        if seg is not None:
            segments.append(seg)
        else:
            # 找到共識但不合格：這批點不是孤立雜訊而是一個自洽子群
            # （例如 Stage 1 沒抓乾淨的殘留靜止污染），移除避免卡住
            # 遞迴，但仍要保留在 discarded 裡供後續診斷/複核
            discarded.extend(removed_this_round)
        candidates = remaining  # 不論合不合格，這批點都已處理，繼續挖剩下的

    discarded.extend(candidates)  # 迴圈結束時仍剩下的，一併算作無法分類

    segments.sort(key=lambda s: s.start_frame)
    _assign_roles(segments, scenario)
    return TrajectoryFitResult(segments=segments, discarded=discarded)
