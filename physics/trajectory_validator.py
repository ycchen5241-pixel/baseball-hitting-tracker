# physics/trajectory_validator.py
# 棒球運動公理驗證器
#
# 核心概念：不論中間使用何種偵測/過濾手段，最終完整軌跡必須符合
# 棒球運動的基本物理公理，否則整個偵測結果視為不可信。
#
# 球的運動公理（invariants）：
#   接觸前 —— 三種且只有三種合法模式：
#     TEE   : 球幾乎靜止（T座）
#     TOSS  : 短拋物線（手拋，低速）
#     PITCH : 幾乎水平前進（投手投球）—— 速度快 + 方向一致
#   接觸後 —— 只有一種合法模式：
#     HIT   : 無速度突刺 + 飛行方向一致（不隨機飛）
#
# 球棒運動公理：
#   SWING : 以人體為圓心的弧形掃過 + 角度單方向遞增（不在圓上跳點）

import math
from dataclasses import dataclass

import numpy as np

from config import constants
from detector.centroid import BallPosition


@dataclass
class PreContactMode:
    """接觸前球的運動模式分類結果。"""
    mode: str       # "TEE" | "TOSS" | "PITCH" | "UNKNOWN"
    confidence: float
    reason: str


@dataclass
class TrajectoryValidation:  # pylint: disable=too-many-instance-attributes
    """完整軌跡合法性驗證結果，含補救行動指示。"""
    pre_mode: PreContactMode
    post_valid: bool
    post_reason: str
    bat_valid: bool | None  # None 表示無棒軌跡資料，不做判斷
    bat_reason: str
    overall_valid: bool
    confidence: float           # 0.0 – 1.0：整體可信度
    spike_frames: list[int]     # 速度突刺的幀號，應從軌跡中移除後重新擬合
    detection_coverage: float   # 真實偵測幀 / 總軌跡幀（< COVERAGE_MIN 代表主要是 Kalman 空預測）
    suppress_trajectory: bool   # True → 不畫球軌跡線（資料品質不足）
    suppress_bat: bool          # True → 不畫棒軌跡線（偵測到靜止背景物非真實揮棒）

    def log(self) -> None:
        """印出驗證摘要至 stdout。"""
        status = "VALID" if self.overall_valid else "INVALID"
        print(
            f"[validator] Trajectory: {status}"
            f"  confidence={self.confidence:.2f}"
            f"  coverage={self.detection_coverage:.0%}"
        )
        print(
            f"[validator]   pre-contact mode : {self.pre_mode.mode}"
            f"  ({self.pre_mode.reason})"
        )
        post_status = "OK" if self.post_valid else "FAIL"
        print(
            f"[validator]   post-contact     : {post_status}"
            f"  {self.post_reason}"
        )
        if self.spike_frames:
            print(
                f"[validator]   spike frames     :"
                f" {self.spike_frames} → will be removed"
            )
        if self.suppress_trajectory:
            print("[validator]   → trajectory suppressed (low coverage)")
        if self.suppress_bat:
            print("[validator]   → bat trajectory suppressed (arc invalid)")
        if self.bat_valid is not None:
            bat_status = "OK" if self.bat_valid else "FAIL"
            print(
                f"[validator]   bat swing        : {bat_status}"
                f"  {self.bat_reason}"
            )


# ── 內部工具 ──────────────────────────────────────────────────────

def _per_frame_speeds(positions: list[BallPosition]) -> list[float]:
    """計算相鄰幀間每幀平均速度（px / frame）。"""
    speeds = []
    for i in range(1, len(positions)):
        dx = positions[i].x - positions[i - 1].x
        dy = positions[i].y - positions[i - 1].y
        dt = max(1, positions[i].frame_index - positions[i - 1].frame_index)
        speeds.append(math.sqrt(dx ** 2 + dy ** 2) / dt)
    return speeds


def _direction_circular_std(positions: list[BallPosition]) -> float | None:
    """計算相鄰幀位移向量的圓形標準差（degrees）。

    圓形標準差不受 0°/360° 斷點影響，能正確量測方向集中程度：
      R = |mean of unit vectors|   （R 越接近 1 越集中）
      circular_std = sqrt(-2 · ln(R))（單位：radians，轉 degrees）

    Returns:
        圓形標準差（degrees）；可移動的位移不足 2 個時回傳 None。
    """
    angles = []
    for i in range(1, len(positions)):
        dx = positions[i].x - positions[i - 1].x
        dy = positions[i].y - positions[i - 1].y
        dt = max(1, positions[i].frame_index - positions[i - 1].frame_index)
        speed = math.sqrt(dx ** 2 + dy ** 2) / dt
        if speed < 1e-3:
            continue
        angles.append(math.atan2(dy, dx))

    if len(angles) < 2:
        return None

    sin_m = float(np.mean(np.sin(angles)))
    cos_m = float(np.mean(np.cos(angles)))
    r_val = math.sqrt(sin_m ** 2 + cos_m ** 2)
    if r_val < 1e-9:
        return 180.0  # 完全隨機方向
    return math.degrees(math.sqrt(-2.0 * math.log(r_val)))


# ── 接觸前模式分類 ────────────────────────────────────────────────

def _classify_pre_contact(positions: list[BallPosition]) -> PreContactMode:
    """分類接觸前球的運動模式：TEE / TOSS / PITCH / UNKNOWN。

    PITCH 需同時滿足：速度夠快 + 方向一致（方向隨機的高速不是投球）。
    """
    if len(positions) < 2:
        return PreContactMode("UNKNOWN", 0.0, "insufficient points")

    speeds = _per_frame_speeds(positions)
    median_s = float(np.median(speeds)) if speeds else 0.0

    if median_s < constants.VALIDATOR_TEE_MAX_SPEED_PX:
        return PreContactMode(
            "TEE", 0.9,
            f"median speed {median_s:.1f} px/f (near static)",
        )

    if median_s >= constants.VALIDATOR_PITCH_MIN_SPEED_PX:
        dir_std = _direction_circular_std(positions)
        max_std = constants.VALIDATOR_PITCH_DIRECTION_STD_MAX_DEG
        if dir_std is None or dir_std <= max_std:
            conf = min(1.0, median_s / 80.0)
            std_info = f"{dir_std:.1f}°" if dir_std is not None else "N/A"
            return PreContactMode(
                "PITCH", conf,
                f"speed {median_s:.1f} px/f, dir_std {std_info}",
            )
        # 速度夠但方向亂 → 不是合法的投球
        return PreContactMode(
            "UNKNOWN", 0.1,
            f"speed ok but direction scattered"
            f" (std {dir_std:.1f}° > {max_std}°)",
        )

    return PreContactMode(
        "TOSS", 0.7,
        f"median speed {median_s:.1f} px/f (slow but moving)",
    )


# ── 接觸後驗證 ────────────────────────────────────────────────────

def _check_speed_spike(
    positions: list[BallPosition],
    contact_fi: int | None = None,
) -> tuple[bool, str, list[int]]:
    """速度突刺：任何幀速度 > 中位速度 × SPIKE_RATIO → FAIL。

    抓住「兩靜止球誤連」bug：兩球之間必定有一幀產生極大速度突刺。

    contact_fi 不為 None 時啟用「接觸鄰近豁免」：
    接觸後前 VALIDATOR_CONTACT_EXEMPT_FRAMES 幀同時排除在：
      1. 中位數計算之外：避免後段近零假偵測壓低基準
      2. 突刺偵測之外：擊球瞬間球速驟升是物理預期，不應被標記

    Returns:
        (valid, reason, spike_frame_indices) —— 突刺幀號供呼叫端移除後重擬合。
    """
    if len(positions) < 3:
        return True, "skip (< 3 points)", []

    speeds = _per_frame_speeds(positions)
    if not speeds:
        return True, "skip (no speed data)", []

    # 建立豁免幀集合（接觸後前 K 幀）
    exempt: set[int] = set()
    if contact_fi is not None:
        k = constants.VALIDATOR_CONTACT_EXEMPT_FRAMES
        exempt = {
            positions[i + 1].frame_index
            for i, _ in enumerate(speeds)
            if positions[i + 1].frame_index <= contact_fi + k
        }

    # 中位數只用非豁免幀速度計算，避免被後段近零假偵測主導
    non_exempt_speeds = [
        s for i, s in enumerate(speeds)
        if positions[i + 1].frame_index not in exempt
    ]
    if not non_exempt_speeds:
        return True, "skip (all frames in exempt window)", []

    median_s = float(np.median(non_exempt_speeds))
    if median_s < 1e-3:
        return True, "skip (near-zero speed)", []

    ratio = constants.VALIDATOR_SPEED_SPIKE_RATIO
    spike_frames = [
        positions[i + 1].frame_index
        for i, s in enumerate(speeds)
        if s > median_s * ratio
        and positions[i + 1].frame_index not in exempt
    ]
    if spike_frames:
        first_idx = next(
            i for i, s in enumerate(speeds)
            if s > median_s * ratio
            and positions[i + 1].frame_index not in exempt
        )
        first_s = speeds[first_idx]
        return (
            False,
            f"speed spike at f{spike_frames[0]}: {first_s:.1f} px/f"
            f" > median {median_s:.1f} × {ratio}",
            spike_frames,
        )
    exempt_note = (
        f" (exempt: {len(exempt)} contact-adj frames)" if exempt else ""
    )
    return True, f"ok (median {median_s:.1f} px/f{exempt_note})", []


def _detection_coverage(positions: list[BallPosition]) -> float:
    """接觸後軌跡中真實偵測（confidence > 0）的比例。

    Kalman 純預測幀的 confidence = 0；比例偏低代表軌跡主要由空預測組成。
    """
    if not positions:
        return 0.0
    real = sum(1 for p in positions if p.confidence > 0)
    return real / len(positions)


def _check_post_direction(
    positions: list[BallPosition],
) -> tuple[bool, str]:
    """方向一致性：接觸後飛行方向的圓形標準差超過閾值 → FAIL。

    速度突刺只看大小，方向隨機飛同樣不合法：
    球打出後應持續朝同一方向飛行，方向散佈不應過大。
    """
    min_pts = constants.VALIDATOR_POST_MIN_POINTS
    if len(positions) < min_pts:
        return True, f"skip (only {len(positions)} pts, need {min_pts})"

    dir_std = _direction_circular_std(positions)
    if dir_std is None:
        return True, "skip (insufficient movement)"

    max_std = constants.VALIDATOR_POST_DIRECTION_STD_MAX_DEG
    if dir_std > max_std:
        return (
            False,
            f"direction scattered: std {dir_std:.1f}° > {max_std:.1f}°"
            " (ball flying in random directions)",
        )
    return True, f"direction std {dir_std:.1f}°"


def _validate_post_contact(
    positions: list[BallPosition],
    contact_fi: int | None = None,
) -> tuple[bool, str, list[int]]:
    """速度突刺 + 方向一致性，兩者皆通過才合法。

    Returns:
        (valid, reason, spike_frames)
    """
    spike_ok, spike_msg, spike_frames = _check_speed_spike(
        positions, contact_fi,
    )
    if not spike_ok:
        return False, spike_msg, spike_frames

    dir_ok, dir_msg = _check_post_direction(positions)
    if not dir_ok:
        return False, dir_msg, []

    return True, dir_msg, []


# ── 球棒弧形驗證 ───────────────────────────────────────────────────

def _fit_circle_lsq(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[float, float, float] | None:
    """代數法最小二乘圓擬合，回傳 (cx, cy, r)；退化時回傳 None。"""
    mat = np.column_stack([x, y, np.ones(len(x))])
    b_vec = x ** 2 + y ** 2
    coeffs, _, _, _ = np.linalg.lstsq(mat, b_vec, rcond=None)
    cx = coeffs[0] / 2.0
    cy = coeffs[1] / 2.0
    r_sq = coeffs[2] + cx ** 2 + cy ** 2
    if r_sq <= 0:
        return None
    return cx, cy, math.sqrt(r_sq)


def _validate_bat_arc(
    bat_positions: list[BallPosition],
) -> tuple[bool, str]:
    """球棒弧形驗證：圓形 RMSE + 角度單向遞增。

    支援棒頭（confidence > 0）與棒身補全（confidence = 0）混合序列：
    - 弧形 RMSE：只用高信心度的棒頭點計算，確保圓弧品質不被不同半徑的棒身點干擾
    - 角度單向遞增：使用所有點（含棒身補全），確認整段揮棒方向一致
    """
    min_pts = constants.VALIDATOR_BAT_MIN_POINTS
    if len(bat_positions) < min_pts:
        return True, f"skip (only {len(bat_positions)} pts, need {min_pts})"

    # 弧形 RMSE：只用棒頭點（高信心度）
    head_pts = [p for p in bat_positions if p.confidence > 0]
    fit_pts = head_pts if len(head_pts) >= min_pts else bat_positions
    x_fit = np.array([p.x for p in fit_pts], dtype=np.float64)
    y_fit = np.array([p.y for p in fit_pts], dtype=np.float64)

    circle = _fit_circle_lsq(x_fit, y_fit)
    if circle is None:
        return False, "degenerate circle fit (r² ≤ 0)"
    cx, cy, r = circle

    radii = np.sqrt((x_fit - cx) ** 2 + (y_fit - cy) ** 2)
    rmse = float(np.sqrt(np.mean((radii - r) ** 2)))
    max_rmse = constants.VALIDATOR_BAT_ARC_RMSE_MAX_PX
    if rmse > max_rmse:
        return False, f"arc RMSE {rmse:.1f}px > {max_rmse:.1f}px"

    # 角度單向遞增：使用全部點（含棒身補全點）
    x_all = np.array([p.x for p in bat_positions], dtype=np.float64)
    y_all = np.array([p.y for p in bat_positions], dtype=np.float64)
    raw_angles = np.arctan2(y_all - cy, x_all - cx)
    unwrapped = np.unwrap(raw_angles)
    diffs = np.diff(unwrapped)

    dominant = 1 if np.sum(diffs > 0) >= np.sum(diffs < 0) else -1
    monotone_count = int(np.sum(diffs * dominant > 0))
    monotone_ratio = monotone_count / len(diffs)

    min_ratio = constants.VALIDATOR_BAT_MONOTONE_MIN_RATIO
    if monotone_ratio < min_ratio:
        n_body = len(bat_positions) - len(head_pts)
        return (
            False,
            f"arc traversal non-monotone: {monotone_ratio:.0%}"
            f" < {min_ratio:.0%}"
            f" (head={len(head_pts)}, body_supplement={n_body})",
        )

    direction = "CW" if dominant < 0 else "CCW"
    n_body = len(bat_positions) - len(head_pts)
    supplement_info = f" +{n_body}body" if n_body else ""
    return (
        True,
        f"arc RMSE {rmse:.1f}px, r={r:.0f}px,"
        f" monotone {monotone_ratio:.0%} ({direction})"
        f" [{len(head_pts)}head{supplement_info}]",
    )


# ── 公開介面 ───────────────────────────────────────────────────────

def validate_trajectory(
    pre_positions: list[BallPosition],
    post_positions: list[BallPosition],
    bat_positions: list[BallPosition] | None = None,
    contact_fi: int | None = None,
) -> TrajectoryValidation:
    """棒球運動公理驗證器：對完整偵測軌跡做最終合法性判斷。

    此函式不是逐幀篩選，而是在所有偵測與過濾完成後，
    以整段軌跡對棒球物理公理進行驗證。

    Args:
        pre_positions : 接觸前球位置（按 frame_index 升序排序）。
        post_positions: 接觸後球位置（按 frame_index 升序排序）。
        bat_positions : 棒頭位置（可選；None 則跳過球棒驗證）。

    Returns:
        TrajectoryValidation，含模式分類、各段驗證與整體可信度。
        呼叫 .log() 可印出摘要。
    """
    pre_mode = _classify_pre_contact(pre_positions)
    post_valid, post_reason, spike_frames = _validate_post_contact(
        post_positions, contact_fi,
    )
    coverage = _detection_coverage(post_positions)

    bat_valid: bool | None = None
    bat_reason = "no bat data"
    if bat_positions:
        bat_valid, bat_reason = _validate_bat_arc(bat_positions)

    overall_valid = post_valid and (bat_valid is None or bat_valid)

    conf = pre_mode.confidence
    if not post_valid:
        conf *= 0.2
    if bat_valid is False:
        conf *= 0.5

    suppress_traj = (
        not post_valid
        or coverage < constants.VALIDATOR_COVERAGE_MIN
    )
    suppress_bat = bat_valid is False

    return TrajectoryValidation(
        pre_mode=pre_mode,
        post_valid=post_valid,
        post_reason=post_reason,
        bat_valid=bat_valid,
        bat_reason=bat_reason,
        overall_valid=overall_valid,
        confidence=conf,
        spike_frames=spike_frames,
        detection_coverage=coverage,
        suppress_trajectory=suppress_traj,
        suppress_bat=suppress_bat,
    )
