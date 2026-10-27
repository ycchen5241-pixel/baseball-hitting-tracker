# tests/test_pose.py
# 打擊者姿態追蹤測試工具：頭部與軀幹移動軌跡視覺化
#
# 視覺標示：
#   黃色圓點 + 軌跡線  = 頭部（鼻子）移動路徑
#   青色圓點 + 軌跡線  = 軀幹中心移動路徑
#
# 使用方式：
#   py -3.12 tests/test_pose.py <video_path>
#
# 輸出：
#   <video_name>_pose.mp4

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from detector.pose_tracker import PoseTracker, PoseFrame  # noqa: E402

# ── 視覺設定 ──────────────────────────────────────────────────
COLOR_HEAD   = (0, 220, 255)    # 黃色：頭部
COLOR_TORSO  = (255, 220, 0)    # 青色：軀幹
DOT_RADIUS   = 6                # 當前位置圓點半徑
TRAIL_THICKNESS = 2             # 軌跡線粗細
TRAIL_MAX_LEN = 9999            # 保留全部軌跡（設小數字可限制長度）

# ── 穩定性判斷設定 ─────────────────────────────────────────────
# 連續偵測到且每幀位移 < STABLE_MAX_JUMP_PX，達 STABLE_FRAMES_REQUIRED 幀後才開始記錄
STABLE_FRAMES_REQUIRED = 15     # 需要連續幾幀穩定才開始記錄
STABLE_MAX_JUMP_PX     = 30.0   # 單幀最大允許位移（pixels）
INITIAL_SKIP_FRAMES    = 30     # 強制跳過最初 N 幀（MediaPipe 暖機期，不計入穩定性）


def _draw_trail(
    frame: np.ndarray,
    trail: list[tuple[int, int]],
    color: tuple[int, int, int],
) -> None:
    """繪製移動軌跡線與當前位置圓點。

    Args:
        frame: BGR 影像（就地修改）。
        trail: 像素座標序列 [(x, y), ...]，最後一點為當前位置。
        color: BGR 顏色。
    """
    if not trail:
        return

    # 軌跡線：依序連接所有歷史點
    for i in range(1, len(trail)):
        cv2.line(frame, trail[i - 1], trail[i], color, TRAIL_THICKNESS)

    # 當前位置：實心圓點（黑色外框增加對比）
    cx, cy = trail[-1]
    cv2.circle(frame, (cx, cy), DOT_RADIUS + 2, (0, 0, 0), -1)
    cv2.circle(frame, (cx, cy), DOT_RADIUS, color, -1)


def _draw_legend(frame: np.ndarray) -> None:
    """在右上角繪製圖例。"""
    items = [
        ("Head", COLOR_HEAD),
        ("Torso", COLOR_TORSO),
    ]
    x0 = frame.shape[1] - 130
    y0, lh = 20, 22

    for i, (name, color) in enumerate(items):
        y = y0 + i * lh
        cv2.circle(frame, (x0 + 6, y - 4), 6, color, -1)
        cv2.putText(frame, name, (x0 + 18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, name, (x0 + 18, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def _draw_frame_info(
    frame: np.ndarray,
    frame_index: int,
    total: int,
    stable_count: int,
    tracking: bool,
) -> None:
    """在左上角繪製幀號與穩定性狀態。"""
    if tracking:
        status = "TRACKING"
    elif stable_count > 0:
        status = f"STABILIZING ({stable_count}/{STABLE_FRAMES_REQUIRED})"
    else:
        status = "NO POSE"
    text = f"Frame {frame_index}/{total}  {status}"
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)


class _StabilityGate:
    """追蹤穩定性狀態，連續 N 幀穩定後才允許記錄軌跡。

    「穩定」定義：偵測成功 且 與上一幀位移 < STABLE_MAX_JUMP_PX。
    任一幀偵測失敗或位移過大，計數器歸零重新累積。
    一旦開始記錄（tracking=True）不再退回暖機狀態。
    """

    def __init__(self) -> None:
        self.count: int = 0           # 目前連續穩定幀數
        self.tracking: bool = False   # 是否已達到穩定、開始記錄
        self._last: tuple[float, float] | None = None

    def update(
        self,
        pt: tuple[float, float] | None,
        frame_index: int,
    ) -> bool:
        """更新穩定性狀態，回傳本幀是否應記錄到軌跡。

        Args:
            pt: 當前幀偵測到的座標 (x, y)，未偵測到則為 None。
            frame_index: 當前幀索引，用於強制跳過暖機期。

        Returns:
            True 表示已穩定，本幀座標可記入軌跡。
        """
        # 強制跳過最初 N 幀（MediaPipe 尚未鎖定目標）
        if frame_index < INITIAL_SKIP_FRAMES:
            return False

        if self.tracking:
            # 已穩定：偵測到就記錄，遺失時不退回暖機
            return pt is not None

        if pt is None:
            # 偵測失敗 → 重置計數器
            self.count = 0
            self._last = None
            return False

        if self._last is not None:
            dx = pt[0] - self._last[0]
            dy = pt[1] - self._last[1]
            jump = (dx * dx + dy * dy) ** 0.5
            if jump > STABLE_MAX_JUMP_PX:
                # 位移過大 → 重置計數器，以本幀為新起點
                self.count = 1
                self._last = pt
                return False

        self._last = pt
        self.count += 1

        if self.count >= STABLE_FRAMES_REQUIRED:
            self.tracking = True
            print(f"[pose] Tracking stable at frame {frame_index}")

        return self.tracking


def run_pose_test(video_path: str) -> None:
    """執行姿態軌跡視覺化主流程。"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[pose] {width}x{height} @ {fps:.0f}fps  {total} frames")

    output_path = str(Path(video_path).with_suffix("")) + "_pose.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker    = PoseTracker(fps=fps)
    head_gate  = _StabilityGate()
    torso_gate = _StabilityGate()
    head_trail:  list[tuple[int, int]] = []
    torso_trail: list[tuple[int, int]] = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        pose = tracker.update(frame, frame_index)

        head_pt  = (pose.head.x,  pose.head.y)  if pose.head  else None
        torso_pt = (pose.torso.x, pose.torso.y) if pose.torso else None

        # 強制跳過期 + 穩定確認後才記錄軌跡
        if head_gate.update(head_pt, frame_index) and head_pt is not None:
            head_trail.append((int(head_pt[0]), int(head_pt[1])))
            if len(head_trail) > TRAIL_MAX_LEN:
                head_trail.pop(0)

        if torso_gate.update(torso_pt, frame_index) and torso_pt is not None:
            torso_trail.append((int(torso_pt[0]), int(torso_pt[1])))
            if len(torso_trail) > TRAIL_MAX_LEN:
                torso_trail.pop(0)

        # 繪製（以 head 狀態為代表顯示穩定性）
        out = frame.copy()
        _draw_trail(out, torso_trail, COLOR_TORSO)
        _draw_trail(out, head_trail,  COLOR_HEAD)
        _draw_legend(out)
        _draw_frame_info(
            out, frame_index, total,
            stable_count=head_gate.count,
            tracking=head_gate.tracking,
        )

        writer.write(out)
        frame_index += 1

        if frame_index % 50 == 0:
            print(f"[pose] {frame_index}/{total} frames processed...")

    cap.release()
    writer.release()
    tracker.close()

    head_det  = len(head_trail)
    torso_det = len(torso_trail)
    print(f"\n[pose] Head  recorded : {head_det}/{total} "
          f"({head_det/total*100:.1f}%)")
    print(f"[pose] Torso recorded : {torso_det}/{total} "
          f"({torso_det/total*100:.1f}%)")
    print(f"[pose] Output saved   : {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.12 tests/test_pose.py <video_path>")
        print("Example: py -3.12 tests/test_pose.py Test_vod/793471920.220478.mp4")
        sys.exit(1)

    run_pose_test(sys.argv[1])
