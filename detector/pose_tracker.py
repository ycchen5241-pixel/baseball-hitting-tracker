# detector/pose_tracker.py
# MediaPipe Pose Landmarker 封裝（Tasks API，0.10.x 適用）
# 擷取頭部（鼻子）與軀幹中心的像素座標

from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 模型檔路徑（相對於專案根目錄）
_MODEL_PATH = str(Path(__file__).parent.parent / "Training" / "pose_landmarker_lite.task")

# 關鍵點索引（與原始 MediaPipe 33點順序相同）
_NOSE           = 0
_LEFT_SHOULDER  = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP       = 23
_RIGHT_HIP      = 24

MIN_VISIBILITY = 0.5   # 可見度低於此值的點不使用


@dataclass
class PosePoint:
    """單幀中一個追蹤點的像素座標。"""
    x: float
    y: float


@dataclass
class PoseFrame:
    """單幀的姿態追蹤結果。"""
    frame_index: int
    head: PosePoint | None      # 鼻子（頭部代表點）
    torso: PosePoint | None     # 肩膀 + 髖部四點中心（軀幹中心）


class PoseTracker:
    """MediaPipe Pose Landmarker 追蹤器（VIDEO 模式，逐幀追蹤）。"""

    def __init__(self, fps: float = 30.0) -> None:
        self._fps = fps
        base_options = python.BaseOptions(model_asset_path=_MODEL_PATH)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = vision.PoseLandmarker.create_from_options(options)

    def update(self, frame: np.ndarray, frame_index: int) -> PoseFrame:
        """對單幀執行姿態偵測，回傳頭部與軀幹中心座標。

        Args:
            frame: BGR 影像陣列（OpenCV 格式）。
            frame_index: 當前幀索引（從 0 開始）。

        Returns:
            PoseFrame，偵測失敗時 head / torso 為 None。
        """
        h, w = frame.shape[:2]
        rgb = frame[:, :, ::-1].copy()   # BGR → RGB，需連續記憶體
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO 模式需傳入嚴格遞增的時間戳（毫秒）
        timestamp_ms = int(frame_index * 1000 / self._fps)
        result = self._detector.detect_for_video(mp_image, timestamp_ms)

        if not result.pose_landmarks:
            return PoseFrame(frame_index=frame_index, head=None, torso=None)

        lm = result.pose_landmarks[0]   # 取第一個人

        head  = self._get_point(lm, _NOSE, w, h)
        torso = (
            self._midpoint_4(lm, _LEFT_SHOULDER, _RIGHT_SHOULDER,
                             _LEFT_HIP, _RIGHT_HIP, w, h)
            or self._midpoint_2(lm, _LEFT_SHOULDER, _RIGHT_SHOULDER, w, h)
        )

        return PoseFrame(frame_index=frame_index, head=head, torso=torso)

    def close(self) -> None:
        """釋放 MediaPipe 資源。"""
        self._detector.close()

    # ── 內部輔助方法 ──────────────────────────────────────────

    @staticmethod
    def _get_point(lm, idx, w, h) -> PosePoint | None:
        pt = lm[idx]
        if pt.visibility < MIN_VISIBILITY:
            return None
        return PosePoint(x=pt.x * w, y=pt.y * h)

    @staticmethod
    def _midpoint_2(lm, i1, i2, w, h) -> PosePoint | None:
        p1, p2 = lm[i1], lm[i2]
        if p1.visibility < MIN_VISIBILITY or p2.visibility < MIN_VISIBILITY:
            return None
        return PosePoint(x=(p1.x + p2.x) / 2 * w, y=(p1.y + p2.y) / 2 * h)

    @staticmethod
    def _midpoint_4(lm, i1, i2, i3, i4, w, h) -> PosePoint | None:
        pts = [lm[i] for i in (i1, i2, i3, i4)]
        if any(p.visibility < MIN_VISIBILITY for p in pts):
            return None
        return PosePoint(
            x=sum(p.x for p in pts) / 4 * w,
            y=sum(p.y for p in pts) / 4 * h,
        )
