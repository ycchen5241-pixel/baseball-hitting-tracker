# detector/barrel_tracker.py
# 棒頭（barrel）位置追蹤與揮棒視窗偵測

import math
from dataclasses import dataclass

from config import constants
from detector.yolo_tracker import FrameDetections


@dataclass
class BarrelPosition:
    """棒頭（barrel）位置，由多種來源合成。"""
    x: float
    y: float
    frame_index: int
    source: str        # "BAT_TOP" | "TRIANGULATE" | "BAT_BODY"
    confidence: float


class BarrelTracker:
    """棒頭位置追蹤器。

    優先序：
      1. Bat-top 直接偵測（最可靠，直接給出棒頭位置）
      2. Bat + Bat-buttom 三角定位（barrel ≈ 2×bat_center − buttom_center）
      3. Bat body 中心（方向不確定，僅在前兩者均失敗時使用）

    三角定位原理：以 Bat-buttom 為槓桿支點，Bat 整體中心為力點，
    barrel 為施力端，三者共線且 bat_center = (barrel + buttom) / 2。
    """

    def update(
        self,
        frame_dets: FrameDetections,
        frame_index: int,
    ) -> BarrelPosition | None:
        """更新棒頭位置並回傳最佳估算。

        Args:
            frame_dets: 當前幀 YOLO 偵測結果。
            frame_index: 當前幀索引。

        Returns:
            棒頭位置；本幀無任何 bat 偵測時回傳 None。
        """
        top = frame_dets.best_by_class(constants.CLASS_BAT_HEAD)
        body = frame_dets.best_by_class(constants.CLASS_BAT)
        buttom = frame_dets.best_by_class(constants.CLASS_BAT_TAIL)

        return (
            self._from_bat_top(top, frame_index)
            or self._from_triangulate(body, buttom, frame_index)
            or self._from_body(body, frame_index)
        )

    @staticmethod
    def _center(det) -> tuple[float, float]:
        return (
            (det.bbox[0] + det.bbox[2]) / 2.0,
            (det.bbox[1] + det.bbox[3]) / 2.0,
        )

    def _from_bat_top(self, top, frame_index: int) -> BarrelPosition | None:
        if top is None:
            return None
        cx, cy = self._center(top)
        return BarrelPosition(cx, cy, frame_index, "BAT_TOP", top.confidence)

    def _from_triangulate(
        self, body, buttom, frame_index: int,
    ) -> BarrelPosition | None:
        if body is None or buttom is None:
            return None
        bx, by = self._center(body)
        tx, ty = self._center(buttom)
        cx = 2.0 * bx - tx
        cy = 2.0 * by - ty
        conf = min(body.confidence, buttom.confidence)
        return BarrelPosition(cx, cy, frame_index, "TRIANGULATE", conf)

    def _from_body(self, body, frame_index: int) -> BarrelPosition | None:
        if body is None:
            return None
        cx, cy = self._center(body)
        return BarrelPosition(
            cx, cy, frame_index, "BAT_BODY", body.confidence * 0.5,
        )
