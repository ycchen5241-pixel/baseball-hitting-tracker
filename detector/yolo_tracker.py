# detector/yolo_tracker.py
# YOLO 推論與 class-aware NMS 後處理

import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from config import constants


@dataclass
class Detection:
    """單一幀內一個偵測結果。"""
    class_name: str
    bbox: tuple[float, float, float, float]   # (x1, y1, x2, y2) pixels
    confidence: float
    frame_index: int
    keypoints: np.ndarray | None = None       # shape (N, 2) 若模型有關鍵點輸出


@dataclass
class FrameDetections:
    """單一幀的所有偵測結果集合。"""
    frame_index: int
    detections: list[Detection] = field(default_factory=list)

    def get_by_class(self, class_name: str) -> list[Detection]:
        """回傳指定類別的所有偵測。"""
        return [d for d in self.detections if d.class_name == class_name]

    def best_by_class(self, class_name: str) -> 'Detection | None':
        """回傳指定類別信心度最高的偵測；無偵測時回傳 None。"""
        dets = self.get_by_class(class_name)
        return max(dets, key=lambda d: d.confidence) if dets else None

    def best_moving_by_class(
        self,
        class_name: str,
        frame_curr: np.ndarray,
        frame_prev: np.ndarray,
        min_motion: float,
    ) -> 'Detection | None':
        """回傳指定類別中運動分數最高且超過閾值的偵測。

        若所有候選均低於 min_motion（均靜止，如柱子），退回信心度最高者。
        這樣在物件靜止的幀（打者持棒不動）仍能正常追蹤，
        但當場景中同時有靜止背景物與運動物件時，優先選擇後者。

        Args:
            class_name: 目標類別名稱。
            frame_curr: 當前幀（BGR）。
            frame_prev: 前一幀（BGR）；為 None 時等同 best_by_class。
            min_motion: 視為「有在運動」的最低幀差分分數。

        Returns:
            最佳偵測；無偵測時回傳 None。
        """
        dets = self.get_by_class(class_name)
        if not dets:
            return None
        if frame_prev is None or frame_prev.size == 0:
            return max(dets, key=lambda d: d.confidence)
        scored = [
            (detection_motion_score(d, frame_curr, frame_prev), d)
            for d in dets
        ]
        moving = [(s, d) for s, d in scored if s >= min_motion]
        if moving:
            return max(moving, key=lambda sd: sd[0])[1]
        return max(dets, key=lambda d: d.confidence)

    def nearest_by_class(
        self,
        class_name: str,
        ref_x: float,
        ref_y: float,
    ) -> 'Detection | None':
        """回傳距參考位置最近的同類偵測；無偵測時回傳 None。

        用於跨幀位置連貫性追蹤：多個候選物件時優先選距上一幀
        最近者，避免信心度較高的靜止背景物件（如柱子）搶占追蹤。
        """
        dets = self.get_by_class(class_name)
        if not dets:
            return None
        def _cx(d: Detection) -> float:
            return (d.bbox[0] + d.bbox[2]) / 2.0
        def _cy(d: Detection) -> float:
            return (d.bbox[1] + d.bbox[3]) / 2.0
        return min(
            dets,
            key=lambda d: math.sqrt(
                (_cx(d) - ref_x) ** 2 + (_cy(d) - ref_y) ** 2
            ),
        )


def detection_motion_score(
    det: 'Detection',
    frame_curr: np.ndarray,
    frame_prev: np.ndarray,
) -> float:
    """計算偵測框區域的幀差分運動分數（平均像素變化量）。

    靜止物件（背景牆、柱子）：像素幾乎不變 → 分數趨近 0。
    運動中的物件（揮棒中的球棒、飛行中的球）：像素快速改變 → 分數較高。

    分數單位為灰階像素差均值（0–255），不受物件大小影響。

    Args:
        det: 要評分的偵測結果（bbox 提供感興趣區域）。
        frame_curr: 當前幀（BGR）。
        frame_prev: 前一幀（BGR）。若為 None 或空陣列則回傳 0。

    Returns:
        float：運動分數，越高表示該區域越活躍。
    """
    if frame_prev is None or frame_prev.size == 0:
        return 0.0
    h, w = frame_curr.shape[:2]
    x1 = max(0, int(det.bbox[0]))
    y1 = max(0, int(det.bbox[1]))
    x2 = min(w, int(det.bbox[2]))
    y2 = min(h, int(det.bbox[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    roi_curr = frame_curr[y1:y2, x1:x2].astype(np.float32)
    roi_prev = frame_prev[y1:y2, x1:x2].astype(np.float32)
    return float(np.mean(np.abs(roi_curr - roi_prev)))


def load_model(model_path: str) -> YOLO:
    """載入 YOLO 模型。

    Args:
        model_path: .pt 模型檔路徑。

    Returns:
        已載入的 YOLO 模型。
    """
    return YOLO(model_path)


def detect_content_roi(
    frames: list[np.ndarray],
    brightness_threshold: float = 30.0,
    min_height: int = 100,
) -> tuple[int, int] | None:
    """從前幾幀的亮度分布自動偵測有效內容的垂直範圍（y_start, y_end）。

    用於螢幕錄影類影片：影片中包含 UI 黑色邊框（狀態欄、導航欄等），
    需自動裁切至實際影像內容，避免 YOLO 把 UI 元素誤判為目標物件。

    Args:
        frames: 用於分析的幀列表（通常取前 10 幀）。
        brightness_threshold: 列平均亮度的最低門檻，低於此值視為 UI 黑色區域。
        min_height: 有效內容最小高度（px），過小的亮帶視為 UI 圖示，忽略。

    Returns:
        (y_start, y_end) 內容行範圍；找不到時回傳 None（使用完整幀）。
    """
    avg = np.stack(frames).mean(axis=0).astype(np.uint8)
    row_bright = cv2.cvtColor(avg, cv2.COLOR_BGR2GRAY).mean(axis=1)
    mask = row_bright > brightness_threshold

    best: tuple[int, int] | None = None
    best_len = 0
    start: int | None = None

    for i, bright in enumerate(mask):
        if bright and start is None:
            start = i
        elif not bright and start is not None:
            seg_len = i - start
            if seg_len >= min_height and seg_len > best_len:
                best_len = seg_len
                best = (start, i - 1)
            start = None

    if start is not None:
        seg_len = len(mask) - start
        if seg_len >= min_height and seg_len > best_len:
            best = (start, len(mask) - 1)

    return best


def run_inference(
    model: YOLO,
    frame: np.ndarray,
    frame_index: int,
    crop_roi: tuple[int, int] | None = None,
) -> FrameDetections:
    """對單幀執行 YOLO 推論，回傳結構化偵測結果。

    Args:
        model: 已載入的 YOLO 模型。
        frame: BGR 影像陣列（OpenCV 格式）。
        frame_index: 當前幀索引（從 0 開始）。
        crop_roi: (y_start, y_end) 垂直裁切範圍（含端點）。
                  指定時只對 ROI 子圖做推論，偵測框坐標自動映射回原圖。
                  用於過濾螢幕錄影的 UI 黑邊，避免 UI 元素誤判。

    Returns:
        該幀的所有偵測結果（坐標均為原圖座標系）。
    """
    y_offset = 0
    infer_frame = frame
    if crop_roi is not None:
        y0, y1 = crop_roi
        infer_frame = frame[y0: y1 + 1]
        y_offset = y0

    results = model(
        infer_frame,
        conf=constants.YOLO_CONFIDENCE_THRESHOLD,
        iou=constants.YOLO_IOU_THRESHOLD,
        agnostic_nms=False,
        verbose=False,
    )

    frame_dets = FrameDetections(frame_index=frame_index)

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for i, box in enumerate(boxes):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name not in constants.VALID_CLASSES:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # 將 y 坐標映射回原圖
            y1 += y_offset
            y2 += y_offset
            confidence = float(box.conf[0])

            kpts = None
            if result.keypoints is not None:
                kp_xy = result.keypoints.xy
                if i < len(kp_xy):
                    kpts = kp_xy[i].cpu().numpy().copy()
                    if kpts is not None and len(kpts):
                        kpts[:, 1] += y_offset  # keypoint y 同樣偏移

            frame_dets.detections.append(Detection(
                class_name=class_name,
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                frame_index=frame_index,
                keypoints=kpts,
            ))

    return frame_dets
