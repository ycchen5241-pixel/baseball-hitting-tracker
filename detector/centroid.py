# detector/centroid.py
# 球位置資料結構與 bbox 座標轉換工具

from dataclasses import dataclass


@dataclass
class BallPosition:
    """球在某幀的位置資訊。"""
    x: float          # 絕對像素 x（質心）
    y: float          # 絕對像素 y（質心）
    frame_index: int
    confidence: float


def ball_position_from_bbox(
    bbox: tuple[float, float, float, float],
    frame_index: int,
    confidence: float,
) -> BallPosition:
    """從 bounding box 中心建立 BallPosition。

    Args:
        bbox: (x1, y1, x2, y2) bounding box 座標。
        frame_index: 當前幀索引。
        confidence: YOLO 偵測信心度。

    Returns:
        以 bbox 中心為座標的 BallPosition。
    """
    return BallPosition(
        x=(bbox[0] + bbox[2]) / 2.0,
        y=(bbox[1] + bbox[3]) / 2.0,
        frame_index=frame_index,
        confidence=confidence,
    )


def make_raw_positions(ball_dets: dict) -> list[BallPosition]:
    """從 YOLO 原始偵測建立按幀排序的位置序列。

    用於接觸幀偵測與仰角計算，繞過 RANSAC 對方向改變幀的拒絕。

    Args:
        ball_dets: {frame_index: Detection} 的球偵測字典。

    Returns:
        按 frame_index 升序排列的 BallPosition 序列。
    """
    return sorted(
        [
            ball_position_from_bbox(d.bbox, fi, d.confidence)
            for fi, d in ball_dets.items()
        ],
        key=lambda p: p.frame_index,
    )


@dataclass
class BatPositionsInput:
    """CloudBallTracker.process() 的 bat_positions 契約（2026-07-27，架構審查
    Candidate E）。

    CloudBallTracker.process() 建議吃 BatCloudTracker 清洗過的乾淨軌跡（T 座
    辨識的鄰近度備援訊號才穩定），但這個建議原本只寫在參數註解裡，兩種
    來源回傳的型別完全一樣（都是 list[BallPosition]），呼叫端可以在不知情
    的狀況下傳錯而不會有任何警訊——main.py/test_side_view_ev.py 傳原始
    YOLO 偵測，test_cloud_tracker.py 傳清洗過的版本，三邊已經不一致而
    沒有任何機制發現。

    這個型別不禁止傳原始資料（兩者現在的行為都刻意保留，見待決問題.md），
    只強迫呼叫端用 .raw()／.cleaned() 明確標記來源，讓這個選擇在呼叫端
    可見、可搜尋，不再靠巧合維持一致。
    """
    positions: list[BallPosition]
    source: str  # "cleaned" | "raw"

    @classmethod
    def cleaned(cls, positions: list[BallPosition]) -> "BatPositionsInput":
        return cls(positions, source="cleaned")

    @classmethod
    def raw(cls, positions: list[BallPosition]) -> "BatPositionsInput":
        return cls(positions, source="raw")
