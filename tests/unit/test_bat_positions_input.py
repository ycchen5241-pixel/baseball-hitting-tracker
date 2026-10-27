# tests/unit/test_bat_positions_input.py
# detector/centroid.py::BatPositionsInput 的自動化單元測試（架構審查
# Candidate E）：驗證 .cleaned()/.raw() 正確標記來源，且
# CloudBallTracker.process() 真的有解包 .positions 傳給內部邏輯，
# 不是型別對了但內容被忽略。

from unittest.mock import patch

from detector.ball_tracker_cloud import CloudBallTracker
from detector.centroid import BallPosition, BatPositionsInput
from detector.yolo_tracker import Detection


def _det(frame_index: int, x: float, y: float) -> tuple:
    return (frame_index, Detection("Ball", (x - 5, y - 5, x + 5, y + 5), 0.9, frame_index))


def test_raw_sets_source():
    wrapped = BatPositionsInput.raw([])
    assert wrapped.source == "raw"


def test_cleaned_sets_source():
    wrapped = BatPositionsInput.cleaned([])
    assert wrapped.source == "cleaned"


def test_positions_preserved_unchanged():
    positions = [BallPosition(1.0, 2.0, 3, 0.9)]
    assert BatPositionsInput.raw(positions).positions is positions
    assert BatPositionsInput.cleaned(positions).positions is positions


def test_process_unwraps_positions_for_internal_use():
    """迴歸測試：CloudBallTracker.process() 拿到 BatPositionsInput 後，
    傳給內部 _identify_tstand() 的必須是解包後的原始 list，不是包裝物件
    本身——否則型別安全等於白做，接口看起來對了但資料根本沒被使用。
    """
    # 一段靜止叢集（模擬球靜止在原地）+ 一段線性移動（模擬飛行段），
    # 足以讓 process() 走完前面幾個 Stage、實際呼叫到 _identify_tstand()。
    all_dets = [_det(i, 500, 500) for i in range(100)]
    all_dets += [_det(100 + i, 500 + i * 20, 500 - i * 2) for i in range(10)]

    sentinel_positions = [BallPosition(1.0, 1.0, 1, 1.0)]
    captured = {}

    def fake_identify_tstand(static_clusters, bat_positions, flight_seg, total_frames):
        captured["bat_positions"] = bat_positions
        return None

    with patch(
        "detector.ball_tracker_cloud._identify_tstand",
        side_effect=fake_identify_tstand,
    ) as mocked:
        CloudBallTracker().process(
            all_dets, BatPositionsInput.raw(sentinel_positions), total_frames=110,
        )

    assert mocked.called
    assert captured["bat_positions"] is sentinel_positions
