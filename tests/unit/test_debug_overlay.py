# tests/unit/test_debug_overlay.py
# main.py::_DebugOverlay 的輕量單元測試。
#
# 這是繪製/視覺化程式碼，沒有「算出來的數字對不對」可驗證，逐像素斷言
# 很脆弱、改一點繪製細節就會壞掉。這裡只驗證這個 seam 本身是通的：
# 不同資料組合下 render() 不會 crash，且回傳的 ndarray 形狀正確——不
# 驗證畫面實際長怎樣。

import numpy as np
import pytest

from detector.yolo_tracker import Detection
from main import _DebugOverlay, _PhysicsResult, _PipelineData
from physics.trajectory_validator import PreContactMode, TrajectoryValidation


def _blank_frame() -> np.ndarray:
    return np.zeros((100, 200, 3), dtype=np.uint8)


def _pipeline_data(ball_dets: dict | None = None) -> _PipelineData:
    return _PipelineData(
        all_ball_dets=[],
        ball_dets=ball_dets or {},
        bat_head_dets={},
        bat_body_dets={},
        hp_dets={},
        hp_locked_bbox=None,
        barrel_pos=[],
        frames=[],
    )


def _physics_result(
    contact=None, launch_angle=None, validation=None,
) -> _PhysicsResult:
    return _PhysicsResult(
        contact=contact,
        exit_velocity=None,
        launch_angle=launch_angle,
        bat_trajectory=[],
        validation=validation,
        ball_result=None,
    )


def _validation(suppress_trajectory=False, suppress_bat=False, spike_frames=None):
    return TrajectoryValidation(
        pre_mode=PreContactMode(mode="TEE", confidence=0.9, reason="static"),
        post_valid=True,
        post_reason="ok",
        bat_valid=True,
        bat_reason="ok",
        overall_valid=True,
        confidence=0.8,
        spike_frames=spike_frames or [],
        detection_coverage=1.0,
        suppress_trajectory=suppress_trajectory,
        suppress_bat=suppress_bat,
    )


@pytest.mark.parametrize(
    "ball_dets, contact, launch_angle, validation",
    [
        ({}, None, None, None),
        ({0: Detection("Ball", (10, 10, 20, 20), 0.9, 0)}, None, None, None),
        ({}, 0, 12.5, None),
        ({}, 0, None, None),
        ({}, None, None, _validation()),
        ({}, None, None, _validation(suppress_trajectory=True, spike_frames=[3, 4])),
    ],
)
def test_render_does_not_crash_and_preserves_shape(
    ball_dets, contact, launch_angle, validation,
):
    data = _pipeline_data(ball_dets)
    result = _physics_result(contact=contact, launch_angle=launch_angle, validation=validation)
    overlay = _DebugOverlay(data, result, ball_pos_map={})

    frame = _blank_frame()
    out = overlay.render(frame, frame_idx=0)

    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
