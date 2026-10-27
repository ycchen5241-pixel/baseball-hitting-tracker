# tests/extract_frames.py
# 將影片指定時間區段按幀數切割為圖片
#
# 使用方式：
#   py -3.12 tests/extract_frames.py <video_path> <start_sec> <end_sec>
#                                    [--step 2] [--no-label]
#
# 輸出：
#   <video_stem>_frames_<start>s_<end>s/
#     frame_0119.png, frame_0121.png, ...（每 step 幀一張）
#
import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))  # pylint: disable=wrong-import-position

# 修改此變數以更換預設範例影片路徑
EXAMPLE_VIDEO = "Test_vod/793471920.220478.mp4"

FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_COLOR = (0, 255, 255)


def _stamp_frame(img, frame_index, time_sec):
    """在圖片左上角標記幀號與時間戳。"""
    text = f"Frame {frame_index:05d}  |  {time_sec:.3f}s"
    cv2.putText(img, text, (8, 28), FONT, 0.7, (0, 0, 0), 3)
    cv2.putText(img, text, (8, 28), FONT, 0.7, LABEL_COLOR, 1)
    return img


def extract_frames(
    video_path: str,
    start_sec: float,
    end_sec: float,
    step: int,
    no_label: bool,
) -> None:
    """切割影片指定區段並按幀數存成 PNG。

    Args:
        video_path: 輸入影片路徑。
        start_sec: 起始時間（秒）。
        end_sec: 結束時間（秒）。
        step: 每幾幀存一張（2 = 每 2 幀存一張）。
        no_label: 若 True，不在圖片上疊加幀號標籤。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_start = int(start_sec * fps)
    frame_end = min(int(end_sec * fps), total_frames - 1)

    if frame_start >= frame_end:
        print(f"[ERROR] start_sec={start_sec} >= end_sec={end_sec}")
        sys.exit(1)

    stem = Path(video_path).stem
    label = f"{start_sec:.0f}s_{end_sec:.0f}s"
    out_dir = Path(video_path).parent / f"{stem}_frames_{label}"
    out_dir.mkdir(exist_ok=True)

    target_indices = set(range(frame_start, frame_end + 1, step))
    saved = 0

    print(f"[extract] Video  : {video_path}")
    print(f"[extract] FPS    : {fps:.2f}")
    print(f"[extract] Range  : frame {frame_start} ~ {frame_end}"
          f"  ({end_sec - start_sec:.1f}s)")
    print(f"[extract] Step   : every {step} frames")
    print(f"[extract] Target : {len(target_indices)} images")
    print(f"[extract] Output : {out_dir}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)

    for fi in range(frame_start, frame_end + 1):
        ret, frame = cap.read()
        if not ret:
            break

        if fi not in target_indices:
            continue

        time_sec = fi / fps
        if not no_label:
            frame = _stamp_frame(frame, fi, time_sec)

        out_path = out_dir / f"frame_{fi:05d}.png"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    print(f"[extract] Done: {saved} images saved to {out_dir}")


def parse_args():
    """解析命令列參數。"""
    parser = argparse.ArgumentParser(
        description="Baseball Hitting Tracker — Frame Extractor",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "video_path",
        help=f"輸入影片路徑\n例：{EXAMPLE_VIDEO}",
    )
    parser.add_argument(
        "start_sec",
        type=float,
        help="起始時間（秒），例：4",
    )
    parser.add_argument(
        "end_sec",
        type=float,
        help="結束時間（秒），例：6",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=2,
        help="每幾幀存一張（預設：2）",
    )
    parser.add_argument(
        "--no-label",
        action="store_true",
        help="不在圖片上疊加幀號標籤",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract_frames(
        args.video_path,
        args.start_sec,
        args.end_sec,
        args.step,
        args.no_label,
    )
