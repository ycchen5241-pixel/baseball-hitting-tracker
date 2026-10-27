# extract_frames.py
# 自動偵測擊球點並擷取訓練用圖片（每隔 N 幀取一張）
# 用法：py -3.12 extract_frames.py <影片路徑>

import os
import sys
import subprocess

from audio_detect import detect_hits

POST_SEC = 2.0       # 擊球後擷取秒數
SAMPLE_EVERY = 3     # 每隔幾幀取一張（240fps 下每 3 幀 = 80fps，2 秒約 160 張）
OUTPUT_DIR = "training_frames"


def extract(video_path: str, hit_times: list[float]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    total = 0

    for i, t in enumerate(hit_times):
        vf = f"fps=240,select='not(mod(n\\,{SAMPLE_EVERY}))'"
        pattern = os.path.join(OUTPUT_DIR, f"hit{i+1:02d}_f%05d.jpg")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{t:.3f}",
                "-i", video_path,
                "-t", f"{POST_SEC:.1f}",
                "-vf", vf,
                "-vsync", "vfr",
                "-q:v", "2",
                pattern,
            ],
            check=True,
            capture_output=True,
        )
        n = len([f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"hit{i+1:02d}_")])
        total += n
        print(f"  hit{i+1:02d} @ {t:.2f}s → {n} 張")

    print(f"\n完成：共 {total} 張圖片 → {OUTPUT_DIR}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：py -3.12 extract_frames.py <影片>")
        sys.exit(1)

    video = sys.argv[1]
    hit_times = detect_hits(video)

    if not hit_times:
        print("未偵測到擊球點，結束")
        sys.exit(0)

    print(f"\n開始擷取 {len(hit_times)} 個擊球點的訓練幀...\n")
    extract(video, hit_times)
