"""
Extract frames every N frames from a video segment for training data.
"""
import cv2
import os
import sys

VIDEO_PATH = sys.argv[1] if len(sys.argv) > 1 else ""
START_SEC  = float(sys.argv[2]) if len(sys.argv) > 2 else 206.0   # 3:26
END_SEC    = float(sys.argv[3]) if len(sys.argv) > 3 else 218.0   # 3:38
FRAME_STEP = int(sys.argv[4])   if len(sys.argv) > 4 else 5
OUTPUT_DIR = sys.argv[5]        if len(sys.argv) > 5 else "training_frames/jiangzhengho_3m26s"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: cannot open {VIDEO_PATH}")
    sys.exit(1)

fps = cap.get(cv2.CAP_PROP_FPS)
start_frame = int(START_SEC * fps)
end_frame   = int(END_SEC   * fps)
print(f"FPS={fps:.2f} | frames {start_frame}~{end_frame} | step={FRAME_STEP}")
print(f"Expected ~{(end_frame - start_frame) // FRAME_STEP} images -> {OUTPUT_DIR}/")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
saved = 0
frame_idx = start_frame

while frame_idx <= end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    if (frame_idx - start_frame) % FRAME_STEP == 0:
        t = frame_idx / fps
        fname = f"frame_{frame_idx:06d}_t{t:.3f}s.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), frame)
        saved += 1
        if saved % 20 == 0:
            print(f"  saved {saved} frames...")
    frame_idx += 1

cap.release()
print(f"Done. {saved} frames saved to '{OUTPUT_DIR}/'")
