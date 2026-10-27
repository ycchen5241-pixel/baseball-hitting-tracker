"""
Raw YOLO detection test — renders every detected bbox with class/conf label.
Usage: python tools/raw_yolo_test.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

VIDEO_PATH  = "Test_vod/0703 Indi/3304/IMG_3304.MOV"
MODEL_PATH  = "Training/best_v7.pt"
OUT_PATH    = "Test_vod/0703 Indi/3304/IMG_3304_raw_yolo.mp4"
LOG_PATH    = "Test_vod/0703 Indi/3304/IMG_3304_raw_yolo.txt"
START_SEC   = 0.0
END_SEC     = 4.3
CONF_THRESH = 0.15
OUT_FPS     = 60.0
FRAME_STEP  = 1               # 片段模式：每幀都寫入影片

CLASS_COLORS = {
    "Ball":       (0, 255, 255),    # 黃色
    "Bat":        (255, 200, 0),    # 天藍
    "Bat-top":    (0, 165, 255),    # 橘
    "Bat-buttom": (0, 80, 255),     # 紅橘
    "Home-plate": (0, 255, 0),      # 綠
}
DEFAULT_COLOR = (200, 200, 200)

from ultralytics import YOLO
model = YOLO(MODEL_PATH)
print(f"Model: {MODEL_PATH}")

cap = cv2.VideoCapture(VIDEO_PATH)
fps    = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {width}x{height} @ {fps:.2f}fps, total {total} frames")

start_f = int(START_SEC * fps)
end_f   = int(END_SEC   * fps)
print(f"Processing frames {start_f}–{end_f} ({START_SEC}s–{END_SEC}s)")

fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
out      = cv2.VideoWriter(OUT_PATH, fourcc, OUT_FPS, (width, height))
if not out.isOpened():
    raise RuntimeError(f"VideoWriter failed to open: {OUT_PATH}")
log_file = open(LOG_PATH, "w", encoding="utf-8")
log_file.write(f"Video: {VIDEO_PATH}\n")
log_file.write(f"FPS: {fps:.4f}  Size: {width}x{height}  Total: {total}\n")
log_file.write(f"Model: {MODEL_PATH}  Conf: {CONF_THRESH}\n\n")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
fi = start_f
frame_count   = 0
det_frames    = 0
written_count = 0

while fi < end_f:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=CONF_THRESH, verbose=False)
    vis     = frame.copy()
    dets_this_frame = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id  = int(box.cls[0])
            cls_name = model.names[cls_id]
            conf    = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color   = CLASS_COLORS.get(cls_name, DEFAULT_COLOR)
            label   = f"{cls_name} {conf:.2f}"

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis, label, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            dets_this_frame.append(f"{cls_name}@{conf:.2f}")

    t_rel  = fi / fps - START_SEC
    t_abs  = fi / fps
    status = ", ".join(dets_this_frame) if dets_this_frame else "—"
    line   = f"  f{fi:05d} +{t_rel:6.2f}s [{t_abs:.2f}s] {status}"
    print(line)
    log_file.write(line + "\n")

    info = (f"f{fi}  {t_abs:.2f}s  dets={len(dets_this_frame)}")
    cv2.putText(vis, info, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)

    # 每 FRAME_STEP 幀寫一幀進影片
    if (fi - start_f) % FRAME_STEP == 0:
        out.write(vis)
        written_count += 1

    frame_count += 1
    if dets_this_frame:
        det_frames += 1
    fi += 1

cap.release()
out.release()
log_file.write(f"\nDone. {frame_count} frames processed, "
               f"{written_count} written to video.\n")
log_file.write(f"Frames with any detection: {det_frames}/{frame_count} "
               f"({100*det_frames/max(frame_count,1):.1f}%)\n")
log_file.close()

print(f"\nDone. {frame_count} frames processed → log: {LOG_PATH}")
print(f"Video frames written: {written_count} → {OUT_PATH}")
print(f"Frames with any detection: {det_frames}/{frame_count} "
      f"({100*det_frames/max(frame_count,1):.1f}%)")
