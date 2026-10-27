"""
Extract frames from 6-7s of video (every 2 frames) and run YOLO detection.
Saves annotated images to test_frames_output/
"""
import cv2
import os
from pathlib import Path

VIDEO_PATH = "Test_vod/793471920.220478.mp4"
MODEL_PATH = "Training/best_v7.pt"
OUTPUT_DIR = "test_frames_output"
START_SEC = 6.0
END_SEC = 7.0
FRAME_STEP = 2      # every 2 frames
CONF_THRESHOLD = 0.1  # low threshold to catch weak detections

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load YOLO model ---
from ultralytics import YOLO
model = YOLO(MODEL_PATH)
print(f"Model loaded: {MODEL_PATH}")

# --- Open video ---
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"FPS: {fps}, Total frames: {total_frames}")

start_frame = int(START_SEC * fps)
end_frame = int(END_SEC * fps)
print(f"Extracting frames {start_frame} ~ {end_frame} (every {FRAME_STEP} frames)")

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

results_summary = []
frame_idx = start_frame

while frame_idx <= end_frame:
    ret, frame = cap.read()
    if not ret:
        break

    if (frame_idx - start_frame) % FRAME_STEP == 0:
        timestamp = frame_idx / fps

        # Run detection
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        result = results[0]

        detections = []
        annotated = frame.copy()

        for box in result.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{model.names[cls]} {conf:.2f}"
            detections.append({"class": model.names[cls], "conf": conf, "bbox": (x1, y1, x2, y2)})

            color = (0, 255, 0) if conf >= 0.5 else (0, 165, 255)  # green / orange
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Overlay frame info
        info = f"Frame {frame_idx} | t={timestamp:.3f}s | detections={len(detections)}"
        cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        out_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:05d}_t{timestamp:.3f}s.jpg")
        cv2.imwrite(out_path, annotated)

        results_summary.append({
            "frame": frame_idx,
            "time": timestamp,
            "detections": detections,
        })

        status = "DETECTED" if detections else "MISSED"
        det_str = ", ".join(f"{d['class']}@{d['conf']:.2f}" for d in detections) or "none"
        print(f"  [{status}] frame={frame_idx} t={timestamp:.3f}s -> {det_str}")

    frame_idx += 1

cap.release()

print(f"\nDone. {len(results_summary)} frames saved to '{OUTPUT_DIR}/'")
detected = sum(1 for r in results_summary if r["detections"])
print(f"Ball detected: {detected}/{len(results_summary)} frames")
