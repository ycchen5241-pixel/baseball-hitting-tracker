# tools/claude_reviewer.py
# YOLO 誤偵測二次審查：Claude Vision API 自動標記 true/false positive
# 輸出 Roboflow 相容的 YOLO 訓練資料（images/ + labels/）

import base64
import json
import os
import re
import sys
from dataclasses import asdict, dataclass

import anthropic
import cv2
import numpy as np
from dotenv import load_dotenv

# pylint: disable=import-error
from detector.yolo_tracker import load_model, run_inference
# pylint: enable=import-error

# ── 路徑 ─────────────────────────────────────────────────────
MODEL_PATH = "Training/best_v7.pt"
OUTPUT_DIR = "review_output"

# ── 審查策略 ──────────────────────────────────────────────────
CONFIDENCE_AUTO_ACCEPT      = 0.70
CONFIDENCE_REVIEW_THRESHOLD = 0.65
FRAME_SAMPLE_INTERVAL       = 3
UNCERTAIN_THRESHOLD         = 0.75
REVIEW_CLASSES = frozenset({"Ball", "Bat", "Bat-top", "Bat-buttom"})
AUTO_ACCEPT_CLASSES         = frozenset({"Home-plate"})
_ALL_TARGET_CLASSES         = REVIEW_CLASSES | AUTO_ACCEPT_CLASSES

# ── 圖片 ─────────────────────────────────────────────────────
BBOX_PADDING_RATIO = 0.5
JPEG_QUALITY       = 90

# ── Claude API ───────────────────────────────────────────────
CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS   = 256

SYSTEM_PROMPT = """\
You are an expert sports video analysis assistant reviewing YOLO object
detection results for a baseball batting practice tracking system.

Scene context:
- Side-view or slightly overhead video of a batter hitting a baseball
  off a batting tee inside a batting cage
- Background typically contains nets, fences, walls, and cage poles

Target object definitions:
- Ball      : white baseball, round, 25-60 px diameter in frame,
              near the tee or in flight
- Bat       : baseball bat, dark elongated cylindrical object,
              held or swung by the batter
- Bat-top   : the barrel (thick) end of the bat
- Bat-buttom: the grip/handle (thin) end of the bat

Common false positives:
- Ball  -> round light fixtures, circular floor/wall markers,
           white blur spots (bokeh), round holes in netting
- Bat   -> vertical cage poles, fence rails, batter's forearm,
           dark shadows on floor or wall

Your task:
Determine whether the detection marked by the RED RECTANGLE in the image
is a genuine object or a false positive.
Respond ONLY with a single valid JSON object. No extra text.\
"""

USER_PROMPT_TEMPLATE = """\
Video : {video_name}
Frame : {frame_index:05d}
YOLO detected : {class_name}  (confidence {confidence:.2f})

The RED RECTANGLE marks the detected region in the image.
Is this detection correct?

Respond with ONLY this JSON (no markdown, no explanation outside JSON):
{{
  "is_correct"         : true | false,
  "confirmed_class"    : "Ball" | "Bat" | "Bat-top" | "Bat-buttom" | "none",
  "review_confidence"  : 0.0 to 1.0,
  "false_positive_type": "background_light" | "pole_or_fence" | "shadow" |
                         "body_part" | "other_round_object" | "other" | null,
  "reason"             : "max 20 words in English"
}}\
"""


@dataclass
class ReviewDecision:  # pylint: disable=too-many-instance-attributes
    """單一偵測的審查結果。"""
    frame_index:    int
    det_index:      int
    class_name:     str
    yolo_conf:      float
    bbox:           tuple[float, float, float, float]
    is_correct:     bool | None   # None = API 錯誤
    confirmed_class: str
    review_conf:    float
    fp_type:        str | None
    reason:         str
    source:         str           # "auto_accept" | "claude" | "error"


@dataclass
class _ReviewCtx:
    """每幀審查共用的上下文，避免大量參數傳遞。"""
    client:       anthropic.Anthropic
    video_name:   str
    frame_w:      int
    frame_h:      int
    class_id_map: dict[str, int]
    output_dir:   str


# ── 初始化 ────────────────────────────────────────────────────

def _setup_output_dirs(output_dir: str) -> None:
    """建立輸出目錄結構。"""
    for sub in (
        "images", "labels",
        os.path.join("crops", "certain"),
        os.path.join("crops", "uncertain"),
    ):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)


def _get_class_id_map(model) -> dict[str, int]:
    """從 model.names 建立 name→id 對應表（只含目標類別）。"""
    return {
        name: cid
        for cid, name in model.names.items()
        if name in _ALL_TARGET_CLASSES
    }


# ── 圖片處理 ──────────────────────────────────────────────────

def _prepare_crop(frame: np.ndarray, bbox: tuple) -> np.ndarray:
    """裁切 bbox + padding 區域並畫紅框標示原始 bbox。"""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in bbox)
    bw, bh = max(x2 - x1, 1), max(y2 - y1, 1)
    pad_x, pad_y = int(bw * BBOX_PADDING_RATIO), int(bh * BBOX_PADDING_RATIO)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    cv2.rectangle(
        crop, (x1 - cx1, y1 - cy1), (x2 - cx1, y2 - cy1), (0, 0, 255), 2
    )
    return crop


def _encode_image(crop: np.ndarray) -> str:
    """將 BGR 圖片編碼為 base64 JPEG 字串。"""
    _, buf = cv2.imencode(
        ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    return base64.standard_b64encode(buf).decode("utf-8")


# ── Claude API ───────────────────────────────────────────────

def _parse_response(raw: str) -> dict:
    """安全解析 Claude JSON 回應；格式錯誤時回傳錯誤結構。"""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    match = re.search(r'\{.*\}', raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {
        "is_correct": None,
        "confirmed_class": "none",
        "review_confidence": 0.0,
        "false_positive_type": None,
        "reason": "parse_error",
    }


def _error_result(reason: str) -> dict:
    """API 呼叫失敗時的預設回應結構。"""
    return {
        "is_correct": None,
        "confirmed_class": "none",
        "review_confidence": 0.0,
        "false_positive_type": None,
        "reason": f"api_error: {reason[:60]}",
    }


def _call_claude(
    client: anthropic.Anthropic,
    image_b64: str,
    video_name: str,
    frame_idx: int,
    class_name: str,
    conf: float,
) -> dict:
    """呼叫 Claude Vision API 審查單一偵測，回傳解析後的 dict。"""
    user_text = USER_PROMPT_TEMPLATE.format(
        video_name=video_name,
        frame_index=frame_idx,
        class_name=class_name,
        confidence=conf,
    )
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                }},
                {"type": "text", "text": user_text},
            ],
        }],
    )
    return _parse_response(resp.content[0].text)


# ── 決策建立 ──────────────────────────────────────────────────

def _make_auto_decision(det, frame_idx: int, det_idx: int) -> ReviewDecision:
    """建立自動接受的 ReviewDecision（不呼叫 Claude）。"""
    return ReviewDecision(
        frame_index=frame_idx, det_index=det_idx,
        class_name=det.class_name, yolo_conf=det.confidence,
        bbox=det.bbox, is_correct=True,
        confirmed_class=det.class_name, review_conf=1.0,
        fp_type=None, reason="auto_accepted", source="auto_accept",
    )


def _make_claude_decision(
    det, frame_idx: int, det_idx: int, result: dict,
) -> ReviewDecision:
    """從 Claude 回應 dict 建立 ReviewDecision。"""
    is_correct = result.get("is_correct")
    return ReviewDecision(
        frame_index=frame_idx, det_index=det_idx,
        class_name=det.class_name, yolo_conf=det.confidence,
        bbox=det.bbox, is_correct=is_correct,
        confirmed_class=result.get("confirmed_class", "none"),
        review_conf=float(result.get("review_confidence", 0.0)),
        fp_type=result.get("false_positive_type"),
        reason=result.get("reason", ""),
        source="error" if is_correct is None else "claude",
    )


# ── 審查流程 ──────────────────────────────────────────────────

def _save_crop(
    crop: np.ndarray,
    frame_idx: int,
    det_idx: int,
    class_name: str,
    decision: ReviewDecision,
    output_dir: str,
) -> None:
    """將 bbox 截圖存至 certain/ 或 uncertain/ 子目錄。"""
    if decision.source == "auto_accept":
        return
    uncertain = (
        decision.source == "error"
        or decision.review_conf < UNCERTAIN_THRESHOLD
    )
    subdir = "uncertain" if uncertain else "certain"
    if uncertain:
        label = "UNCERTAIN"
    elif decision.is_correct:
        label = "TP"
    else:
        label = "FP"
    fname = f"frame_{frame_idx:05d}_det{det_idx}_{class_name}_{label}.jpg"
    cv2.imwrite(
        os.path.join(output_dir, "crops", subdir, fname),
        crop,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )


def _review_single(
    det,
    det_idx: int,
    frame_idx: int,
    frame: np.ndarray,
    ctx: _ReviewCtx,
) -> ReviewDecision:
    """對單一偵測執行自動接受或 Claude 審查。"""
    if (det.class_name in AUTO_ACCEPT_CLASSES
            or det.confidence >= CONFIDENCE_AUTO_ACCEPT):
        return _make_auto_decision(det, frame_idx, det_idx)

    crop = _prepare_crop(frame, det.bbox)
    image_b64 = _encode_image(crop)
    try:
        result = _call_claude(
            ctx.client, image_b64, ctx.video_name,
            frame_idx, det.class_name, det.confidence,
        )
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  [ERR] f{frame_idx:05d} det{det_idx}: {exc}")
        result = _error_result(str(exc))

    decision = _make_claude_decision(det, frame_idx, det_idx, result)
    _save_crop(
        crop, frame_idx, det_idx, det.class_name, decision, ctx.output_dir
    )

    if decision.is_correct is True:
        tag = "TP"
    elif decision.is_correct is False:
        tag = "FP"
    else:
        tag = "ERR"
    print(
        f"  [{tag}] {det.class_name} yolo={det.confidence:.2f}"
        f" → {decision.confirmed_class}"
        f" rc={decision.review_conf:.2f} — {decision.reason}"
    )
    return decision


# ── 輸出儲存 ──────────────────────────────────────────────────

def _bbox_to_yolo(
    bbox: tuple, frame_w: int, frame_h: int,
) -> tuple[float, float, float, float]:
    """將像素 bbox 轉換為 YOLO 格式（normalized cx cy w h）。"""
    x1, y1, x2, y2 = bbox
    return (
        (x1 + x2) / 2.0 / frame_w,
        (y1 + y2) / 2.0 / frame_h,
        (x2 - x1) / frame_w,
        (y2 - y1) / frame_h,
    )


def _save_frame_outputs(
    frame: np.ndarray,
    frame_idx: int,
    decisions: list[ReviewDecision],
    ctx: _ReviewCtx,
) -> None:
    """儲存整幀影像與修正後的 YOLO label 檔案。"""
    cv2.imwrite(
        os.path.join(ctx.output_dir, "images", f"frame_{frame_idx:05d}.jpg"),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY],
    )
    yolo_lines = []
    for dec in decisions:
        if dec.is_correct is True and dec.confirmed_class in ctx.class_id_map:
            cid = ctx.class_id_map[dec.confirmed_class]
            cx, cy, bw, bh = _bbox_to_yolo(dec.bbox, ctx.frame_w, ctx.frame_h)
            yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(
        os.path.join(ctx.output_dir, "labels", f"frame_{frame_idx:05d}.txt"),
        "w", encoding="utf-8",
    ) as f:
        f.write("\n".join(yolo_lines))


def _process_frame(
    model,
    frame: np.ndarray,
    frame_idx: int,
    ctx: _ReviewCtx,
) -> list[ReviewDecision]:
    """對單一幀執行 YOLO 推論與 Claude 審查，儲存輸出。"""
    frame_dets = run_inference(model, frame, frame_idx, None)
    decisions = []
    for det_idx, det in enumerate(frame_dets.detections):
        if det.class_name not in _ALL_TARGET_CLASSES:
            continue
        decisions.append(_review_single(det, det_idx, frame_idx, frame, ctx))
    if decisions:
        _save_frame_outputs(frame, frame_idx, decisions, ctx)
    return decisions


# ── 報告 ──────────────────────────────────────────────────────

def _write_review_log(decisions: list[ReviewDecision], output_dir: str) -> None:
    """將所有審查結果寫入 review_log.json。"""
    path = os.path.join(output_dir, "review_log.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(d) for d in decisions], f, indent=2, ensure_ascii=False,
        )
    print(f"[reviewer] Log saved: {path}")


def _fp_breakdown(decisions: list[ReviewDecision]) -> dict[str, int]:
    """統計 false positive 的類型分布。"""
    breakdown: dict[str, int] = {}
    for dec in decisions:
        if dec.is_correct is False:
            key = dec.fp_type or "unknown"
            breakdown[key] = breakdown.get(key, 0) + 1
    return breakdown


def _write_summary(
    decisions: list[ReviewDecision],
    video_name: str,
    n_processed: int,
    output_dir: str,
) -> None:
    """產生統計報告並寫入 summary.txt。"""
    auto    = [d for d in decisions if d.source == "auto_accept"]
    claude  = [d for d in decisions if d.source == "claude"]
    errors  = [d for d in decisions if d.source == "error"]
    tp      = [d for d in claude if d.is_correct is True]
    fp      = [d for d in claude if d.is_correct is False]
    uncertain = errors + [
        d for d in claude if d.review_conf < UNCERTAIN_THRESHOLD
    ]
    lines = [
        "=== Claude Reviewer Summary ===",
        f"Video      : {video_name}",
        f"Processed  : {n_processed} frames",
        f"Detections : {len(decisions)} total",
        "",
        f"Auto-accepted (conf >= {CONFIDENCE_AUTO_ACCEPT:.2f}) : {len(auto)}",
        f"Claude reviewed : {len(claude) + len(errors)}",
        f"  True positive  : {len(tp)}",
        f"  False positive : {len(fp)}",
        f"  Uncertain/Error: {len(uncertain)}",
        "",
        "False positive breakdown:",
    ]
    for fp_type, count in sorted(
        _fp_breakdown(decisions).items(), key=lambda x: -x[1],
    ):
        lines.append(f"  {fp_type:<25}: {count}")
    report = "\n".join(lines)
    with open(
        os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8",
    ) as f:
        f.write(report + "\n")
    print(f"\n{report}")


# ── Entry Point ───────────────────────────────────────────────

def main(video_path: str) -> None:
    """處理影片，對 YOLO 偵測進行 Claude 審查並輸出訓練資料。"""
    load_dotenv()
    client       = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    model        = load_model(MODEL_PATH)
    class_id_map = _get_class_id_map(model)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_name = os.path.basename(video_path)

    _setup_output_dirs(OUTPUT_DIR)
    ctx = _ReviewCtx(
        client=client, video_name=video_name,
        frame_w=frame_w, frame_h=frame_h,
        class_id_map=class_id_map, output_dir=OUTPUT_DIR,
    )

    all_decisions: list[ReviewDecision] = []
    n_processed = 0
    frame_idx   = 0

    fps_str = f"{cap.get(cv2.CAP_PROP_FPS):.0f}fps"
    print(
        f"[reviewer] {video_name}  {frame_w}x{frame_h} {fps_str}"
        f"  {total} frames → ~{total // FRAME_SAMPLE_INTERVAL} sampled"
    )

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_SAMPLE_INTERVAL == 0:
            decisions = _process_frame(model, frame, frame_idx, ctx)
            all_decisions.extend(decisions)
            n_processed += 1
            if decisions:
                n_claude = sum(
                    1 for d in decisions if d.source != "auto_accept"
                )
                print(
                    f"[f{frame_idx:05d}/{total}]"
                    f" {len(decisions)} det(s), {n_claude} reviewed"
                )
        frame_idx += 1

    cap.release()
    _write_review_log(all_decisions, OUTPUT_DIR)
    _write_summary(all_decisions, video_name, n_processed, OUTPUT_DIR)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: py -3.12 tools/claude_reviewer.py <video_path>")
        print(
            "Example: py -3.12 tools/claude_reviewer.py"
            " Test_vod/IMG_2840.MOV"
        )
        sys.exit(1)
    main(sys.argv[1])
