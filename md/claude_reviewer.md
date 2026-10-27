# Claude Reviewer

## 設定

`.env` → `ANTHROPIC_API_KEY=sk-ant-...`
依賴：`anthropic`, `python-dotenv`, `ultralytics`, `opencv-python`

## API 呼叫

```python
load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

_, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
image_b64 = base64.standard_b64encode(buf).decode("utf-8")

response = client.messages.create(
    model="claude-sonnet-4-6", max_tokens=256,
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64",
         "media_type": "image/jpeg", "data": image_b64}},
        {"type": "text", "text": user_prompt},
    ]}],
)
raw = response.content[0].text  # JSON string
```

---

## System Prompt

```
You are an expert sports video analysis assistant reviewing YOLO object
detection results for a baseball batting practice tracking system.

Scene context:
- Side-view or slightly overhead video of a batter hitting a baseball
  off a batting tee inside a batting cage
- Background typically contains nets, fences, walls, and cage poles

Target object definitions:
- Ball      : white baseball, round, 25–60 px diameter in frame,
              near the tee or in flight
- Bat       : baseball bat, dark elongated cylindrical object,
              held or swung by the batter
- Bat-top   : the barrel (thick) end of the bat
- Bat-buttom: the grip/handle (thin) end of the bat

Common false positives:
- Ball  → round light fixtures, circular floor/wall markers,
          white blur spots (bokeh), round holes in netting
- Bat   → vertical cage poles, fence rails, batter's forearm,
          dark shadows on floor or wall

Your task:
Determine whether the detection marked by the RED RECTANGLE in the image
is a genuine object or a false positive.
Respond ONLY with a single valid JSON object. No extra text.
```

---

## User Prompt Template

```python
USER_PROMPT_TEMPLATE = """Video : {video_name}
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
}}"""
```

---

## Response Schema

```json
{
  "is_correct": false,
  "confirmed_class": "none",
  "review_confidence": 0.91,
  "false_positive_type": "background_light",
  "reason": "Round light fixture on cage wall, not a baseball"
}
```

`review_confidence < 0.75` → 存入 `crops/uncertain/` 供人工複查

---

## 輸出結構

```
review_output/
  images/          ← 整幀 JPEG（Roboflow 上傳用）
  labels/          ← 修正後 YOLO .txt；空檔 = 背景負樣本
  crops/certain/   ← 確認判斷的 bbox 截圖
  crops/uncertain/ ← 低信心判斷，需人工確認
  review_log.json
  summary.txt
```

YOLO `.txt`：`class_id cx cy w h`（normalized，`class_id` 從 `model.names` 動態取得）

---

## 參數

```python
CONFIDENCE_REVIEW_THRESHOLD = 0.65   # 低於此值送 Claude
CONFIDENCE_AUTO_ACCEPT      = 0.70   # 高於此值直接接受
FRAME_SAMPLE_INTERVAL       = 3      # 每 N 幀取一幀
BBOX_PADDING_RATIO          = 0.5    # bbox 外擴 50%
UNCERTAIN_THRESHOLD         = 0.75   # 低於此值人工複查
REVIEW_CLASSES    = ["Ball", "Bat", "Bat-top", "Bat-buttom"]
AUTO_ACCEPT_CLASSES = ["Home-plate"]
CLAUDE_MODEL      = "claude-sonnet-4-6"
MAX_TOKENS        = 256
```
