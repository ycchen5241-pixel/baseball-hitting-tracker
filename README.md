# Baseball Hitting Tracker

側面攝影機慢動作影片 → YOLO 偵測球／球棒／本壘板 → 點雲式軌跡追蹤 → 計算擊球初速
（Exit Velocity）與仰角（Launch Angle）→ 疊圖輸出標注影片。目前是 Python 原型階段，
用來驗證核心 CV／物理運算邏輯是否可行，之後會轉 ONNX → TFLite / CoreML 搬進手機
App（詳見 `md/PROJECT_SPEC.md`）。

## 這個專案在做什麼

- **輸入**：側面拍攝、畫面含本壘板、建議 240fps 慢動作影片。
- **輸出**：標注後的影片（偵測框、追蹤軌跡、接觸幀標記、EV/LA 數據面板）+ console
  印出的逐步分析過程。
- **定位**：手機端棒球擊球分析 App「Flow」的核心運算邏輯原型（iOS + Android /
  React Native），不是最終產品本身。

## 核心概念（給第一次看這個專案的人）

- **YOLOv8s 五類偵測**：`Ball` / `Bat` / `Bat-top`（棒頭）/ `Bat-buttom`（棒尾，
  訓練標註本身的 typo，刻意不修正以保持對應）/ `Home-plate`。
- **點雲式追蹤（CloudBallTracker / BatCloudTracker）**：不是逐幀線上追蹤，而是
  離線批次分析——先收集全片所有偵測點，用 XY 密度聚類找出「靜止背景球」「持棒
  待擊」等靜止群聚並移除，再對剩餘稀疏點做 Sequential RANSAC 穩健擬合，分辨出
  來球段／飛行段／反彈段。
- **Contact Frame（擊球幀）**：球棒擊中球的那一個 frame_index。`CloudBallTracker`
  內建三訊號判定，失敗時退回距離最小值／方向改變的備援法。
- **Velocity Window（速度視窗）**：Contact Frame 之後的乾淨球位置子集，是 Launch
  Angle 與 Exit Velocity 共用的輸入資料。
- **Exit Velocity**：側拍單軸法——用本壘板 bbox 的已知實際寬度定標，把球的
  image-x 像素速度換算成真實速度，再除以 cos(仰角) 還原成總速度。
- **Launch Angle**：像素 Y/X 位移比例（割線法）+ 攝影機仰角造成的視差補正。
- 更完整、更精確的術語定義見 `md/CONTEXT.md`。

## 快速開始

### 安裝

需要 Python 3.12（指令以 `py -3.12` 呼叫）。

```bash
pip install opencv-python numpy ultralytics
# 以下為選用，只有特定輔助腳本會用到：
pip install mediapipe          # detector/pose_tracker.py、diagnostics/pose_check.py（實驗性姿勢追蹤，未接入 main.py）
pip install anthropic python-dotenv  # tools/claude_reviewer.py（YOLO 誤偵測二次審查工具）
pip install pytest             # 跑 tests/unit/ 下的自動化測試
```

### 執行主流程

```bash
py -3.12 main.py <影片路徑>
# 範例：
py -3.12 main.py "Test_vod/0703 Indi/3291/IMG_3291.MOV"
```

輸出：同目錄下 `<影片檔名>_tracked.mp4`，以及 console 印出的偵測數量／Contact
Frame／Launch Angle／Exit Velocity 等逐步分析結果。

### 跑自動化測試

```bash
pytest
```

`pytest.ini` 已設定只收集 `tests/unit/` 下的真測試，不會誤觸 `diagnostics/`
裡需要真實影片＋YOLO 模型的手動視覺化診斷腳本。

## 專案架構

```
baseball-hitting-tracker/
├── main.py                    # 整合入口：讀影片 → YOLO推論 → 球/棒追蹤 → 物理計算 → 疊圖輸出
├── config/
│   └── constants.py            # 全部數值常數集中管理（嚴禁魔術數字散落各模組）
├── detector/                   # 偵測與追蹤
│   ├── yolo_tracker.py         # YOLO 推論 + class-aware NMS 封裝；Detection/FrameDetections 資料結構
│   ├── ball_tracker_cloud.py   # CloudBallTracker：球的點雲式追蹤主邏輯（主線）
│   ├── bat_tracker_cloud.py    # BatCloudTracker：球棒的點雲式追蹤（揮棒軌跡/視窗判定）
│   ├── trajectory_fitter.py    # Sequential RANSAC 軌跡擬合（取代舊版 Stage5/6）
│   ├── centroid.py             # BallPosition 資料結構、bbox→座標轉換
│   ├── clustering.py           # XY 密度聚類共用工具
│   ├── barrel_tracker.py       # 棒頭（barrel）位置追蹤，Contact Frame 備援判定用
│   └── pose_tracker.py         # MediaPipe 姿勢估計封裝（實驗性，未接入 main.py）
├── physics/                     # 純物理/數值計算，唯一真實來源
│   ├── pipeline.py              # compute_ball_physics()：Contact Frame→Velocity Window→LA→EV 整合入口
│   ├── contact.py                # 擊球幀備援判定（距離最小值/T座/方向改變三路徑）
│   ├── velocity.py               # Exit Velocity 計算（側拍單軸法 + Theil-Sen 斜率）
│   ├── angle.py                   # Launch Angle 計算 + 攝影機仰角視差補正
│   ├── trajectory.py              # 球棒軌跡用輔助函式（P25過濾/補洞/RANSAC圓擬合等）
│   └── trajectory_validator.py    # 軌跡合理性事後驗證（維護/除錯用，不影響使用者可見數據）
├── visualizer/
│   └── overlay.py                # 疊圖繪製：軌跡線漸層、數據面板
├── tools/                        # 獨立輔助工具（不是 main.py pipeline 的一部分）
│   ├── homography.py              # 本壘板角點提取 + Homography 計算函式庫
│   ├── check_homography.py        # 全片掃描找最佳信心幀，檢驗 homography 重投影誤差
│   ├── homography_diag.py         # 單一硬編碼影片/幀的 homography 診斷
│   └── claude_reviewer.py         # 用 Claude Vision API 二次審查 YOLO 誤偵測，產出 Roboflow 訓練格式
├── diagnostics/                  # 人工視覺化診斷腳本（argparse CLI，非 pytest 測試）
│   ├── bat_trail_cloud.py         # 球棒追蹤可視化 + 點雲 PNG 輸出
│   ├── cloud_tracker.py           # CloudBallTracker 可視化
│   ├── side_view_ev.py            # 側拍 EV/LA 計算診斷（main.py 物理邏輯的獨立驗證基準）
│   ├── homography_check.py        # 本壘板角點提取的另一套獨立演算法，未整合進 main.py
│   └── pose_check.py              # pose_tracker.py 可視化
├── tests/unit/                   # 真正的 pytest 自動化測試（pytest.ini 只收集這裡）
├── md/                            # 專案文件（規格/進度/物理驗證原則/待決問題等，見下方索引）
├── Training/                      # YOLO 模型權重（best_v1~v7.pt）+ MediaPipe pose 模型
├── training_frames/、training_photo/、test_frames_output/   # 訓練/除錯用影格圖片輸出
├── extract_training_frames.py    # 根目錄現役工具：手動指定時間區段從影片擷取訓練幀
└── test_detection_frames.py      # 根目錄孤兒腳本：YOLO 偵測+標註視覺化測試用（硬編碼路徑，保留未分類）
```

## 處理流程（main.py 主線）

1. **YOLO 推論**：`detector/yolo_tracker.py` 逐幀偵測 Ball / Bat / Bat-top /
   Bat-buttom / Home-plate。
2. **資料收集**：`main.py::_collect_data` 彙整全片偵測，分流成
   `all_ball_dets`（全部球偵測）/ `bat_head_dets` / `bat_body_dets` / `hp_dets`。
3. **球棒軌跡**：`detector/bat_tracker_cloud.py::BatCloudTracker` 點雲式過濾 →
   揮棒視窗判定。
4. **球體追蹤**：`detector/ball_tracker_cloud.py::CloudBallTracker` 點雲式過濾 →
   `detector/trajectory_fitter.py` Sequential RANSAC 擬合 → 情境判定（T座／投球／
   側拋）+ Contact Frame。
5. **物理計算**：`physics/pipeline.py::compute_ball_physics()` 統一算 Contact
   Frame／Velocity Window／Launch Angle／Exit Velocity。
6. **軌跡驗證**：`physics/trajectory_validator.py` 事後合理性檢查（只影響除錯
   面板與是否抑制繪製，不影響數值本身）。
7. **疊圖輸出**：`visualizer/overlay.py` + `main.py::_DebugOverlay` 畫框線／
   軌跡／數據面板，寫出 `*_tracked.mp4`。

## 文件索引（`md/`）

- `md/PROJECT_SPEC.md` — 專案規格總覽（目標平台、MVP 功能、編碼規範）
- `md/CONTEXT.md` — 領域術語表（Ball Physics 相關概念的精確定義）
- `md/PHYSICS_VALIDATION_PRINCIPLES.md` — 物理驗證原則
- `md/PROJECT_PROGRESS_0705.md` — 開發進度記錄
- `md/待決問題.md` — 待決問題權威追蹤清單（最新，優先參考）
- `md/BAT_CLOUD_MERGE_NOTES.md` — BatCloudTracker 合併筆記
- `md/prd.md` — 產品需求文件

## 已知限制

- `tools/homography.py` 整套本壘板角點提取／Homography 子系統目前**未被
  main.py 引用**，純屬診斷工具，是否要整合進主線仍待決定。
- `detector/pose_tracker.py` 姿勢追蹤為實驗性功能，未整合進 main.py。
