# Baseball Hitting Tracker — 專案進度紀錄

**最後更新：2026-07-26**（main.py 與 test_side_view_ev.py 物理計算邏輯同步、main.py 移除 Homography 死碼、EV 改為 xy 合成法、YOLO 模型更新為 v7；問題追蹤細節見 待決問題.md，本文件僅同步更新直接受影響的架構描述）

---

## 一、專案目標

從側面架設的攝影機錄製的棒球打擊影片，自動計算：

| 指標 | 英文縮寫 | 說明 |
|---|---|---|
| 擊球初速 | EV (Exit Velocity) | 球離棒後的初始速度（km/h） |
| 仰角 | LA (Launch Angle) | 球離棒後的飛行仰角（degrees） |

**計算原理**：
- Launch Angle：像素座標中 X/Y 移動比例計算仰角（`calculate_launch_angle()`，割線法）。原本設計的攝影機仰角視差補正機制（`_correct_for_camera_elevation()`）仍存在於 `physics/angle.py`，但 2026-07-26 起 `main.py`／`test_side_view_ev.py` 皆不再傳入非零的攝影機仰角，實務上未啟用
- Exit Velocity：側拍 xy 合成法——`vx`、`vy` 同一組取樣點各自做 Theil-Sen 求斜率，以本壘板 bbox 已知寬度做像素↔公制簡單定標，直接 `√(vx²+vy²)` 合成速度大小（2026-07-26 起不再是「只取 image-x 再除以 cos(仰角)」，理由與比較見待決問題.md 一、已解決問題表格第 6 項；不使用完整 Homography 矩陣轉換，避開側拍下垂直分量映射誤差）
- Homography 矩陣目前僅供診斷工具（`tools/check_homography.py`、`tools/homography_diag.py`）與 `tests/test_homography.py` 使用，`main.py`／`test_side_view_ev.py` 的生產路徑皆不再依賴它

---

## 二、使用者條件

### 硬體 / 拍攝規格
- **攝影機位置**：球場側面，與本壘板同高或略低
- **影片幀率**：主力測試影片為 **240fps**（iPhone MOV 格式，實際約 209.7fps）
- **解析度**：1080p（1920×1080）
- **打擊模式**：目前主測為 T-seat 打擊（ball 靜置於 T 架，揮棒擊出）；投球 / 側拋情境管線已支援但實測樣本較少

### 軟體環境
- Python 3.12（Windows，Git Bash / PowerShell 執行）
- YOLO 模型：`Training/best_v6.pt`（自訓練，v6 版）
- 偵測類別：`Ball`, `Bat`, `Bat-top`, `Bat-buttom`（typo 保留）, `Home-plate`
- 主要執行腳本：`tests/test_side_view_ev.py`、`main.py`（兩者 EV 計算皆為側拍單軸法）

### 已知 YOLO 偵測限制（240fps 影片）
- **高速球在 240fps 下有嚴重運動模糊**，YOLO 對飛行中的球偵測率偏低
- 靜止背景物（頭盔、地面球等）容易被誤偵測為 "Ball"，形成密集靜止叢集
- **架構層級的應對方式已改變**：不再依賴 Legacy 版本的光流（Optical Flow）逐幀補追，改用 CloudBallTracker 的離線點雲式分析——收集全片段偵測後做密度聚類移除靜止背景，再用 Sequential RANSAC 對稀疏的真實飛行點做穩健軌跡重建

---

## 三、主線架構切換說明（重要）

本專案**已完全放棄 Legacy 即時追蹤器路線**（`ball_tracker.py` / `ball_selector.py` / `tstand_detector.py` / `ball_tracker_batch.py`，Kalman + RANSAC gate + Hough + 光流的逐幀線上追蹤），這些檔案已於 `181a8bd`（2026-07-17）自 repo 移除。

**目前唯一主線為 CloudBallTracker + BatCloudTracker + TrajectoryFitter 的離線點雲式批次分析**：不逐幀線上追蹤，而是先收集整段影片的全部 YOLO 偵測，再用全域統計方法（密度聚類、Sequential RANSAC）離線重建軌跡。這條路線本質上是 `ball_tracker_batch.py`（原「已暫停」的實驗性批次追蹤器）換了核心演算法後的復活版：Batch 版當年失敗的原因是「完全依賴 YOLO 偵測、無補救機制」，CloudBallTracker 用「靜止叢集辨識 + 場景判斷 + Sequential RANSAC 遞迴擬合」取代了原本設想的光流補幀，解決同一個根本問題。

`BarrelTracker`（`detector/barrel_tracker.py`）為唯一保留下來的舊版模組，但角色已限縮為 Phase 1 逐幀收集棒頭候選位置，供 Phase 2 接觸幀判定失敗時的**備援訊號**（`physics.contact.find_contact_frame`），不再是主要棒頭追蹤方式。

---

## 四、各檔案目的

### `config/`
| 檔案 | 目的 |
|---|---|
| `constants.py` | 全專案數值常數集中管理，含 fps 自適應縮放邏輯（`apply_fps_scaling()`）。所有 frame-based 常數在此定義基準值，runtime 依實際 fps 自動重新校正。含 `CLOUD_*`（CloudBallTracker）、`BAT_CLOUD_*`（BatCloudTracker）、`TRAJFIT_*`（TrajectoryFitter）三組獨立校準的常數群 |

### `detector/`
| 檔案 | 目的 |
|---|---|
| `yolo_tracker.py` | YOLO 推論入口 + class-aware NMS 後處理；定義 `Detection`、`FrameDetections` dataclass |
| `ball_tracker_cloud.py` | **主線球體點雲追蹤器**（`CloudBallTracker`）；Stage 0 P25 大框過濾 → Stage 1 XY 密度聚類識別靜止叢集 → Stage 2 靜止叢集移除 → Stage 3 時序段落過濾 → Stage 4 情境判斷（T座/投球/側拋）→ TrajectoryFitter 接手做 Sequential RANSAC 遞迴擬合 → T 座球鎖定 + 接觸幀確認（獨立於 TrajectoryFitter，供交叉驗證） |
| `trajectory_fitter.py` | **軌跡穩健擬合核心**（`fit_trajectory_segments`）；Sequential RANSAC 遞迴擬合（先試直線再試二次曲線），逐輪取出共識點集組成 `FittedSegment`，並依因果時間序辨識段落角色（來球段 / 飛行段 / 反彈段）。取代舊版 Stage 5 全域速度 MAD 與三點夾角撞擊截斷法 |
| `bat_tracker_cloud.py` | **棒頭（Bat-top）點雲式軌跡過濾器**（`BatCloudTracker`）；沿用 CloudBallTracker 的密度聚類手法識別「持棒站姿 / 跟隨動作定格」靜止叢集並移除，再用純速率 MAD 過濾孤立誤偵測跳點；另含轉向角掃描 + 圓弧倒推合併機制偵測揮棒視窗（詳見 `BAT_CLOUD_MERGE_NOTES.md`） |
| `barrel_tracker.py` | 棒頭追蹤（舊版模組，已保留但角色限縮為接觸幀判定的備援訊號來源，見上方「主線架構切換說明」） |
| `centroid.py` | bbox 中心點計算、`BallPosition` dataclass 定義、`ball_position_from_bbox()` |
| `pose_tracker.py` | 打者姿態追蹤（目前為輔助診斷用） |

### `physics/`
| 檔案 | 目的 |
|---|---|
| `contact.py` | 接觸幀判定備援路徑（`find_contact_frame`）：CloudBallTracker 內建三訊號判定失敗時，走距離最小值 / 方向改變後備法 |
| `velocity.py` | EV 計算；`calculate_exit_velocity_side_view()` 為兩條主線腳本（`test_side_view_ev.py`、`main.py`）唯一使用的 EV 函式——**2026-07-26 起**改為 `vx`／`vy` 同一組取樣點各自 Theil-Sen 求斜率、以 HP bbox 寬度做簡單 px/cm 定標（定標常數改用 `HOME_PLATE_SIDE_VIEW_FULL_LENGTH_CM=43.14cm`，見待決問題.md），直接 `√(vx²+vy²)` 合成速度大小，不再是「只測 image-x 再除以 cos(仰角)」；`EV_MIN_R2`／覆蓋率／輸出範圍等可信度 gate 已移除（假設輸入已是上游驗證過的軌跡點）；`calculate_exit_velocity()`（Homography 公制法 + `_pixel_velocity_to_cm_via_jacobian()`）目前未被任何主線程式呼叫，是尚未清除的已棄用函式；Theil-Sen 中位數回歸抗離群值 |
| `angle.py` | LA 計算；像素 Y/X 位移比例法（割線法）+ 攝影機仰角視差補正機制（`_correct_for_camera_elevation()`，2026-07-26 起兩條主線皆不再傳入非零仰角，機制保留但未啟用） |
| `trajectory.py` | 棒頭軌跡輔助函式（P25 過濾 `filter_dets_by_p25`、缺口補全 `fill_bat_top_gaps_with_body`、揮棒軌跡萃取 `extract_bat_swing_trajectory`）與圓弧擬合工具（`_ransac_circle_fit`，供 `bat_tracker_cloud.py` 合併驗證使用） |
| `trajectory_validator.py` | 軌跡品質驗證器（PASS/FAIL）：接觸前模式分類、速度突刺偵測、方向一致性 R²、覆蓋率 gate |

### `geometry/`
| 檔案 | 目的 |
|---|---|
| `homography.py` | 本壘板 Homography 核心：角點提取（Canny / KPT5 / BBOX 三種品質等級）、`compute_homography()`、`validate_homography()`（runtime 自動驗證公制誤差）、`pixels_to_metric()`、攝影機仰角估算、視差補正係數 |

### `visualizer/`
| 檔案 | 目的 |
|---|---|
| `overlay.py` | 輸出影片標注：球軌跡、棒軌跡、EV/LA 數據面板（自適應縮放）、接觸幀箭頭 |

### `tests/`
| 檔案 | 目的 |
|---|---|
| `test_side_view_ev.py` | **主測腳本**；側拍單軸法。三階段管線：Phase 1 單趟 YOLO 原始偵測，Phase 2 CloudBallTracker + BatCloudTracker + 物理計算，Phase 3 輸出標注影片 |
| `test_cloud_tracker.py` | CloudBallTracker 單獨測試腳本；輸出點雲分布圖（PNG）+ 標注影片 |
| `test_bat_trail_cloud.py` | BatCloudTracker 單獨測試腳本；逐 stage 分類視覺化 + 揮棒視窗標示 + 點雲分布圖 |
| `test_homography.py` | Homography 偵測工具（跳幀早退版）：跳幀取樣 → YOLO 找 HP → KPT5/Hough/加權 Canny → 首幀通過幾何驗證即鎖定 |
| `extract_frames.py` | 從影片提取幀圖像（訓練資料準備） |

### `tools/`
| 檔案 | 目的 |
|---|---|
| `raw_yolo_test.py` | Raw YOLO 偵測視覺化（不經任何追蹤過濾，直接輸出標注影片） |
| `check_homography.py` | Homography 品質逐幀診斷 |
| `homography_diag.py` | Homography 詳細診斷工具 |
| `audio_detect.py` | 音訊偵測擊球聲（bandpass filter + adaptive threshold，輔助 contact 幀驗證與訓練資料裁切） |
| `extract_frames.py` | 訓練幀提取 |
| `claude_reviewer.py` | 程式碼審查輔助 |

### 根目錄
| 檔案 | 目的 |
|---|---|
| `main.py` | 整合入口；同樣已改用 CloudBallTracker + BatCloudTracker 作為追蹤管線，EV 計算方式與 `test_side_view_ev.py` 相同（2026-07-26 起為側拍 xy 合成法，見待決問題.md 一、已解決問題表格第 6 項）。**2026-07-26 更新**：與 `test_side_view_ev.py` 的物理計算邏輯已同步（contact 備援、velocity window、Launch Angle 皆一致），Homography 角點擷取管線已確認為死碼並移除，`main.py` 現在完全不使用 Homography |
| `extract_training_frames.py` | 訓練資料幀提取（根目錄版），已串接 `tools/audio_detect.py` 的擊球聲裁切結果 |
| `test_detection_frames.py` | 偵測幀測試 |
| `Training/best_v7.pt` | 自訓練 YOLO 模型（v7，目前主力版本，`main.py`／`test_side_view_ev.py`／`test_cloud_tracker.py` 皆已指向此版） |

---

## 五、計算管線（`test_side_view_ev.py` 主線，側拍單軸法）

```
影片
  │
  ▼ Phase 1（單趟 YOLO 原始偵測，不做追蹤過濾）
  ├── Ball：收集全部偵測（非 best-per-frame，供 CloudBallTracker 密度分析）
  ├── Bat-top / Bat-body：best-per-frame（供 BatCloudTracker）
  ├── BarrelTracker：逐幀收集棒頭候選（接觸幀判定備援用）
  └── Home-plate：首次高信心度偵測即鎖定 bbox
  │
  ▼ Phase 2（純計算，不再呼叫 YOLO）
  ├── CloudBallTracker 六階段管線
  │     Stage 0 P25 大框過濾
  │     Stage 1 XY 密度聚類 → 靜止叢集識別
  │     Stage 2 靜止叢集移除
  │     Stage 3 時序段落過濾
  │     Stage 4 情境判斷（T座 / 投球 / 側拋）
  │     TrajectoryFitter：Sequential RANSAC 遞迴擬合 → 段落角色辨識
  │     T 座球鎖定 + 接觸幀確認（三訊號中位數，供交叉驗證）
  ├── BatCloudTracker：靜止叢集移除 → 雜訊過濾 → 揮棒視窗偵測（轉向角掃描 + 圓弧合併）→ 速率離群值移除
  ├── Contact 幀判定：優先採用 CloudBallTracker 內建判定，失敗才退回 physics.contact 備援路徑
  ├── Launch Angle（像素比例法 + 視差補正）
  └── Exit Velocity（側拍單軸法：image-x 速度 + HP bbox 定標 + cos 仰角補正）
  │
  ▼ Phase 3（輸出）
  └── 標注影片（球軌跡 + 棒軌跡 + EV/LA 面板）
```

`main.py` 為平行的第二條入口，EV 計算方式與上述相同（側拍單軸法），Homography 僅用於估算攝影機仰角供 LA 幾何補正，追蹤管線同樣是 CloudBallTracker + BatCloudTracker。

---

## 六、目前正在修改 / 調查的部分

### 6-1 TrajectoryFitter 校準狀態 — 進行中

`TRAJFIT_RESIDUAL_MIN_THRESHOLD_PX = 2.0` 等門檻目前僅為單支影片（IMG_3289）觀察值，尚未跨影片校準。詳見 `REVIEW_2026-04-06.md` 第 9 點。

### 6-2 CloudBallTracker 接觸幀偵測的邊界情境 — 已記錄方向，尚待實測驗證

包含 T座靜止叢集可能吞掉接觸後最初幾幀、非T座情境完全沒有獨立 contact_frame 訊號、TrajectoryFitter 段落檢查不合格會整批永久丟棄不重試、T座 contact_frame 實際只有兩訊號在運作且無空間吻合度驗證。詳見 `REVIEW_2026-04-06.md` 第 12–15 點。

### 6-3 BatCloudTracker 片段合併機制 — 已知限制

轉向角掃描切出的片段用圓弧倒推驗證做合併，但貪婪演算法會被小片段卡住合併鏈（見 `detector/BAT_CLOUD_MERGE_NOTES.md` 已知問題 1），且所有合併門檻數值都只在單支影片測試過、未校準（已知問題 3）。

### 6-4 YOLO 訓練資料 — 已知問題，待改善

**問題**：
- 靜止背景物（頭盔、地面球）容易被誤偵測為 "Ball"（FP）
- 240fps 高速飛行球偵測率偏低（FN）；這也是 CloudBallTracker 選擇離線點雲式分析而非即時逐幀追蹤的主要原因——即時追蹤在飛行球幾乎不被偵測時沒有足夠訊號可用，離線批次分析可以綜觀全片段統計特徵繞開這個限制

**待辦**：
- 收集更多 240fps 運動模糊球的標注訓練樣本
- 改善靜止物 FP 問題（可能需要負樣本增強）

---

## 七、已知問題清單

| # | 問題 | 影響 | 狀態 |
|---|---|---|---|
| 1 | YOLO 對 240fps 飛行球偵測率偏低 | CloudBallTracker 靠密度聚類+RANSAC 從稀疏點重建軌跡，偵測率越低，可用共識點越少 | 部分緩解（架構已換，資料問題未解決） |
| 2 | Homography 在本壘板遮擋（土、揮棒殘影）時品質下降 | 部分影片 EV 輸出 N/A 或誤差大 | 持續監控 |
| 3 | **2026-07-26 更新**：EV 計算已從「只測 image-x 再除以 cos(仰角)」改為「vx/vy 同一組取樣點合成」（見待決問題.md F1），但「側拍下 scale_x=scale_y」這個幾何前提仍尚未做雷達槍等 ground truth 驗證；另新發現全片單一 fps 假設（`cv2.CAP_PROP_FPS` 全片平均值）與影片實際逐幀時間戳有系統性落差，已改用固定 240fps 部分緩解（見待決問題.md F2） | 高仰角擊球 EV 準確度未知；fps 假設誤差已知約 9.8%（單一案例，未跨影片驗證） | 詳見待決問題.md F 節，本文件不重複維護。`REVIEW_2026-04-06.md` 第 10 點記錄的 Jacobian 垂直分量誤判問題，其對象函式 `calculate_exit_velocity()` 目前未被任何主線程式呼叫，該問題本身尚未過時但已不在目前主線的執行路徑上 |
| 4 | TrajectoryFitter 殘差門檻僅單影片校準 | 靜止或雜訊環境不同的影片可能誤判離群值 | 待跨影片資料校準 |
| 5 | CloudBallTracker/BatCloudTracker 多項邊界情境僅記錄方向未實測 | 見上方 6-2、6-3 | 待實測 |

---

## 八、測試影片

| 影片 | 說明 |
|---|---|
| `Test_vod/0703 Indi/IMG_3293.MOV` | 240fps T-seat 打擊，主力測試影片 |
| `Test_vod/793888764.103208.mp4` | 預設範例影片（`test_side_view_ev.py` 預設） |
| `Test_vod/793471920.220478*.mp4` | CloudBallTracker 相關診斷輸出（homography / physics / tracked 標注版本） |

---

## 九、執行指令

```bash
# 主線分析（側拍單軸法，CloudBallTracker + BatCloudTracker）
py -3.12 tests/test_side_view_ev.py "Test_vod/0703 Indi/IMG_3293.MOV" --conf 0.25

# CloudBallTracker 單獨測試（點雲圖 + 標注影片）
py -3.12 tests/test_cloud_tracker.py "Test_vod/0703 Indi/IMG_3293.MOV" --conf 0.25

# BatCloudTracker 單獨測試（揮棒視窗視覺化）
py -3.12 tests/test_bat_trail_cloud.py "Test_vod/0703 Indi/IMG_3293.MOV" --conf 0.25

# Raw YOLO 偵測視覺化（不過濾）
py -3.12 tools/raw_yolo_test.py "Test_vod/0703 Indi/IMG_3293.MOV"

# 整合入口（main.py，同側拍單軸法，無 --conf 參數）
py -3.12 main.py "Test_vod/0703 Indi/IMG_3293.MOV"
```
