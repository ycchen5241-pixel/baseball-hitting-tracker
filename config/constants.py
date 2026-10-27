# config/constants.py
# 所有數值常數集中管理，嚴禁在其他模組使用魔術數字

# ── 影片設定 ───────────────────────────────────────────────
VIDEO_FPS = 240               # 錄影幀率（fps）
MS_TO_KMH = 3.6               # m/s 轉 km/h 換算係數

# ── 本壘板標準尺寸（公制，單位：cm）─────────────────────────
HOME_PLATE_FRONT_WIDTH_CM = 43.18    # 前端寬度（17 inches）
HOME_PLATE_SIDE_LENGTH_CM = 21.59    # 兩側長度（8.5 inches）
HOME_PLATE_REAR_DIAGONAL_CM = 30.48  # 後方斜邊（12 inches）

# 側拍本壘板全長：前緣中點直下到最後方尖點（2026-07-26 依角度換算）。
# 前緣中點→兩側邊底部 8.5in（21.59cm，精確）＋ 兩側邊底部→尖點的等腰直角
# 三角形高（12in 斜邊、頂點直角，12/√2in≈8.485in≈21.55cm），
# 合計 16.985in ≈ 43.14cm。專供側拍單軸法 EV 定標使用（相機水平方向對應
# 本壘板前後深度，不是 8.5in 側邊），不影響 HOME_PLATE_SIDE_LENGTH_CM
# 原本供 homography 角點幾何使用的定義。
HOME_PLATE_SIDE_VIEW_FULL_LENGTH_CM = 43.14

# ── 本壘板角點提取設定（Homography）─────────────────────────
HOME_PLATE_BLUR_KERNEL = 5           # GaussianBlur kernel size（奇數）
HOME_PLATE_APPROX_EPSILON = 0.04     # approxPolyDP epsilon 係數
# 最大輪廓面積 / ROI 面積的最小比例（用於過濾雜訊輪廓）
HOME_PLATE_MIN_CONTOUR_RATIO = 0.05
# 角點凸包面積的最小值（px²），防止角點退化至同一位置
HOME_PLATE_MIN_CORNER_AREA_PX = 100.0
# Auto-Canny：由梯度幅值的 Otsu 閾值自動決定 Canny 高閾值，適應各種光線/遮擋條件
HOME_PLATE_CANNY_LOW_HIGH_RATIO: float = 0.5    # low = high × ratio（標準 1:2 比例）
HOME_PLATE_CANNY_MIN_HIGH_THRESH: float = 20.0  # Otsu 高閾值下限（防止均勻影像輸出 0）
HOME_PLATE_CANNY_RETRY_FACTOR: float = 0.7      # 每次重試將高閾值乘以此係數（偵測更多邊緣）
HOME_PLATE_CANNY_MAX_RETRIES: int = 3           # 最多重試次數（含初次共 4 次嘗試）
# ROI 高度最小值：低於此值時本壘板嚴重透視壓縮，Canny 角點不可靠，直接退到 BBOX
HOME_PLATE_CANNY_MIN_ROI_HEIGHT_PX: int = 25
# 多輪廓合併策略：遮擋導致輪廓破碎時，合併前 N 大輪廓後取凸包再做角點提取
HOME_PLATE_MULTI_CONTOUR_TOP_N: int = 3         # 合併的輪廓數量上限

# ── YOLO 推論設定 ─────────────────────────────────────────
YOLO_CONFIDENCE_THRESHOLD = 0.25  # 最低信心度閾值
YOLO_IOU_THRESHOLD = 0.7          # NMS IoU 閾值（class-aware）

# ── 偵測類別名稱（與 Training/best_v7.pt 訓練標籤完全一致）─────
# 注意：模型中 Bat-buttom 為原始標註的 typo，不修正以確保對應正確
CLASS_BALL = "Ball"
CLASS_BAT = "Bat"
CLASS_BAT_HEAD = "Bat-top"
CLASS_BAT_TAIL = "Bat-buttom"
CLASS_HOME_PLATE = "Home-plate"

# 有效類別集合（排除雜訊類別 "0"）
VALID_CLASSES = {
    CLASS_BALL, CLASS_BAT, CLASS_BAT_HEAD,
    CLASS_BAT_TAIL, CLASS_HOME_PLATE,
}

# ── 霍夫圓偵測設定（HoughCircles）────────────────────────
# 在 YOLO bbox 內精確定位球心，並作為信心度增強的依據
HOUGH_DP = 1.2             # 累加器解析度（1=原尺寸，>1縮小）
HOUGH_MIN_DIST = 20        # 相鄰圓心最小距離（pixels）
HOUGH_PARAM1 = 60          # Canny 高閾值
HOUGH_PARAM2 = 15          # 累加器閾值（越小越敏感）
HOUGH_MIN_RADIUS = 3       # 球最小半徑（pixels）
HOUGH_MAX_RADIUS = 50      # 球最大半徑（pixels）
HOUGH_CONFIDENCE_BOOST = 0.75  # 霍夫圓確認時的最低合成信心度
HOUGH_ROI_PADDING_PX: int = 5  # bbox 擴展量（避免圓被裁到邊緣）
HOUGH_BLUR_KERNEL: int = 5     # ROI 高斯模糊 kernel size（奇數）

# ── RANSAC 軌跡離群值過濾設定 ─────────────────────────────
# 維護最近 WINDOW_SIZE 個已接受位置，以 RANSAC 線性回歸過濾突然跳位的偵測
RANSAC_WINDOW_SIZE = 6              # 滑動視窗大小（幀數）
RANSAC_MIN_SAMPLES = 3              # 啟動 RANSAC 所需最少確認樣本數
RANSAC_INLIER_THRESHOLD_PX = 40.0  # RANSAC 模型擬合內點距離閾值（pixels）
RANSAC_GATE_THRESHOLD_PX = 200.0   # 新量測離群值排除距離閾值（pixels）
# 緩衝最後一個確認位置與當前幀的最大允許間隔；超過則視緩衝為過期並停用閘門
RANSAC_MAX_GAP_FRAMES = 10
# 連續拒絕此幀數後判定為軌跡方向突變（如擊球），立即重置緩衝開始新軌跡
# 隨 fps 反向縮放（fps 越高 → 幀數越少 → 更快重置，減少 contact 後無資料窗口）
# 30fps=3幀(100ms)，240fps=1幀(4ms)
RANSAC_DIRECTION_CHANGE_FRAMES: int = 3

# ── 卡爾曼濾波器設定 ──────────────────────────────────────
# 狀態：[x, y, vx, vy]；量測：[x, y]（常速度模型）
KALMAN_PROCESS_NOISE = 1e-2    # 過程雜訊協方差（越大越跟隨量測）
KALMAN_MEASUREMENT_NOISE = 5e-1  # 量測雜訊協方差（越大越依賴預測）
KALMAN_ERROR_INIT = 1.0        # 初始誤差協方差
# 球棒頭診斷用 Kalman（test_bat_trail）：設高以跟隨急轉彎，與球的低噪設定不同
KALMAN_BAT_PROCESS_NOISE = 80.0
KALMAN_BAT_MEASUREMENT_NOISE = 10.0
KALMAN_BAT_ERROR_INIT = 100.0      # 初始誤差協方差（高值使 KF 快速收斂至量測）

# ── 運動球選取設定 ────────────────────────────────────────
# 連續幀間球位移 > 此值 → 判定為運動中的球（靜止球 ≈ 0~5px 相機抖動）
# 30fps 場景：100km/h 的球約 100px/frame；設 15px 可有效分離
BALL_MOTION_THRESHOLD_PX = 15.0

# ── 揮棒偵測設定 ─────────────────────────────────────────────
SWING_VELOCITY_THRESHOLD_PX: float = 15.0   # barrel 速度判定揮棒開始（px/frame，基準 30fps）
SWING_END_THRESHOLD_PX: float = 5.0         # barrel 速度判定揮棒結束（px/frame，基準 30fps）
SWING_END_SUSTAINED_FRAMES: int = 5         # 速度低於閾值持續幾幀才判定結束（基準 30fps）

# ── 擊球幀判定 ────────────────────────────────────────────────
CONTACT_MIN_DIRECTION_CHANGE_DEG: float = 30.0  # 球方向改變最低確認閾值（degrees）
CONTACT_DIRECTION_SUSTAINED_FRAMES: int = 2     # 方向改變需持續幾幀才確認（基準 30fps）
CONTACT_VECTOR_FRAME_OFFSET = 2                 # 計算前後向量時使用的幀偏移量
CONTACT_BAT_PROXIMITY_FRAMES = 3               # 後備法：bat_head 出現幀數容忍範圍（±frames）
CONTACT_WINDOW_MARGIN: int = 10                # 接觸幀在揮棒視窗外的容忍幀數（基準 30fps）
# 球心到棒頭 bbox 最近邊緣距離閾值（pixels）；距離最小值法與空間接近驗證共用
CONTACT_PROXIMITY_THRESHOLD_PX: float = 150.0

# ── T 座偵測設定 ───────────────────────────────────────────────
TSTAND_MIN_STATIC_FRAMES: int = 15          # 判定為 T 座靜止球的最少連續幀數（基準 30fps）
TSTAND_CLUSTER_RADIUS_PX: float = 20.0     # T 座球聚類半徑（px，不隨 fps 縮放）
TSTAND_CONTACT_PROXIMITY_PX: float = 100.0  # 棒頭與靜止球距離上限（px，不隨 fps 縮放）
TSTAND_BAT_GRACE_FRAMES: int = 3           # 靜止球消失後等待棒頭偵測的寬容幀數（30fps 基準）
TSTAND_HP_PROXIMITY_PX: float = 120.0      # T 座球必須在本壘板中心此距離內才納入叢集（px）

# ── 物理計算採樣設定 ───────────────────────────────────────
# 以時間視窗（ms）而非固定幀數控制採樣範圍：fps 越高自動取更多幀，維持相同物理時窗
# 50ms 視窗：氣動阻力造成速度誤差 < 1.5%（150km/h 下約 0.6m/s）
VELOCITY_SAMPLE_WINDOW_MS: float = 200.0  # 初速迴歸目標時間視窗（毫秒）
VELOCITY_SAMPLE_MIN_FRAMES: int = 3       # 初速迴歸最少幀數（線性迴歸穩定性下限）
ANGLE_SAMPLE_WINDOW_MS: float = 50.0     # 仰角迴歸目標時間視窗（毫秒）
ANGLE_SAMPLE_MIN_FRAMES: int = 3         # 仰角迴歸最少幀數
ANGLE_REGRESSION_DEGREE: int = 1         # 仰角估算線性回歸的多項式階數

# ── 視差修正設定 ─────────────────────────────────────────────
# 視差原理：Homography 校正於地面平面（z=0），對高度 h 的球體測速會因透視
# 而過高估計，比例因子為 H_cam/(H_cam - h_ball)。
# 修正公式：V_true = V_measured × (H_cam - h_ball) / H_cam
# 典型誤差：H_cam=150cm, h_ball=80cm → 未修正值約為真值的 2.1 倍
PARALLAX_CORRECTION_ENABLED: bool = True
PARALLAX_BALL_HEIGHT_CM: float = 80.0     # 預設擊球高度（好球帶中點，離地約 80cm）
PARALLAX_CAMERA_HEIGHT_CM: float = 150.0  # 預設攝影機高度（標準腳架，約 150cm）
# 近端/遠端邊長比最小有效值；低於此值時視為幾乎俯視，仰角估算無意義
PARALLAX_ELEVATION_MIN_RATIO: float = 1.02

# ── 靜止球鎖定偵測設定 ────────────────────────────────────────────
# 追蹤器若連續偵測到「幾乎不動」的球，自動判定為背景假偵測，
# 並在更遠的新偵測出現時強制重置，讓運動中的球接管追蹤權。
# MIN 值為感測器噪聲下限：240fps 時原始值縮放至 1.25px，低於噪聲，需設下限
MIN_BALL_STATIC_THRESHOLD_PX: float = 5.0   # BALL_STATIC_THRESHOLD_PX 縮放下限
BALL_STATIC_THRESHOLD_PX: float = 10.0  # 幀間位移 < 此值視為靜止（含些微晃動）
BALL_STATIC_MAX_FRAMES: int = 6         # 連續靜止幀數上限（基準 30fps；fps 越高幀數越多）
BALL_STATIC_ESCAPE_MIN_PX: float = 30.0 # 靜止鎖定時，新偵測距靜止點 > 此值才強制接管

# ── Ball bbox 尺寸過濾（拒絕可能跨多顆球的大 bbox 誤偵測）────────────
# 單顆棒球在正常拍攝距離下 bbox 邊長通常不超過 60px
# 超過此閾值幾乎確定是多球合並偵測或背景物誤偵測，直接拒絕
BALL_MAX_BBOX_SIDE_PX: float = 80.0

# ── Bat-top bbox 自適應尺寸過濾（P25 × 倍率）──────────────────────────
# 使用「全片 25th percentile × 倍率」而非硬性像素值，自動適應不同拍攝距離/解析度。
# 原理：正常棒頭偵測的 bbox 尺寸分佈集中在低端（P25 接近真實正常值）；
# 異常大框（背景誤偵測、模糊殘影）尺寸通常是正常值的 5–8 倍，
# 倍率 4.0 在兩者之間留有充裕安全邊界，不受攝影機遠近影響。
BAT_HEAD_BBOX_P25_RATIO: float = 4.0

# ── 幀級靜止預過濾設定（Kalman 初始化前過濾背景球）────────────────
# 在 YOLO 偵測進入追蹤器前，先識別並排除持續出現在同一位置的靜止/慢速背景球
# 聚類半徑：YOLO 對同一靜止物的偵測位置變異（bbox 雜訊），30fps 下約 20–30px
BALL_PREFILTER_CLUSTER_RADIUS_PX: float = 25.0
# 連續出現此幀數後視為靜止背景（30fps × 4 幀 = 133ms）
# 飛行球在 133ms 內至少移動數百像素，不會累積到此閾值
BALL_PREFILTER_MAX_STATIC_FRAMES: int = 4

# ── 光流（Lucas-Kanade）追蹤設定 ────────────────────────────
# YOLO 未偵測到球時，從上一個已知球心以 LK 光流追蹤像素運動
OPFLOW_WIN_SIZE: int = 21            # LK 搜尋視窗大小（pixels，奇數）
OPFLOW_MAX_LEVEL: int = 3            # 影像金字塔層數
OPFLOW_MAX_ERROR: float = 15.0      # 雙向追蹤誤差上限（px），超過視為追蹤失敗
OPFLOW_CONFIDENCE: float = 0.50     # 光流追蹤結果的信心度（高於 FDIF=0.35）

# ── 幀差法補充偵測設定 ──────────────────────────────────────
# 當 YOLO 未偵測到球時，透過幀差法在 Kalman 預測位置附近搜尋運動物體
FRAME_DIFF_THRESHOLD: int = 25           # 像素灰度差異閾值（0–255）
FRAME_DIFF_MIN_AREA_PX: float = 50.0    # 有效運動區塊最小面積（px²）
FRAME_DIFF_MAX_AREA_PX: float = 4000.0  # 有效運動區塊最大面積（px²），排除球棒等大面積干擾
# 以 Kalman 預測位置為圓心的搜尋半徑（px，不隨 fps 縮放）
FRAME_DIFF_SEARCH_RADIUS_PX: float = 120.0
FRAME_DIFF_CONFIDENCE: float = 0.35     # 幀差法偵測的信心度（低於 YOLO，高於純預測 0.0）

# ── 視覺化設定 ────────────────────────────────────────────
TRAJECTORY_COLOR_START = (0, 0, 255)    # BGR：紅色（起點）
TRAJECTORY_COLOR_END = (128, 0, 128)    # BGR：紫色（終點）
TRAJECTORY_LINE_THICKNESS = 3           # 軌跡線條粗細（pixels）
BAT_TRAJECTORY_SMOOTH_WINDOW = 9        # 球棒揮棒軌跡移動平均視窗（幀數，奇數）
TRAJECTORY_DIRECTION_MAX_ANGLE_DEG = 50.0  # 飛行軌跡方向一致性閾值（deg）：偏離主方向超過此值的點不繪製
TRAJECTORY_DIRECTION_ANCHOR_POINTS = 5    # 建立主方向向量所用的前 N 個偵測點數

# ── 球棒揮棒軌跡設定 ──────────────────────────────────────
BAT_TRAJECTORY_COLOR_START = (255, 255, 255)  # BGR：白色（起點）
BAT_TRAJECTORY_COLOR_END = (0, 255, 255)      # BGR：黃色（終點）
BAT_TRAJECTORY_LINE_THICKNESS = 2
# 棒頭位置連續性過濾：幀間位移超過此值 × frame_gap 視為誤偵測（基準 30fps）
BAT_MAX_INTERFRAME_JUMP_PX: float = 200.0
# 連續性過濾的幀間隔上限：防止長時間偵測空白使容忍距離無限膨脹
# 30fps 下 5 幀 = 167ms，最大容忍距離 = 200 × 5 = 1000px（約影片短邊）
BAT_CONTINUITY_GAP_CAP: int = 5
# 球棒軌跡視覺化斷點（test_bat_trail）：相鄰可見點幀差 > 此值則不連線
BAT_TRAIL_VIZ_GAP_FRAMES: int = 8

# ── 球棒靜止黑名單設定（速度分佈判別）─────────────────────────────
# 偵測點在 RADIUS 半徑內累積 ≥ MIN_COUNT 個，且叢集內所有偵測點在
# VELOCITY_LOOKBACK 幀視窗內的最大速度 < MAX_VELOCITY_PX，
# 判定為持續性背景物件（如柱子），從 Bat-top 序列移除。
# 速度閾值區分背景（YOLO 定位噪聲 1–5px/frame）與球棒（揮棒速度 ≥ 10px/frame）。
BAT_STATIC_CLUSTER_RADIUS_PX: float = 25.0
BAT_STATIC_MIN_COUNT: int = 6
BAT_STATIC_MAX_VELOCITY_PX: float = 10.0    # 叢集最大速度上限（px/frame，基準 30fps）
BAT_STATIC_VELOCITY_LOOKBACK: int = 5        # 計算速度的時間視窗（幀數，基準 30fps）
BAT_STATIC_LINK_RADIUS_PX: float = 150.0    # 時序鏈接的最大空間距離（px，不隨 fps 縮放）

# ── 球棒弧線優先篩選設定（select_swing_by_arc）────────────────────────
# 前提：一部影片只有一次完整揮棒擊球。
# 全片中弧長最長且時間跨度合理的偵測點群即為該次揮棒；其餘視為背景。
BAT_CHAIN_MAX_FRAME_GAP: int = 10          # 連接兩點的最大幀間隔（30fps 基準）
BAT_CHAIN_MAX_SPATIAL_PX: float = 400.0   # 連接兩點的最大空間距離（px）
BAT_CHAIN_MIN_ARC_LENGTH_PX: float = 80.0 # 候選鏈段最小弧長（px）；低於此值為靜止物
BAT_CHAIN_MAX_TIMESPAN_RATIO: float = 0.5 # 鏈段時間跨度不超過全片此比例；超過視為背景常駐物

# 局部弧線過濾：相鄰兩點距離超過此值視為「跳躍」，強制切斷 arc segment
# 防止大幅跳躍（誤偵測位移）以累積像素距離讓垃圾 segment 贏過真實揮棒弧段
BAT_ARC_MAX_STEP_PX: float = 100.0
# 局部弧線過濾：速度啟動點偵測（Step A）
# 啟動閾值 = P90 峰值速度 × 此比例；低於閾值的前端連續段視為備揮並裁除
BAT_ARC_ONSET_PEAK_RATIO: float = 0.30
# 啟動閾值下限（px/frame，30fps 基準）；防止靜止影片 P90 過低時誤切
BAT_ARC_ONSET_MIN_SPEED_PX: float = 3.0
# 局部弧線過濾：方向平滑視窗（Step B）
# 滾動平均最近 N 個方向向量，容許單一噪聲點不立即切斷弧段
BAT_ARC_SMOOTH_WINDOW: int = 3
# 局部弧線過濾：最小有效步進距離（Step B）
# 小於此值的相鄰點對視為 YOLO 定位噪聲，不更新方向視窗也不觸發切割
BAT_ARC_MIN_STEP_PX: float = 2.0
# 橋接時 Bat body 位置距最近 Bat-top 的最大容許距離；超過視為背景物件（如柱子）
BAT_BRIDGE_MAX_DIST_FROM_HEAD_PX: float = 200.0
PANEL_POSITION_OFFSET_X = 20            # 數據面板距右邊距（pixels）
PANEL_POSITION_OFFSET_Y = 20            # 數據面板距底邊距（pixels）
LAUNCH_ARROW_LEN_PX: int = 160         # 接觸幀仰角箭頭長度（pixels）

# ── 數據面板自適應縮放設定（依影像短邊動態調整，橫直式通用）──
PANEL_FONT_SCALE_MIN = 0.7     # 字體縮放最小值
PANEL_FONT_SCALE_MAX = 1.4     # 字體縮放最大值
PANEL_FONT_SCALE_REF = 600     # 縮放基準短邊像素數（短邊/此值=字體倍率）
PANEL_LINE_HEIGHT_BASE = 28    # 行高最小值（px）
PANEL_LINE_HEIGHT_DIV = 18     # 行高 = max(BASE, short_side // DIV)
PANEL_PADDING_BASE = 10        # 內距最小值（px）
PANEL_PADDING_DIV = 55         # 內距 = max(BASE, short_side // DIV)

# ── Homography 多幀篩選設定 ─────────────────────────────────
# Canny 成功且信心度達此值時提前停止搜索，不需繼續掃描更多幀
HOME_PLATE_HIGH_CONF_THRESHOLD: float = 0.85
# 每個角點反投影誤差上限（cm）；超過則視為退化 Homography 並捨棄
HOMOGRAPHY_VALIDATE_MAX_ERROR_CM: float = 5.0
# 5 角點 approxPolyDP epsilon 縮放係數（較小以保留後端尖角）
FIVE_CORNER_EPSILON_FACTOR: float = 0.7

# ── Homography 品質等級 ──────────────────────────────────────────
H_QUALITY_HIGH: str = "HIGH"       # KPT5 / CANNY5 / CANNY4+1（精確幾何）
H_QUALITY_MEDIUM: str = "MEDIUM"   # CANNY5→4（降級幾何）
H_QUALITY_BBOX: str = "BBOX_ONLY"  # 僅 bbox 四角（速度輸出 N/A）

# ── 速度仰角補正設定 ──────────────────────────────────────────
# 仰角絕對值小於此值（degrees）時不做補正，視為近水平擊球
VELOCITY_ANGLE_CORRECTION_MIN_DEG: float = 1.0
# 補正下限：cos(60°) = 0.5；超過 60° 的仰角視為異常值，不補正
VELOCITY_ANGLE_CORRECTION_MIN_COS: float = 0.5

# ── 輸出可信度 Gate 設定 ───────────────────────────────────────
# 原本的 EV_MIN_R2／EV_MIN_DETECTION_COVERAGE／EV_MIN_SPEED_KMH／
# EV_MAX_SPEED_KMH（線性度、覆蓋率、輸出範圍三道 gate）已於 2026-07-26
# 與使用者討論後從 physics/velocity.py 移除，理由見該檔案函式內註解：
# 假設輸入軌跡點已是上游驗證完成的最終資料，EV 計算不重複做資料品質
# 判斷。四個常數本身亦已移除，紀錄於此供之後需要時參考。
# 接觸後垂直方向過濾：確認進入下降段所需連續幀數（基準 30fps）
POST_CONTACT_DESCENT_MIN_FRAMES: int = 3
# 每幀最小下降量（px）：低於此值不計入下降段計數（基準 30fps）
POST_CONTACT_DESCENT_THRESHOLD_PX: float = 3.0

# ── 棒球運動公理驗證器設定 ─────────────────────────────────────
# 接觸前模式分類
# TEE：中位速度低於此值視為靜止（與 BALL_MOTION_THRESHOLD_PX 同義）
VALIDATOR_TEE_MAX_SPEED_PX: float = 15.0
# PITCH：中位速度超過此值視為投手投球；方向一致性由 PITCH_DIR_STD 另外驗證
VALIDATOR_PITCH_MIN_SPEED_PX: float = 30.0
# PITCH 方向一致性：位移向量的圓形標準差上限（degrees）
# 投球應幾乎直線前進，30° 已是較寬鬆的容忍值
VALIDATOR_PITCH_DIRECTION_STD_MAX_DEG: float = 30.0

# 接觸後方向一致性（速度大小可通過但方向隨機飛仍應 FAIL）
VALIDATOR_POST_MIN_POINTS: int = 5          # 觸發驗證所需最少軌跡點數
# post-contact 真實偵測（confidence > 0）佔軌跡總幀的比例下限
# 低於此值代表軌跡主要由 Kalman 空預測組成，不應繪製
VALIDATOR_COVERAGE_MIN: float = 0.30
# 某幀速度超過中位速度 × 此倍數 → 速度突刺（兩靜止球誤追的典型特徵）
VALIDATOR_SPEED_SPIKE_RATIO: float = 5.0
# 接觸後前 N 幀豁免速度突刺偵測：擊球瞬間球速物理上必然驟升，不應視為異常
# 同時排除在中位數計算之外，避免後段近零假偵測壓低基準而把真實球速誤判為突刺
# 基準 30fps：10 幀 ≈ 0.33 秒（球離開球棒後的初始飛行段）
VALIDATOR_CONTACT_EXEMPT_FRAMES: int = 10
# 接觸後位移向量的圓形標準差上限（degrees）
# 打出的球應單向飛行；45° 容許像素量測噪聲造成的小偏折
VALIDATOR_POST_DIRECTION_STD_MAX_DEG: float = 45.0

# ── 球棒弧線棒身補全設定 ───────────────────────────────────────────
# 棒頭缺失時以棒身偵測補全揮棒路徑的物理約束
# 棒頭弧需至少此點數才能擬合圓心（少於此值不嘗試延伸）
BAT_BODY_ARC_MIN_HEAD_POINTS: int = 3
# 棒身相對圓心的半徑與棒頭弧半徑的比值範圍
# 棒身中心 ≈ 棒長的 40–60%，設 0.20–0.90 提供寬鬆容錯
BAT_BODY_ARC_RADIUS_MIN_RATIO: float = 0.20
BAT_BODY_ARC_RADIUS_MAX_RATIO: float = 0.90
# 每次補全允許的最大角度步進（radians）
# 快速揮棒單 burst 最多約 135°（π×0.75），超過代表補全對象不是同一揮棒
BAT_BODY_ARC_MAX_ANGLE_STEP_RAD: float = 2.356  # π × 0.75
# 棒身引導式棒頭叢集過濾：棒頭與棒身中位數的最大允許距離（px）
# 柱子等背景誤偵測距真實球棒通常 > 500px，200px 可有效分離兩叢集
BAT_BODY_CLUSTER_RADIUS_PX: float = 200.0
# 幀差分運動驗證：偵測框區域內的最低平均像素差值（0–255）
# 低於此值視為靜止物件（背景牆、柱子）；超過者才有資格優先選擇
# 設 3.0 可過濾靜止背景，同時允許少量照明抖動
MOTION_SCORE_MIN_BAT: float = 3.0

# ── 點雲球體追蹤器設定（CloudBallTracker）────────────────────────────
# Stage 0: P25 大框過濾（球的 bbox）
# P25 × 此倍率為上限；球的正常 bbox 尺寸分布集中，異常大框通常是多球合并或背景誤偵測
BALL_BBOX_P25_RATIO: float = 3.0

# Stage 1: 靜止叢集識別
# 靜止叢集 XY 聚類半徑：需大於 YOLO 對靜止物的位置漂移（典型 50-100px）
# BALL_PREFILTER_CLUSTER_RADIUS_PX（25px）用於 Legacy 逐幀過濾；靜止偵測需要更大半徑
# 以涵蓋整個靜止球在整片影片中的 YOLO 定位抖動範圍
CLOUD_STATIC_CLUSTER_RADIUS_PX: float = 80.0
# 靜止叢集允許的最大 XY 散布（px）；設定為半徑的 2 倍作為安全邊界
CLOUD_STATIC_MAX_SPREAD_PX: float = 150.0
# 叢集偵測率下限（偵測次數 / 時間幀數）；低於此值代表偵測稀疏，不視為靜止背景
CLOUD_STATIC_MIN_RATE: float = 0.3
# 第二輪靜止叢集判斷（在 flight_pts 上、TrajectoryFitter 擬合前執行，
# 攔截第一輪因貪婪聚類順序而漏網的靜止殘留）使用的散布門檻，比第一輪
# 收緊許多——round 1 的候選池混雜大量真實運動軌跡所以門檻要寬，round 2
# 的候選池已經是 Stage 3 存活點，混雜的靜止殘留理論上就是原地不動。
# 尚未用多支影片實際資料校準，數值為初始估計，落地前需驗證
CLOUD_STATIC_ROUND2_MAX_SPREAD_PX: float = 20.0
# 叢集內部依時間切段的相對倍數門檻：空窗超過該叢集自身相鄰間隔中位數的
# 幾倍才視為段落邊界。用相對倍數而非絕對幀數，避免因單一長遮擋（如揮棒
# 經過）拖垮全域密度導致漏判，且不受拍攝距離/fps 影響，不需另外縮放
CLOUD_STATIC_SEGMENT_GAP_RATIO: float = 5.0
# 上面那個相對倍數在中位數間隔很小（物體幾乎每幀都被偵測到）時會過度敏感，
# 幾幀的正常空窗換算倍數就會顯得異常。實際切段門檻改採
# max(中位數 × RATIO, 這個絕對下限)，避免高密度物體被過度切碎
# （基準 30fps；高 fps 自動縮放）
CLOUD_STATIC_SEGMENT_GAP_MIN_FRAMES: int = 30

# Stage 3: 時序雜訊過濾
# 相鄰點幀差超過此值視為新段落（基準 30fps；高 fps 自動縮放）
CLOUD_GAP_THRESHOLD_FRAMES: int = 30
# 段落內點數少於此值視為孤立雜訊，丟棄
CLOUD_MIN_SEGMENT_POINTS: int = 3

# Stage 5: MAD 離群值移除
# threshold = median_rate + K × MAD；K 越大越寬鬆
CLOUD_MAD_K: float = 3.5
# X 方向反轉需要兩側速率皆超過 threshold × 此比例才標記
# 防止 YOLO bbox jitter（1-3px/frame）小波動被誤殺
CLOUD_DIR_REVERSAL_MIN_RATIO: float = 0.3

# Stage 6: T 座球鎖定
# 靜止叢集消失幀與飛行軌跡起始幀的最大容忍幀數（基準 30fps；高 fps 自動縮放）
CLOUD_TSTAND_TIME_TOL: int = 30
# 靜止叢集中心與飛行軌跡起始位置的最大容忍距離（px）
# 球被打出後可能移動數幀才被 YOLO 偵測到，需較大容忍距離
CLOUD_TSTAND_SPACE_TOL_PX: float = 350.0

# 飛行段落截斷（撞擊點偵測）
# 夾角 cos θ 閾值：低於此值才視為銳利夾角（0.0 = 夾角需 > 90°）
CLOUD_IMPACT_ANGLE_COS_THRESHOLD: float = 0.0
# 夾角後需連續維持新方向的點數（排除孤立離群值）
CLOUD_IMPACT_N_CONFIRM: int = 3

# ── 軌跡擬合（TrajectoryFitter，取代 Stage 5 MAD + 飛行段落截斷）───────
# Sequential RANSAC 遞迴擬合：對候選點反覆穩健擬合，取出共識線段，
# 依時間序 + 方向/曲率轉折貼標籤（來球段/飛行段/反彈段），不依賴 contact_frame。
# 尚未用多支影片實際資料校準，數值為初始估計，落地前需驗證。

# RANSAC 隨機取樣迭代次數與種子（沿用 physics/trajectory.py::_ransac_circle_fit
# 的固定種子慣例，結果可重現，不受執行環境亂數狀態影響）
TRAJFIT_RANSAC_ITERATIONS: int = 200
TRAJFIT_RANSAC_SEED: int = 42
# RANSAC 搜尋階段的粗略內點容忍值（px）；只用來從隨機取樣的最小子集模型中
# 選出支持度最高者，真正的離群值判斷交給後續的 MAD 收斂（見下）
TRAJFIT_RANSAC_TOL_PX: float = 15.0
# MAD 收斂：threshold = median(residual) + K × MAD(residual)，K 越大越寬鬆
TRAJFIT_RESIDUAL_MAD_K: float = 3.0
# MAD 門檻下限（px），避免共識點高度共線時 MAD≈0 導致門檻退化為 0
TRAJFIT_RESIDUAL_MIN_THRESHOLD_PX: float = 2.0

# 單一線段合格判斷
# 最少共識點數：低於此數量，median/MAD 統計量不可靠，直接停止遞迴
TRAJFIT_MIN_CONSENSUS_POINTS: int = 5
# 候選集合點數低於此值就不再嘗試擬合新線段（遞迴停止條件）
TRAJFIT_MIN_CANDIDATES_FOR_FIT: int = 5
# 密度下限：點數 / 時間跨度，低於此值視為稀疏巧合共線，不接受
TRAJFIT_MIN_DENSITY_RATE: float = 0.3
# 水平方向判斷單調時，位移小於此值視為 YOLO bbox jitter，不計入方向翻轉判斷
TRAJFIT_MONOTONIC_MIN_DX_PX: float = 2.0
# 段落整體空間跨度下限（px）：RANSAC+MAD 收斂出的共識點，若全程 XY 跨度
# 都小於此值代表這批點根本沒有真正移動（近乎靜止殘留湊巧共線/共曲率），
# 密度/間距/單調性三項檢查對這種情況是盲的（見 CLOUD_STATIC_ROUND2_MAX_SPREAD_PX
# 說明），故需獨立於前三項之外另外把關。與 round 2 靜止叢集判斷用同一個
# 量級的門檻值，尚未用多支影片實際資料校準，落地前需驗證
TRAJFIT_MIN_SPATIAL_SPREAD_PX: float = 20.0

# 邊界候選點（T座起點／反彈終點）併入飛行段驗證：比較加入候選點前後，
# 飛行段曲線在候選點那一側邊界的速度向量方向變化多少（見與使用者討論
# 2026-07-23：以方向角度為主，先不加入速度量級判斷——「是不是同一條
# 線」是純幾何問題，速度快慢是額外的物理假設，不必然成立）。
# 未用多支影片實際資料校準，數值為初始估計，落地前需驗證。
TRAJFIT_EXTENSION_ANGLE_TOL_DEG: float = 5.0

# 飛行段與緊接其後下一段（可能因單幀異常跳點被誤切為 bounce）的整段
# 合併判斷：同一套速度向量方向角度比較邏輯，但比單點併入更嚴格——整段
# 點集重新擬合會平均掉單點雜訊，真正同一條軌跡的方向差異理論上應該
# 更小、更穩定（見與使用者討論 2026-07-28）。獨立命名、不與上面共用，
# 因為兩者驗證對象不同（單點 vs 整段），沒有理由假設同一個容忍值適用
# 於兩種情況。未跨影片校準，數值為初始估計。
TRAJFIT_SEGMENT_MERGE_ANGLE_TOL_DEG: float = 5.0

# ── Bat-top 點雲式追蹤器設定（BatCloudTracker，實驗性 test_bat_trail_cloud）──
# 沿用 CloudBallTracker 的「全片段收集 + 密度聚類」概念，但只有一個追蹤
# 類別（Bat-top），不需跨類別幾何修正；也拿掉方向反轉離群值規則——揮棒的
# 方向反轉（後擺頂點、跟隨動作轉折）是真實訊號，不是雜訊。
# 常數獨立校準，不沿用球的 CLOUD_* 數值。

# Stage 1: 靜止叢集識別（持棒站姿 / 跟隨動作定格）
# 聚類半徑：持棒手震動幅度通常小於背景球在整片影片中的 YOLO 定位漂移
BAT_CLOUD_STATIC_CLUSTER_RADIUS_PX: float = 30.0
BAT_CLOUD_STATIC_MAX_SPREAD_PX: float = 60.0
# 判定靜止所需最少幀數（基準 30fps；站姿持棒可能持續數秒）
BAT_CLOUD_STATIC_MIN_FRAMES: int = 10
# 叢集偵測率下限（偵測次數 / 時間幀數）；低於此值代表偵測稀疏，不視為靜止
BAT_CLOUD_STATIC_MIN_RATE: float = 0.3

# Stage 3: 時序雜訊過濾（基準 30fps；高 fps 自動縮放）
BAT_CLOUD_GAP_THRESHOLD_FRAMES: int = 10
BAT_CLOUD_MIN_SEGMENT_POINTS: int = 3

# Bat-top 缺幀補洞（用 Bat 整體 bbox 當備援資料源，基準 30fps）
# 實測（IMG_3294）：Bat-top 缺幀但 Bat 有偵測到的情況多為 1-2 幀小缺口，
# 動態模糊最嚴重時兩者常常同時缺測，此時補不了，維持缺幀，不強行外推
BAT_CLOUD_GAP_FILL_MAX_FRAMES: int = 10

# Stage 5': 純速率 MAD 離群值移除（不含方向反轉規則）
# threshold = median_rate + K × MAD
BAT_CLOUD_MAD_K: float = 3.0

# 揮棒視窗偵測（轉向幅度 + 方向一致性掃描，取代 RANSAC 圓弧擬合）
# 粗略框出揮棒範圍，供視覺化/診斷用，精度需求低，容許幀級誤差
#
# 設計理由：RANSAC 圓弧擬合對候選點集的微小變動極度敏感（實測 margin
# 差 1 幀，coverage 從 307° 跳到 43°），根本原因是真實揮棒路徑（倒鉤/
# 橢圓形）不是單一固定圓心的圓，硬套圓模型導致擬合本身不穩定。改成不
# 假設任何全域形狀，直接用局部「每一步轉向角度」掃描：找「轉向幅度
# 夠大（不是隨便彎一點就算）且方向持續同一個正負號」的最長連續段，
# 這個條件本身就同時完成「粗定位」跟「形狀驗證」，不需要事後再驗證
# 一次。找到連續段後，在段內找「離起點距離最大」的一幀當終點（延伸
# 到底），自然裁掉收棒回拉，不需要方向反轉角度門檻這種容易被雜訊
# 誤觸發的局部判斷。
#
# MIN_ 開頭常數為感測器噪聲下限：高 fps 時 px/frame 類常數縮放後可能低於
# 可信量測精度（同 MIN_BALL_STATIC_THRESHOLD_PX 既有模式），需設下限
MIN_BAT_CLOUD_TURN_STEP_PX: float = 2.0            # BAT_CLOUD_TURN_MIN_STEP_PX 縮放下限
MIN_BAT_CLOUD_TURN_SPEED_PX: float = 12.0          # BAT_CLOUD_TURN_MIN_SPEED_PX 縮放下限
# 實測發現（IMG_3294）：用相鄰單幀算轉向角，偵測雜訊（bbox 中心跳動）
# 在角度計算上會被放大成劇烈的正負號亂跳，即使在最快揮棒區間也一樣
# （逐幀算出 74.6, -89.7, 33.8... 完全不連貫）。改用較寬的基線（跳過
# BAT_CLOUD_TURN_STRIDE_FRAMES 幀再算方向）把雜訊平均掉；實測 210fps
# 下跳 8 幀左右（約38ms）訊號才變乾淨、方向持續一致。基準值取 1 幀
# （代表固定時間長度，同其他 _FRAME_COUNT_BASES 幀數常數縮放邏輯）。
BAT_CLOUD_TURN_STRIDE_FRAMES: int = 1
BAT_CLOUD_TURN_MIN_STEP_PX: float = 3.0            # 跨 stride 的位移小於此值 × stride，角度不計算（基準30fps，px/frame）
BAT_CLOUD_TURN_MIN_ANGLE_DEG: float = 15.0         # 單步轉向角度門檻，低於此值視為近似直線、中斷連續段
BAT_CLOUD_TURN_MIN_TOTAL_DEG: float = 90.0         # 整段累積轉角總量門檻，避免抓到太短的雜訊段
BAT_CLOUD_TURN_MIN_SPEED_PX: float = 20.0          # 整段內至少要有一點瞬時速度超過此值（px/frame，基準30fps）
BAT_CLOUD_SWING_SECONDARY_RATIO: float = 0.5       # 次要視窗峰值需達主視窗峰值此比例以上，否則判定晃動雜訊
BAT_CLOUD_SWING_MIN_SPAN_PX: float = 150.0         # 視窗空間跨距下限，過濾快但範圍小的晃動（不隨fps縮放）

# 片段合併驗證（相鄰片段是否為同一次揮棒被誤切，用圓弧擬合倒推確認）
# 詳見 detector/BAT_CLOUD_MERGE_NOTES.md——這條路徑在 IMG_3294 測試中
# 未被觸發過，數值未經真實資料驗證，僅為設計時的起始猜測
BAT_CLOUD_MERGE_MAX_GAP_FRAMES: int = 15           # 相鄰片段允許嘗試合併的最大時間間距（基準30fps）
BAT_CLOUD_MERGE_MIN_FRAGMENT_POINTS: int = 3       # 片段至少要幾個點才參與合併驗證（不隨fps縮放）
BAT_CLOUD_MERGE_MIN_INLIER_RATIO: float = 0.6      # 合併驗證，A、B個別內點率門檻（不隨fps縮放）

# 圓弧擬合（合併驗證專用——範圍已先由轉向角掃描縮小到片段級別，不是
# 對整個區段直接擬合，穩定性風險比之前放棄的全域圓弧擬合方案低很多）
BAT_CLOUD_ARC_RANSAC_ITERATIONS: int = 200         # 不隨fps縮放
BAT_CLOUD_ARC_TOL_FRAME_WIDTH_RATIO: float = 0.015 # 不隨fps縮放，相對畫面寬度的容差比例

# ── 批次球體軌跡過濾設定（BatchBallTracker）──────────────────────────
# 靜止背景物聚類：在 BALL_PREFILTER_CLUSTER_RADIUS_PX 範圍內若出現 >= MIN_FRAMES 且
# 空間散佈 < MAX_SPREAD_PX，則判定為靜止物並移除（T 座雙重豁免除外）
BATCH_STATIC_MIN_FRAMES: int = 8             # 判定靜止背景物最少連續幀（基準 30fps）
BATCH_STATIC_MAX_SPREAD_PX: float = 30.0    # 靜止物允許的最大空間散佈（px）
# T 座豁免 — 防線 1：bat_head 曾在靜止群附近掃過
BATCH_TSTAND_BAT_SEARCH_RADIUS_PX: float = 150.0  # bat_head 與靜止群的搜尋半徑（px）
BATCH_TSTAND_FRAME_WINDOW: int = 20          # bat_head 搜尋的時間窗口（基準 30fps）
# 雙向 B1 連續性過濾
BATCH_B1_WINDOW: int = 4                    # 每側用於線性外推的最多點數
BATCH_B1_OUTLIER_THRESHOLD_PX: float = 50.0  # 前後外推誤差均超過此值才視為離群點（px）
BATCH_B1_MAX_ITERATIONS: int = 3            # 迭代過濾最大次數（避免無限移除）
BATCH_B1_MIN_KEEP: int = 3                  # 剩餘點數下限（觸底時停止過濾）

# ── 弧線一致性過濾（filter_bat_by_arc）─────────────────────────────
# RANSAC 圓擬合迭代次數
BAT_ARC_RANSAC_ITERATIONS: int = 200
# 內點率下限：低於此值改用備用方向模式
BAT_ARC_RANSAC_MIN_INLIER_RATIO: float = 0.60
# 弧覆蓋角下限（度）：涵蓋角太小表示點群太集中，圓心估計不穩
BAT_ARC_COVERAGE_MIN_DEG: float = 60.0
# 弧距容差：以畫面寬度的比例表示，兼容不同解析度
BAT_ARC_TOL_FRAME_WIDTH_RATIO: float = 0.015
# 備用模式：相鄰速度向量的最大夾角（度）
BAT_ARC_DIRECTION_MAX_ANGLE_DEG: float = 90.0

# 球棒弧形驗證
VALIDATOR_BAT_MIN_POINTS: int = 4           # 觸發驗證所需最少棒頭點數
# 最小二乘圓擬合 RMSE 上限（px）；揮棒非完美圓形，容錯比球軌跡大
VALIDATOR_BAT_ARC_RMSE_MAX_PX: float = 80.0
# 球棒在圓弧上角度遞增方向一致比例下限（0.75 = 允許 25% 噪聲）
# 確保棒頭在圓上是「依序」掃過，而非隨機跳點
VALIDATOR_BAT_MONOTONE_MIN_RATIO: float = 0.75

# ── FPS 自適應縮放 ─────────────────────────────────────────
# 基準：所有 frame-based 常數以 30fps 校正
# 幀計數類（時間視窗）：fps 越高需要更多幀 → × (fps / 30)
# 每幀位移類（像素閾值）：fps 越高每幀位移越小 → × (30 / fps)
BASE_FPS: float = 30.0          # 常數設計基準幀率；apply_fps_scaling 以此為原點

# 模組載入時自動捕捉各常數當前值，確保與上方定義同步（DRY）
_FRAME_COUNT_BASES: dict = {
    "RANSAC_MAX_GAP_FRAMES": RANSAC_MAX_GAP_FRAMES,
    "RANSAC_WINDOW_SIZE": RANSAC_WINDOW_SIZE,
    "CONTACT_VECTOR_FRAME_OFFSET": CONTACT_VECTOR_FRAME_OFFSET,
    "CONTACT_BAT_PROXIMITY_FRAMES": CONTACT_BAT_PROXIMITY_FRAMES,
    "CONTACT_DIRECTION_SUSTAINED_FRAMES": CONTACT_DIRECTION_SUSTAINED_FRAMES,
    "CONTACT_WINDOW_MARGIN": CONTACT_WINDOW_MARGIN,
    "BALL_STATIC_MAX_FRAMES": BALL_STATIC_MAX_FRAMES,
    "POST_CONTACT_DESCENT_MIN_FRAMES": POST_CONTACT_DESCENT_MIN_FRAMES,
    "SWING_END_SUSTAINED_FRAMES": SWING_END_SUSTAINED_FRAMES,
    "TSTAND_MIN_STATIC_FRAMES": TSTAND_MIN_STATIC_FRAMES,
    "TSTAND_BAT_GRACE_FRAMES": TSTAND_BAT_GRACE_FRAMES,
    "BAT_STATIC_VELOCITY_LOOKBACK": BAT_STATIC_VELOCITY_LOOKBACK,
    "BAT_CHAIN_MAX_FRAME_GAP": BAT_CHAIN_MAX_FRAME_GAP,
    "BATCH_STATIC_MIN_FRAMES": BATCH_STATIC_MIN_FRAMES,
    "BATCH_TSTAND_FRAME_WINDOW": BATCH_TSTAND_FRAME_WINDOW,
    "CLOUD_GAP_THRESHOLD_FRAMES": CLOUD_GAP_THRESHOLD_FRAMES,
    "CLOUD_STATIC_SEGMENT_GAP_MIN_FRAMES": CLOUD_STATIC_SEGMENT_GAP_MIN_FRAMES,
    "CLOUD_TSTAND_TIME_TOL": CLOUD_TSTAND_TIME_TOL,
    "BAT_CLOUD_STATIC_MIN_FRAMES": BAT_CLOUD_STATIC_MIN_FRAMES,
    "BAT_CLOUD_GAP_THRESHOLD_FRAMES": BAT_CLOUD_GAP_THRESHOLD_FRAMES,
    "BAT_CLOUD_GAP_FILL_MAX_FRAMES": BAT_CLOUD_GAP_FILL_MAX_FRAMES,
    "BAT_CLOUD_TURN_STRIDE_FRAMES": BAT_CLOUD_TURN_STRIDE_FRAMES,
    "BAT_CLOUD_MERGE_MAX_GAP_FRAMES": BAT_CLOUD_MERGE_MAX_GAP_FRAMES,
}
_PIXEL_RATE_BASES: dict = {
    "RANSAC_GATE_THRESHOLD_PX": RANSAC_GATE_THRESHOLD_PX,
    "RANSAC_INLIER_THRESHOLD_PX": RANSAC_INLIER_THRESHOLD_PX,
    "BALL_MOTION_THRESHOLD_PX": BALL_MOTION_THRESHOLD_PX,
    "BAT_MAX_INTERFRAME_JUMP_PX": BAT_MAX_INTERFRAME_JUMP_PX,
    "POST_CONTACT_DESCENT_THRESHOLD_PX": POST_CONTACT_DESCENT_THRESHOLD_PX,
    "BALL_STATIC_THRESHOLD_PX": BALL_STATIC_THRESHOLD_PX,
    "SWING_VELOCITY_THRESHOLD_PX": SWING_VELOCITY_THRESHOLD_PX,
    "SWING_END_THRESHOLD_PX": SWING_END_THRESHOLD_PX,
    "BAT_STATIC_MAX_VELOCITY_PX": BAT_STATIC_MAX_VELOCITY_PX,
    "BAT_ARC_ONSET_MIN_SPEED_PX": BAT_ARC_ONSET_MIN_SPEED_PX,
    "BAT_CLOUD_TURN_MIN_STEP_PX": BAT_CLOUD_TURN_MIN_STEP_PX,
    "BAT_CLOUD_TURN_MIN_SPEED_PX": BAT_CLOUD_TURN_MIN_SPEED_PX,
}
# 反向幀計數縮放：fps 越高幀數越少（base × BASE_FPS / fps）
# 適用於「應在固定物理時間內完成判定」的幀計數閾值
_INVERSE_FRAME_COUNT_BASES: dict = {
    "RANSAC_DIRECTION_CHANGE_FRAMES": RANSAC_DIRECTION_CHANGE_FRAMES,
}


def apply_fps_scaling(fps: float) -> None:
    """依影片幀率重新校正所有 frame-based 常數（基準：30fps）。

    幀計數類（RANSAC_MAX_GAP_FRAMES 等）：
        scaled = max(1, round(base × fps / 30))
    每幀位移類（RANSAC_GATE_THRESHOLD_PX 等）：
        scaled = base × 30 / fps

    多次呼叫以相同基準值重新計算，結果冪等。

    Args:
        fps: 影片實際幀率（Hz），必須大於 0。

    Raises:
        ValueError: fps ≤ 0。
    """
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps!r}")

    gvars = globals()
    frame_scale = fps / BASE_FPS
    pixel_scale = BASE_FPS / fps

    for name, base in _FRAME_COUNT_BASES.items():
        gvars[name] = max(1, round(base * frame_scale))

    for name, base in _PIXEL_RATE_BASES.items():
        gvars[name] = base * pixel_scale

    # 感測器噪聲下限：防止高 fps 時縮放結果低於可信量測精度
    gvars["BALL_STATIC_THRESHOLD_PX"] = max(
        MIN_BALL_STATIC_THRESHOLD_PX,
        gvars["BALL_STATIC_THRESHOLD_PX"],
    )
    gvars["BAT_CLOUD_TURN_MIN_STEP_PX"] = max(
        MIN_BAT_CLOUD_TURN_STEP_PX,
        gvars["BAT_CLOUD_TURN_MIN_STEP_PX"],
    )
    gvars["BAT_CLOUD_TURN_MIN_SPEED_PX"] = max(
        MIN_BAT_CLOUD_TURN_SPEED_PX,
        gvars["BAT_CLOUD_TURN_MIN_SPEED_PX"],
    )

    for name, base in _INVERSE_FRAME_COUNT_BASES.items():
        gvars[name] = max(1, round(base * pixel_scale))
