# Baseball Hitting Tracker

側面攝影機棒球擊球分析：從偵測到的球／棒軌跡計算擊球初速與仰角。本文件只收錄這個 context 特有的領域詞彙，不收錄一般程式設計概念，也不記錄實作細節。

## Language

**Ball Physics（球體物理計算）**：
從一次揮擊的追蹤結果（`CloudTrackResult`）算出來的一組數值——Contact Frame、Velocity Window、Launch Angle、Exit Velocity——視為一個整體概念，因為這四者共用同一段推導邏輯、理應由同一個模組算出。
_Avoid_: physics result（太籠統，不特指這四者的組合）

**Contact Frame（擊球幀）**：
球棒擊中球的那一個影片 frame_index。可能由 `CloudBallTracker` 內建判定給出，也可能在該判定失敗時由備援路徑（距離最小值／方向改變）推算得到；兩種來源對外都視為同一個 Contact Frame，呼叫端不需區分來源。
_Avoid_: contact point（可能跟球棒-球的空間座標混淆，Contact Frame 專指時間軸上的幀號）

**Velocity Window（速度視窗）**：
Contact Frame 之後、frame_index ≥ Contact Frame 的乾淨球位置子集，是 Launch Angle 與 Exit Velocity 計算共用的輸入資料。是追蹤結果的一個切片，不是獨立重新篩選出來的資料。
_Avoid_: velocity_post（程式碼內部變數名，不是要對外溝通的概念名）

**CloudTrackResult**：
`CloudBallTracker` 點雲式追蹤管線的輸出：含清洗過的球位置序列、擊球情境（T-stand／投球／側拋）、內建 Contact Frame 判定。是 Ball Physics 计算的唯一輸入來源。
_Avoid_: ball_result（變數名，非概念名，但常在程式碼中代稱此型別）

**Launch Angle（仰角）**：
球離棒瞬間的飛行仰角，以像素座標的 Y/X 位移比例（割線法）推算，向上為正。
_Avoid_: LA（僅程式碼/輸出面板用縮寫，文件討論時用全名）

**Exit Velocity（擊球初速）**：
球離棒瞬間的初速，以側拍影片中球的 image-x／image-y 像素速度合成、透過本壘板已知尺寸定標換算而得。
_Avoid_: EV（僅程式碼/輸出面板用縮寫，文件討論時用全名）

**T-stand（踢座）**：
擊球情境之一：球靜置於打擊座上，接觸前無位移，接觸後從靜止突然飛出。與「投球」「側拋」並列為 `CloudBallTracker` 判斷的三種情境之一。
_Avoid_: tee, tstand（僅程式碼內部命名使用）
