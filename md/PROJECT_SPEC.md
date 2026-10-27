# Flow — Your personal baseball trainer
### Baseball Hitting Tracker 專案規格文件

## 專案概述
手機端棒球擊球分析 App，計算 Exit Velocity、Launch Angle 並視覺化軌跡。目標平台：iOS + Android（React Native）。YOLOv8s 採 AGPL-3.0，以整個專案開源合規商用。

## MVP 功能
240fps 慢動作錄影（打擊側邊角，畫面含本壘板）、Exit Velocity（km/h）、Launch Angle（deg）、球軌跡疊加（紅→紫漸層）、觸控友善的影片回放。

## 技術架構
- **偵測**：YOLOv8s（5類：ball / bat / bat_head / bat_tail / home_plate），class-aware NMS，IoU 0.7~0.9
- **追蹤**：CloudBallTracker / BatCloudTracker（離線點雲式批次分析：全片段偵測收集 → 密度聚類移除靜止背景 → Sequential RANSAC 穩健軌跡擬合），取代逐幀線上追蹤
- **計算**：OpenCV（Canny 角點提取本壘板）；Homography 主要供攝影機仰角估算與 LA 視差補正，EV 改用側拍單軸法直接以 HP bbox 寬度定標
- **部署**：Python 原型 → ONNX → TFLite / CoreML 手機推論
- **後端**：Supabase（Auth + 分析結果），影片僅存本機

## 物理計算
擊球幀判定：CloudBallTracker 內建三訊號判定（T座消失幀 / 軌跡起點 / 棒頭最近幀，中位數綜合），無法判定時退回距離最小值 / 方向改變後備法。Exit Velocity：側拍單軸法，取接觸後乾淨軌跡的 image-x 速度，以本壘板 bbox 寬度定標後除以 cos(仰角) 還原總速度。Launch Angle：像素 Y/X 位移比例法 + 視差補正。

## 編碼規範（最高優先）
KISS · YAGNI · DRY · SOLID；Google Python/JSON/Markdown 風格；
**嚴禁**：魔術數字、硬編碼、靜默忽略例外、`from x import *`、被註解的死碼、旗標參數。所有配置集中於 `config/constants`。
