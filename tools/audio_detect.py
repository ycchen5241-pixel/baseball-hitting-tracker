# audio_detect.py
# 從棒球影片音訊中偵測擊球時間點，輸出擊球時間戳並可選擇性裁切影片或抽取訓練幀

import argparse
import os
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, filtfilt, find_peaks


# ── 音訊偵測參數 ──────────────────────────────────────────────
ENERGY_WINDOW_MS = 5         # 短時能量計算窗口（ms）

# 主偵測頻段：棒球擊球衝擊聲集中在 3k~8kHz
BANDPASS_LOW_HZ = 3000
BANDPASS_HIGH_HZ = 8000

# 局部自適應閾值：在滑動窗口內計算局部 baseline，適應不同段落的背景噪音
# 解決問題：較遠擊球（onset 較低）在全局閾值下被漏偵測
ADAPTIVE_WINDOW_SEC = 15.0   # 局部閾值計算的滑動窗口寬度（秒）
ONSET_THRESHOLD_SIGMA = 2.5  # 局部 mean + N × std

ONSET_MIN_DISTANCE_SEC = 0.3 # 相鄰擊球點最小間隔（秒）
ONSET_MIN_PROMINENCE = 0.15  # onset 峰值突出度

# 瞬間衝擊驗證（排除說話聲、持續背景噪音）
TRANSIENT_RISE_WINDOWS = 3   # 上升應在 N×5ms 內完成（15ms）
TRANSIENT_DECAY_WINDOWS = 20 # 衰減至峰值 50% 應在 N×5ms 內（100ms）
TRANSIENT_DECAY_RATIO = 0.50 # 衰減目標比例

# 頻率比例驗證：棒球擊球在高頻（5k~8kHz）的比例遠高於球彈地、說話聲
# 解決問題：2:15 球反彈（低頻較多）、3:20 說話聲（缺乏高頻）
HF_LOW_HZ = 5000             # 高頻段下限
HF_HIGH_HZ = 8000            # 高頻段上限
MF_LOW_HZ = 2000             # 中頻段下限（對照用）
MF_HIGH_HZ = 5000            # 中頻段上限
HF_RATIO_MIN = 0.35          # 高頻 / (高頻+中頻) 能量比例下限

# ── 輸出參數 ──────────────────────────────────────────────────
CLIP_PRE_SEC = 1.0           # 擊球前裁切秒數
CLIP_POST_SEC = 3.0          # 擊球後裁切秒數（留足球飛行時間）
FRAME_POST_SEC = 1.0         # 抽取訓練幀的時間窗口（擊球後 N 秒）


def _extract_audio_wav(video_path: str, wav_path: str) -> None:
    """用 ffmpeg 從影片抽出單聲道 44100Hz WAV 音訊。"""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", video_path,
            "-vn", "-ar", "44100", "-ac", "1",
            "-acodec", "pcm_s16le", wav_path,
        ],
        check=True,
        capture_output=True,
    )


def _bandpass_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """對音訊套用 Butterworth 帶通濾波（1k~4kHz）。"""
    nyq = sr / 2.0
    b, a = butter(4, [BANDPASS_LOW_HZ / nyq, BANDPASS_HIGH_HZ / nyq], btype="band")
    return filtfilt(b, a, audio)


def _short_time_energy(audio: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """計算短時能量序列，回傳 (energy_array, hop_sec)。"""
    hop = int(sr * ENERGY_WINDOW_MS / 1000)
    energy = np.array([
        float(np.sum(audio[i: i + hop] ** 2))
        for i in range(0, len(audio) - hop, hop)
    ])
    return energy, ENERGY_WINDOW_MS / 1000.0


def _compute_band_energy(audio: np.ndarray, sr: int, low_hz: float, high_hz: float) -> np.ndarray:
    """計算指定頻段的短時能量序列。"""
    nyq = sr / 2.0
    b, a = butter(4, [low_hz / nyq, high_hz / nyq], btype="band")
    filtered = filtfilt(b, a, audio)
    energy, _ = _short_time_energy(filtered, sr)
    return energy


def _adaptive_threshold(onset: np.ndarray, hop_sec: float) -> np.ndarray:
    """計算局部自適應閾值（滑動窗口 mean + N×std）。

    每個位置的閾值由其前後 ADAPTIVE_WINDOW_SEC 秒的局部統計決定，
    解決影片不同段落背景噪音強度不同導致的漏偵測問題。
    """
    half = int(ADAPTIVE_WINDOW_SEC / hop_sec / 2)
    n = len(onset)
    thresholds = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half)
        local = onset[lo:hi]
        thresholds[i] = np.mean(local) + ONSET_THRESHOLD_SIGMA * np.std(local)
    return thresholds


def _is_transient(energy: np.ndarray, peak_idx: int) -> bool:
    """驗證峰值是否為瞬間衝擊（排除說話聲、持續背景噪音）。"""
    peak_val = energy[peak_idx]
    if peak_val < 1e-12:
        return False

    # 上升速率：peak 前 N 個窗口的能量應遠低於峰值
    pre_start = max(0, peak_idx - TRANSIENT_RISE_WINDOWS)
    pre_max = float(np.max(energy[pre_start:peak_idx])) if peak_idx > 0 else 0.0
    if pre_max > peak_val * 0.4:
        return False

    # 衰減速率：peak 後應快速衰減至峰值的 TRANSIENT_DECAY_RATIO 以下
    post_end = min(len(energy), peak_idx + TRANSIENT_DECAY_WINDOWS)
    post_min = (
        float(np.min(energy[peak_idx + 1: post_end]))
        if post_end > peak_idx + 1
        else peak_val
    )
    if post_min > peak_val * TRANSIENT_DECAY_RATIO:
        return False

    return True


def _has_bat_spectrum(
    hf_energy: np.ndarray,
    mf_energy: np.ndarray,
    peak_idx: int,
) -> bool:
    """驗證峰值附近的高頻 / (高頻+中頻) 比例是否符合棒球擊球特徵。

    棒球擊球：木棒+硬球碰撞產生大量 5k~8kHz 高頻成分。
    球彈地聲：低中頻為主，高頻比例低。
    說話聲：諧波集中在 4kHz 以下，高頻比例低。
    """
    w = 4  # 取峰值附近 ±4 個窗口的平均（20ms）
    lo = max(0, peak_idx - w)
    hi = min(len(hf_energy), peak_idx + w + 1)
    hf = float(np.mean(hf_energy[lo:hi]))
    mf = float(np.mean(mf_energy[lo:hi]))
    total = hf + mf
    if total < 1e-12:
        return False
    return (hf / total) >= HF_RATIO_MIN


def detect_hits(video_path: str, debug: bool = False) -> list[float]:
    """偵測影片中所有擊球時間點，回傳時間戳列表（秒）。

    使用 onset 偵測（能量躍升速率）而非原始能量，有效區分
    棒球擊球的瞬間脈衝與說話聲、背景噪音的持續能量。

    Args:
        video_path: .mov 或 .mp4 影片路徑。
        debug: 若為 True，額外儲存能量與 onset 的 CSV 供手動分析。

    Returns:
        擊球時間戳列表（秒），依時間順序排列。
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name

    try:
        print(f"[audio] 抽取音訊: {os.path.basename(video_path)}")
        _extract_audio_wav(video_path, wav_path)

        sr, raw = wavfile.read(wav_path)
        audio = raw.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        print(f"[audio] 音訊長度: {len(audio)/sr:.2f}s @ {sr}Hz")

        # ── Step 1: 主頻段帶通濾波 (3k~8kHz) → 短時能量 → onset ──
        filtered = _bandpass_filter(audio, sr)
        energy, hop_sec = _short_time_energy(filtered, sr)
        onset = np.maximum(0.0, np.diff(energy, prepend=energy[0]))

        o_max = onset.max()
        if o_max < 1e-12:
            print("[audio] onset 能量接近零，無法偵測")
            return []
        onset_norm = onset / o_max

        # ── Step 2: 局部自適應閾值（解決不同段落背景噪音差異）──
        local_thresh = _adaptive_threshold(onset, hop_sec)
        local_thresh_norm = local_thresh / o_max
        min_distance_hops = int(ONSET_MIN_DISTANCE_SEC / hop_sec)

        # find_peaks 需要單一 height 值，改用 boolean mask 預篩後再找峰值
        above_thresh = onset_norm >= local_thresh_norm
        onset_masked = np.where(above_thresh, onset_norm, 0.0)
        peaks, _ = find_peaks(
            onset_masked,
            distance=min_distance_hops,
            prominence=ONSET_MIN_PROMINENCE,
        )
        # 再過一次確保每個 peak 確實超過其局部閾值
        peaks = [p for p in peaks if onset_norm[p] >= local_thresh_norm[p]]

        # ── Step 3: 瞬間衝擊驗證（排除說話聲、持續背景噪音）──
        # 頻率比例驗證已移除：iPhone 麥克風頻率響應與室內聲學環境
        # 差異太大，無法用固定比例可靠區分，反而誤殺正確擊球點
        valid_peaks, reject_log = [], []
        for p in peaks:
            if not _is_transient(energy, p):
                reject_log.append((p * hop_sec, "持續性聲音"))
                continue
            valid_peaks.append(p)

        hit_times = sorted([float(p * hop_sec) for p in valid_peaks])
        print(f"[audio] 候選峰值: {len(peaks)}  通過驗證: {len(hit_times)}  "
              f"被過濾: {len(reject_log)}")
        if reject_log:
            print("[audio] 被過濾的候選：")
            for t, reason in sorted(reject_log):
                print(f"  {t:.2f}s  ({reason})")
        print("[audio] 最終擊球點：")
        for i, t in enumerate(hit_times):
            print(f"  [{i+1:02d}] {t:.3f}s  (onset {onset_norm[valid_peaks[i]]:.3f})")

        if debug:
            csv_path = os.path.splitext(video_path)[0] + "_audio_debug.csv"
            times_arr = np.arange(len(onset)) * hop_sec
            data = np.column_stack([times_arr, energy / (energy.max() + 1e-12), onset_norm])
            np.savetxt(csv_path, data, delimiter=",",
                       header="time_sec,energy_norm,onset_norm", comments="")
            print(f"[debug] CSV 已儲存: {csv_path}  (可用 Excel 開啟確認)")

        return hit_times

    finally:
        os.unlink(wav_path)


def clip_hits(
    video_path: str,
    hit_times: list[float],
    output_dir: str,
    pre_sec: float = CLIP_PRE_SEC,
    post_sec: float = CLIP_POST_SEC,
) -> list[str]:
    """依擊球時間點裁切影片片段，回傳輸出路徑列表。

    使用時間戳定位，正確處理 iPhone VFR（Variable Frame Rate）影片。

    Args:
        video_path: 原始影片路徑。
        hit_times: 擊球時間戳列表（秒）。
        output_dir: 輸出目錄。
        pre_sec: 擊球前保留秒數。
        post_sec: 擊球後保留秒數。

    Returns:
        裁切後的影片路徑列表。
    """
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    out_paths = []

    for i, t in enumerate(hit_times):
        start = max(0.0, t - pre_sec)
        duration = pre_sec + post_sec
        out_path = os.path.join(output_dir, f"{base}_hit{i+1:02d}.mp4")

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{start:.3f}",
                "-i", video_path,
                "-t", f"{duration:.3f}",
                "-c:v", "libx264", "-crf", "18",
                "-c:a", "aac",
                "-vsync", "vfr",          # 保留 VFR 時間戳
                out_path,
            ],
            check=True,
            capture_output=True,
        )
        print(f"[clip] 已儲存: {out_path}")
        out_paths.append(out_path)

    return out_paths


def extract_training_frames(
    video_path: str,
    hit_times: list[float],
    output_dir: str,
    post_sec: float = FRAME_POST_SEC,
) -> None:
    """從每個擊球點後擷取飛行球影格，供 YOLO 訓練標注使用。

    擷取擊球後 post_sec 秒內的所有幀，以 JPG 儲存。

    Args:
        video_path: 原始影片路徑。
        hit_times: 擊球時間戳列表（秒）。
        output_dir: 輸出目錄（每個擊球點建立子目錄）。
        post_sec: 擊球後擷取秒數。
    """
    base = os.path.splitext(os.path.basename(video_path))[0]

    for i, t in enumerate(hit_times):
        hit_dir = os.path.join(output_dir, f"{base}_hit{i+1:02d}_frames")
        os.makedirs(hit_dir, exist_ok=True)

        subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", f"{t:.3f}",
                "-i", video_path,
                "-t", f"{post_sec:.3f}",
                "-vsync", "vfr",
                "-q:v", "2",
                os.path.join(hit_dir, "frame_%05d.jpg"),
            ],
            check=True,
            capture_output=True,
        )
        n_frames = len([f for f in os.listdir(hit_dir) if f.endswith(".jpg")])
        print(f"[frames] hit{i+1:02d}: {n_frames} 幀 → {hit_dir}")


def _plot_energy(video_path: str, hit_times: list[float]) -> None:
    """繪製能量曲線與偵測到的擊球點（需要 matplotlib）。"""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib 未安裝，跳過繪圖")
        return

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        _extract_audio_wav(video_path, wav_path)
        sr, raw = wavfile.read(wav_path)
        audio = raw.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        filtered = _bandpass_filter(audio, sr)
        energy, hop_sec = _short_time_energy(filtered, sr)
        energy_norm = energy / energy.max()
        times = np.arange(len(energy)) * hop_sec

        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(times, energy_norm, color="steelblue", linewidth=0.8, label="Energy (norm)")
        for t in hit_times:
            ax.axvline(t, color="red", linewidth=1.5, linestyle="--")
            ax.text(t + 0.1, 0.95, f"{t:.2f}s", color="red", fontsize=8, va="top")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalised Energy")
        ax.set_title(f"Audio Hit Detection: {os.path.basename(video_path)}")
        ax.legend()
        plt.tight_layout()

        out_img = os.path.splitext(video_path)[0] + "_audio_energy.png"
        plt.savefig(out_img, dpi=120)
        plt.close()
        print(f"[plot] 能量圖已儲存: {out_img}")
    finally:
        os.unlink(wav_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="棒球擊球聲偵測")
    parser.add_argument("video", help="影片路徑（.mov 或 .mp4）")
    parser.add_argument(
        "--clip", action="store_true",
        help="裁切每個擊球片段為獨立影片",
    )
    parser.add_argument(
        "--frames", action="store_true",
        help="擷取每個擊球後的飛行球訓練幀",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="輸出音訊能量曲線圖（需要 matplotlib）",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="輸出 CSV 供手動分析（每 5ms 一行：時間、能量、onset）",
    )
    parser.add_argument(
        "--out", default=None,
        help="輸出目錄（預設為影片同目錄下的 audio_clips/）",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"[錯誤] 找不到影片: {args.video}")
        return

    output_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.video)), "audio_clips"
    )

    hit_times = detect_hits(args.video, debug=args.debug)

    if not hit_times:
        print("[audio] 未偵測到擊球點，結束")
        return

    if args.plot:
        _plot_energy(args.video, hit_times)

    if args.clip:
        clip_hits(args.video, hit_times, output_dir)

    if args.frames:
        extract_training_frames(args.video, hit_times, output_dir)

    if not args.clip and not args.frames and not args.plot:
        print("\n[提示] 加上參數可進行後續處理：")
        print("  --clip    裁切每個擊球片段")
        print("  --frames  擷取飛行球訓練幀")
        print("  --plot    輸出能量曲線圖")


if __name__ == "__main__":
    main()
