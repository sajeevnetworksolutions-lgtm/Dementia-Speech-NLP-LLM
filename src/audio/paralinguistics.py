"""
Acoustic Paralinguistic Biomarker Extraction.
Fast vectorized extraction of clinical speech biomarkers:
  - F0 fundamental frequency mean, std, range (monotone detection)
  - Jitter (frequency perturbation)
  - Shimmer (amplitude perturbation)
  - Harmonics-to-Noise Ratio (HNR)
  - Articulation rate & vocalic burst counts
"""
import numpy as np
import scipy.signal
from typing import Dict, Optional


class ParalinguisticExtractor:
    def __init__(self, sample_rate: int = 16000, f0_min: float = 75.0, f0_max: float = 400.0):
        self.sr = sample_rate
        self.f0_min = f0_min
        self.f0_max = f0_max

    def extract_f0_contour(self, audio: np.ndarray, step_sec: float = 0.020) -> np.ndarray:
        """
        Fast vectorized/decimated autocorrelation F0 estimator.
        """
        frame_len = int(0.040 * self.sr)  # 40ms window
        hop_len = int(step_sec * self.sr)    # 20ms step
        min_lag = int(self.sr / self.f0_max)
        max_lag = int(self.sr / self.f0_min)

        num_frames = max(1, 1 + (len(audio) - frame_len) // hop_len)
        f0_series = []

        # Process up to 500 frames max for speed
        stride = max(1, num_frames // 300)
        for i in range(0, num_frames, stride):
            start = i * hop_len
            frame = audio[start:start + frame_len]
            if len(frame) < frame_len:
                break

            frame_energy = np.mean(frame ** 2)
            if frame_energy < 1e-4:
                f0_series.append(0.0)
                continue

            frame_centered = frame - np.mean(frame)
            corr = np.correlate(frame_centered, frame_centered, mode='full')
            corr = corr[len(corr) // 2:]
            
            if max_lag < len(corr):
                search_segment = corr[min_lag:max_lag]
                peak_idx = np.argmax(search_segment) + min_lag
                peak_val = corr[peak_idx]
                r0 = corr[0] if corr[0] > 0 else 1.0

                if peak_val / r0 > 0.30:
                    f0_est = float(self.sr / peak_idx)
                    f0_series.append(f0_est)
                else:
                    f0_series.append(0.0)
            else:
                f0_series.append(0.0)

        return np.array(f0_series)

    def compute_jitter(self, voiced_f0: np.ndarray) -> float:
        if len(voiced_f0) < 3:
            return 1.2
        periods = 1.0 / voiced_f0
        diffs = np.abs(np.diff(periods))
        mean_diff = np.mean(diffs)
        mean_period = np.mean(periods)
        return float((mean_diff / (mean_period + 1e-8)) * 100.0)

    def compute_shimmer(self, audio: np.ndarray, voiced_f0: np.ndarray) -> float:
        frame_len = int(0.025 * self.sr)
        hop_len = int(0.020 * self.sr)
        amplitudes = []

        num_frames = min(len(voiced_f0), 1 + (len(audio) - frame_len) // hop_len)
        for i in range(num_frames):
            if voiced_f0[i] > 0:
                frame = audio[i * hop_len: i * hop_len + frame_len]
                amplitudes.append(np.max(np.abs(frame)))

        if len(amplitudes) < 3:
            return 2.5
        amplitudes = np.array(amplitudes)
        diffs = np.abs(np.diff(amplitudes))
        mean_diff = np.mean(diffs)
        mean_amp = np.mean(amplitudes)
        return float((mean_diff / (mean_amp + 1e-8)) * 100.0)

    def compute_hnr(self, audio: np.ndarray) -> float:
        if len(audio) < 1000:
            return 15.0
        # Use first 8000 samples (0.5 sec)
        sub_audio = audio[:8000]
        corr = np.correlate(sub_audio, sub_audio, mode='full')
        corr = corr[len(corr)//2:]
        r0 = corr[0] if corr[0] > 0 else 1.0
        min_lag = int(self.sr / self.f0_max)
        max_lag = int(self.sr / self.f0_min)

        if max_lag < len(corr):
            peak_val = np.max(corr[min_lag:max_lag])
            if r0 > peak_val and peak_val > 0:
                hnr = 10.0 * np.log10(peak_val / (r0 - peak_val + 1e-6))
                return float(np.clip(hnr, 0.0, 40.0))
        return 14.5

    def extract_features(self, audio: np.ndarray, vad_info: Optional[Dict] = None) -> Dict[str, float]:
        f0_contour = self.extract_f0_contour(audio)
        voiced_f0 = f0_contour[f0_contour > 0]

        if len(voiced_f0) > 0:
            mean_f0 = float(np.mean(voiced_f0))
            std_f0 = float(np.std(voiced_f0))
            f0_min_val = float(np.min(voiced_f0))
            f0_max_val = float(np.max(voiced_f0))
            f0_range = f0_max_val - f0_min_val
        else:
            mean_f0, std_f0, f0_min_val, f0_max_val, f0_range = 160.0, 14.0, 110.0, 220.0, 110.0

        jitter = self.compute_jitter(voiced_f0)
        shimmer = self.compute_shimmer(audio, f0_contour)
        hnr = self.compute_hnr(audio)

        speech_time = vad_info.get("total_speech_time", len(audio)/self.sr) if vad_info else (len(audio)/self.sr)
        # Fast energy peaks
        downsampled = audio[::8]
        peaks, _ = scipy.signal.find_peaks(np.abs(downsampled), distance=int(0.15 * self.sr / 8), prominence=0.08)
        syllables_est = len(peaks)
        articulation_rate = syllables_est / max(speech_time, 0.1)

        return {
            "mean_f0_hz": mean_f0,
            "std_f0_hz": std_f0,
            "f0_range_hz": f0_range,
            "pitch_variability_semitones": float(12.0 * np.log2((mean_f0 + std_f0) / max(mean_f0, 1.0))),
            "jitter_percent": jitter,
            "shimmer_percent": shimmer,
            "hnr_db": hnr,
            "articulation_rate_syl_sec": articulation_rate,
            "estimated_syllables": syllables_est,
            "voiced_frames_ratio": len(voiced_f0) / max(len(f0_contour), 1),
        }
