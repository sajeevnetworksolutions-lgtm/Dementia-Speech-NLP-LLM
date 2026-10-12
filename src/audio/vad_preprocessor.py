"""
Audio Voice Activity Detection (VAD) & Preprocessing Module.
Handles:
  - 16kHz audio normalization & mono conversion
  - Energy/spectral Voice Activity Detection
  - Micro-hesitation and silence pause extraction (durations, frequency, pause-to-speech ratio)
"""
import numpy as np
import scipy.signal
import scipy.io.wavfile
import math
from typing import Dict, Tuple, List, Optional

try:
    from config import AUDIO_CONFIG
except ImportError:
    from ...config import AUDIO_CONFIG


class AudioVADPreprocessor:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or AUDIO_CONFIG
        self.target_sr = self.config.get("sample_rate", 16000)
        self.pause_min_sec = self.config.get("pause_min_duration_sec", 0.25)
        self.vad_threshold = self.config.get("vad_threshold", 0.45)

    def load_and_normalize(self, audio_data: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """
        Convert multi-channel to mono, resample to target_sr, and RMS normalize.
        """
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Convert int16 to float32 [-1.0, 1.0] if necessary
        if np.issubdtype(audio_data.dtype, np.integer):
            max_val = np.iinfo(audio_data.dtype).max
            audio_data = audio_data.astype(np.float32) / max_val
        else:
            audio_data = audio_data.astype(np.float32)

        # Resample if needed
        if sample_rate != self.target_sr and sample_rate > 0:
            num_samples = int(len(audio_data) * float(self.target_sr) / sample_rate)
            audio_data = scipy.signal.resample(audio_data, num_samples)
            sample_rate = self.target_sr

        # RMS Peak Normalization
        rms = np.sqrt(np.mean(audio_data ** 2) + 1e-8)
        if rms > 0:
            target_rms = 0.1
            audio_data = audio_data * (target_rms / (rms + 1e-6))
            # Clip to [-1.0, 1.0]
            audio_data = np.clip(audio_data, -1.0, 1.0)

        return audio_data, self.target_sr

    def compute_frame_energy(self, signal: np.ndarray, frame_length: int = 400, hop_length: int = 160) -> np.ndarray:
        """
        Compute short-time energy for each frame (25ms window, 10ms hop at 16kHz).
        """
        num_frames = 1 + (len(signal) - frame_length) // hop_length
        if num_frames <= 0:
            return np.array([np.sum(signal ** 2)])

        frames = np.lib.stride_tricks.as_strided(
            signal,
            shape=(num_frames, frame_length),
            strides=(signal.strides[0] * hop_length, signal.strides[0])
        )
        # Apply Hanning window
        window = np.hanning(frame_length)
        windowed = frames * window
        energy = np.sum(windowed ** 2, axis=1) / frame_length
        # Normalize energy [0, 1]
        max_e = np.max(energy) if np.max(energy) > 0 else 1.0
        return energy / max_e

    def detect_vad_segments(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """
        Performs frame-by-frame VAD and extracts speech and silence/hesitation segments.
        """
        frame_len = int(0.025 * sample_rate)  # 25ms
        hop_len = int(0.010 * sample_rate)    # 10ms
        energy = self.compute_frame_energy(audio, frame_len, hop_len)

        # Dynamic thresholding based on audio background noise
        noise_floor = np.percentile(energy, 20)
        dynamic_threshold = max(self.vad_threshold * 0.1, noise_floor * 2.5)

        is_speech = energy > dynamic_threshold

        # Smooth VAD decisions with morphological closing
        smoothed = np.copy(is_speech).astype(int)
        kernel_size = 5
        if len(smoothed) >= kernel_size:
            smoothed = np.convolve(smoothed, np.ones(kernel_size)/kernel_size, mode='same') > 0.4

        # Extract speech and pause intervals
        speech_intervals = []
        pause_intervals = []

        in_speech = False
        start_idx = 0

        for i, val in enumerate(smoothed):
            time_sec = (i * hop_len) / sample_rate
            if val and not in_speech:
                in_speech = True
                if i > 0:
                    pause_duration = time_sec - (start_idx * hop_len) / sample_rate
                    if pause_duration >= self.pause_min_sec:
                        pause_intervals.append({
                            "start": (start_idx * hop_len) / sample_rate,
                            "end": time_sec,
                            "duration": pause_duration
                        })
                start_idx = i
            elif not val and in_speech:
                in_speech = False
                speech_duration = time_sec - (start_idx * hop_len) / sample_rate
                speech_intervals.append({
                    "start": (start_idx * hop_len) / sample_rate,
                    "end": time_sec,
                    "duration": speech_duration
                })
                start_idx = i

        total_duration = len(audio) / sample_rate
        total_speech_time = sum(s["duration"] for s in speech_intervals)
        total_pause_time = sum(p["duration"] for p in pause_intervals)

        pause_durations = [p["duration"] for p in pause_intervals]
        mean_pause = float(np.mean(pause_durations)) if pause_durations else 0.0
        std_pause = float(np.std(pause_durations)) if pause_durations else 0.0
        pause_rate = (len(pause_intervals) / (total_duration / 60.0)) if total_duration > 0 else 0.0

        return {
            "total_duration": total_duration,
            "total_speech_time": total_speech_time,
            "total_pause_time": total_pause_time,
            "speech_ratio": total_speech_time / max(total_duration, 1e-5),
            "pause_ratio": total_pause_time / max(total_duration, 1e-5),
            "pause_count": len(pause_intervals),
            "pause_rate_per_min": pause_rate,
            "mean_pause_duration": mean_pause,
            "pause_duration_variance": std_pause ** 2,
            "speech_intervals": speech_intervals,
            "pause_intervals": pause_intervals,
            "vad_mask": smoothed,
        }
