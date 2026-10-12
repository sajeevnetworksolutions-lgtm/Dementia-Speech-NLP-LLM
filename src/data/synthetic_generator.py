"""
Synthetic Clinical Audio & Transcript Generator.
Generates realistic multi-modal test samples based on the Boston Cookie Theft picture description task
and DementiaBank Pitt / Interspeech ADReSS corpora.
"""
import numpy as np
import scipy.io.wavfile
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

try:
    from config import SAMPLES_DIR, AUDIO_CONFIG
except ImportError:
    from ...config import SAMPLES_DIR, AUDIO_CONFIG


SAMPLE_PROFILES = [
    {
        "id": "HC_Patient_101",
        "category": "Control",
        "label": 0,
        "mmse": 29.0,
        "age": 68,
        "gender": "Female",
        "transcript": "Well, in this picture, the boy is standing on a stool trying to reach into the cookie jar in the upper cupboard. The stool is tilting and he looks like he might fall. His sister is reaching up with her hand to ask for a cookie. Meanwhile, the mother is drying dishes at the kitchen sink, but the water is overflowing onto the floor because she is distracted looking out the open window.",
        "pitch_mean": 210.0,
        "pitch_std": 28.5,
        "pause_rate": 8.0,
        "pause_mean": 0.32,
        "jitter": 0.85,
    },
    {
        "id": "HC_Patient_102",
        "category": "Control",
        "label": 0,
        "mmse": 28.0,
        "age": 72,
        "gender": "Male",
        "transcript": "I see a kitchen scene. The mother is washing a plate and the water is running over the edge of the sink onto the floor. Behind her, a little boy has climbed onto a wobbly stool to get cookies from the jar and the stool is about to tip over. A young girl is standing beside him waiting for him to hand her one.",
        "pitch_mean": 135.0,
        "pitch_std": 22.0,
        "pause_rate": 10.0,
        "pause_mean": 0.38,
        "jitter": 1.10,
    },
    {
        "id": "MCI_Patient_201",
        "category": "MCI",
        "label": 1,
        "mmse": 23.0,
        "age": 74,
        "gender": "Female",
        "transcript": "The... um... lady is by the water, washing the... plates. And the water is spilling on the floor. And the kids are... well, the boy is on the little chair, trying to take the... cookies from the jar. And the girl wants one too, but the chair is slipping.",
        "pitch_mean": 195.0,
        "pitch_std": 14.5,
        "pause_rate": 18.0,
        "pause_mean": 0.72,
        "jitter": 2.15,
    },
    {
        "id": "AD_Patient_301",
        "category": "Dementia",
        "label": 1,
        "mmse": 17.0,
        "age": 77,
        "gender": "Male",
        "transcript": "There is a... uh... woman. She is doing something with the... the dishes. The... water is coming out... oh my, on the floor. And over there, the... uh... the little one... the boy is getting into the... that thing up there... the jar with the sweet things. And he's... uh... falling down maybe. The stool... the stool is broke... tipping.",
        "pitch_mean": 120.0,
        "pitch_std": 11.2,
        "pause_rate": 26.0,
        "pause_mean": 1.15,
        "jitter": 3.40,
    },
    {
        "id": "AD_Patient_302",
        "category": "Dementia",
        "label": 1,
        "mmse": 12.0,
        "age": 81,
        "gender": "Female",
        "transcript": "Well... um... somebody is there. She is... uh... look at that water... it's all going down there on the... thing. And the boy... he is on that... that wooden thing. He wants the... you know... the good things inside the top. And she is... uh... um... laughing or looking... I don't know what she is doing.",
        "pitch_mean": 180.0,
        "pitch_std": 8.5,
        "pause_rate": 32.0,
        "pause_mean": 1.48,
        "jitter": 4.10,
    }
]


class ClinicalSyntheticGenerator:
    def __init__(self, sample_rate: int = 16000, output_dir: Optional[Union[str, Path]] = None):
        self.sr = sample_rate
        self.output_dir = Path(output_dir) if output_dir else SAMPLES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def synthesize_speech_audio(self, profile: Dict) -> np.ndarray:
        words = profile["transcript"].split()
        f0_mean = profile["pitch_mean"]
        f0_std = profile["pitch_std"]
        pause_mean = profile["pause_mean"]
        jitter_pct = profile["jitter"]

        audio_segments = []

        for i, word in enumerate(words):
            word_dur = max(0.2, min(0.65, len(word) * 0.075 + np.random.uniform(-0.05, 0.05)))
            t = np.linspace(0, word_dur, int(word_dur * self.sr), endpoint=False)

            jitter_noise = np.random.normal(0, (jitter_pct / 100.0) * f0_mean, size=len(t))
            f0 = f0_mean + np.sin(2 * np.pi * 1.5 * t) * f0_std + jitter_noise
            f0 = np.clip(f0, 60.0, 450.0)

            phase = 2 * np.pi * np.cumsum(f0) / self.sr

            sig = (
                0.60 * np.sin(phase) +
                0.30 * np.sin(2 * phase) +
                0.15 * np.sin(3 * phase) +
                0.08 * np.sin(4 * phase)
            )

            env = np.hanning(len(t)) ** 0.6
            word_sig = sig * env
            audio_segments.append(word_sig)

            is_hesitation = ("..." in word) or (word.lower() in ["um", "uh", "er", "well"])
            if is_hesitation:
                p_dur = pause_mean * np.random.uniform(1.2, 2.0)
            else:
                p_dur = max(0.04, np.random.normal(pause_mean * 0.4, 0.08))

            pause_len = int(p_dur * self.sr)
            if pause_len > 0:
                noise = np.random.normal(0, 0.003, pause_len)
                audio_segments.append(noise)

        full_audio = np.concatenate(audio_segments)
        full_audio = full_audio / (np.max(np.abs(full_audio)) + 1e-6) * 0.85
        return full_audio.astype(np.float32)

    def generate_all_samples(self) -> List[Dict]:
        manifest = []
        for p in SAMPLE_PROFILES:
            audio = self.synthesize_speech_audio(p)
            wav_filename = f"{p['id']}_{p['category']}.wav"
            wav_path = self.output_dir / wav_filename
            
            scipy.io.wavfile.write(str(wav_path), self.sr, (audio * 32767).astype(np.int16))

            entry = {
                "id": p["id"],
                "category": p["category"],
                "label": p["label"],
                "mmse": p["mmse"],
                "age": p["age"],
                "gender": p["gender"],
                "transcript": p["transcript"],
                "audio_path": str(wav_path),
                "duration_sec": len(audio) / self.sr
            }
            manifest.append(entry)

        manifest_path = self.output_dir / "samples_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        return manifest
