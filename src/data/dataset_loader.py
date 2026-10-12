"""
PyTorch Dataset Loader for Dementia & Cognitive Decline Screening.
Supports Interspeech ADReSS, DementiaBank Pitt Corpus, and generated clinical datasets with in-memory caching.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io.wavfile
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

from ..audio.vad_preprocessor import AudioVADPreprocessor
from ..audio.paralinguistics import ParalinguisticExtractor
from ..nlp.lexical_analyzer import LexicalAnalyzer
from ..nlp.llama_reasoner import ClinicalLlamaReasoner

_GLOBAL_FEATURE_CACHE = {}


def _to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.clone().detach().to(dtype)
    elif isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)
    else:
        return torch.tensor(x, dtype=dtype)


class DementiaDataset(Dataset):
    def __init__(
        self,
        samples_manifest: List[Dict],
        target_sr: int = 16000,
        max_audio_sec: float = 6.0,
        max_text_len: int = 128
    ):
        self.samples = samples_manifest
        self.target_sr = target_sr
        self.max_audio_samples = int(max_audio_sec * target_sr)
        self.max_text_len = max_text_len

        self.vad = AudioVADPreprocessor()
        self.para = ParalinguisticExtractor(sample_rate=target_sr)
        self.lex = LexicalAnalyzer()
        self.llm = ClinicalLlamaReasoner()

        self.cache = []
        self._precompute_features()

    def extract_biomarker_vector(self, audio: np.ndarray, vad_info: Dict, transcript: str) -> np.ndarray:
        para_feats = self.para.extract_features(audio, vad_info)
        lex_feats = self.lex.analyze_transcript(transcript)

        vector = [
            vad_info.get("pause_ratio", 0.0),
            vad_info.get("pause_rate_per_min", 0.0) / 40.0,
            vad_info.get("mean_pause_duration", 0.0) / 2.0,
            vad_info.get("speech_ratio", 0.0),
            para_feats.get("mean_f0_hz", 150.0) / 300.0,
            para_feats.get("std_f0_hz", 15.0) / 50.0,
            para_feats.get("jitter_percent", 1.0) / 5.0,
            para_feats.get("shimmer_percent", 3.0) / 10.0,
            para_feats.get("hnr_db", 15.0) / 30.0,
            para_feats.get("articulation_rate_syl_sec", 3.0) / 8.0,
            lex_feats.get("type_token_ratio", 0.5),
            lex_feats.get("brunet_w", 10.0) / 20.0,
            lex_feats.get("honore_r", 100.0) / 200.0,
            lex_feats.get("yules_k", 50.0) / 200.0,
            lex_feats.get("hesitation_rate", 0.05) / 0.3,
            lex_feats.get("pronoun_to_noun_ratio", 1.0) / 5.0,
        ]
        return np.array(vector, dtype=np.float32)

    def _precompute_features(self):
        with torch.no_grad():
            for item in self.samples:
                cache_key = (str(item.get("audio_path", "")), str(item.get("transcript", "")))
                
                if cache_key in _GLOBAL_FEATURE_CACHE:
                    cached = _GLOBAL_FEATURE_CACHE[cache_key]
                    audio = cached["audio"]
                    biomarkers = cached["biomarkers"]
                    text_emb = cached["text_emb"]
                    text_mask = cached["text_mask"]
                else:
                    audio_path = item.get("audio_path")
                    if audio_path and Path(audio_path).exists():
                        sr, raw_audio = scipy.io.wavfile.read(audio_path)
                        audio, _ = self.vad.load_and_normalize(raw_audio, sr)
                    else:
                        audio = np.zeros(self.max_audio_samples, dtype=np.float32)

                    if len(audio) > self.max_audio_samples:
                        audio = audio[:self.max_audio_samples]
                    elif len(audio) < self.max_audio_samples:
                        pad = np.zeros(self.max_audio_samples - len(audio), dtype=np.float32)
                        audio = np.concatenate([audio, pad])

                    vad_info = self.vad.detect_vad_segments(audio, self.target_sr)
                    transcript = item.get("transcript", "")
                    biomarkers = self.extract_biomarker_vector(audio, vad_info, transcript)

                    text_emb, text_mask = self.llm.extract_text_embeddings(transcript, self.max_text_len)
                    text_emb = text_emb.detach().cpu().numpy()
                    text_mask = text_mask.detach().cpu().numpy()

                    _GLOBAL_FEATURE_CACHE[cache_key] = {
                        "audio": audio,
                        "biomarkers": biomarkers,
                        "text_emb": text_emb,
                        "text_mask": text_mask,
                    }

                label = item.get("label", 0)
                mmse = item.get("mmse", 28.0)

                t_emb = _to_tensor(text_emb)
                t_mask = _to_tensor(text_mask)
                if t_emb.dim() == 3:
                    t_emb = t_emb.squeeze(0)
                if t_mask.dim() == 2:
                    t_mask = t_mask.squeeze(0)

                self.cache.append({
                    "audio": _to_tensor(audio),
                    "text_emb": t_emb,
                    "text_mask": t_mask,
                    "biomarkers": _to_tensor(biomarkers),
                    "label": torch.tensor([label], dtype=torch.float32),
                    "mmse": torch.tensor([mmse], dtype=torch.float32),
                    "id": item.get("id", "sample"),
                })

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.cache[idx]


def dementia_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    audios = torch.stack([b["audio"] for b in batch], dim=0)
    biomarkers = torch.stack([b["biomarkers"] for b in batch], dim=0)
    labels = torch.stack([b["label"] for b in batch], dim=0)
    mmses = torch.stack([b["mmse"] for b in batch], dim=0)

    max_len = max(b["text_emb"].shape[0] for b in batch)
    emb_dim = batch[0]["text_emb"].shape[1]

    padded_text_emb = torch.zeros(len(batch), max_len, emb_dim)
    padded_text_mask = torch.zeros(len(batch), max_len)

    for i, b in enumerate(batch):
        seq_len = b["text_emb"].shape[0]
        padded_text_emb[i, :seq_len] = b["text_emb"]
        padded_text_mask[i, :seq_len] = 1.0

    return {
        "audio": audios,
        "text_emb": padded_text_emb,
        "text_mask": padded_text_mask,
        "biomarkers": biomarkers,
        "label": labels,
        "mmse": mmses,
    }


def create_dataloaders(
    manifest: List[Dict],
    batch_size: int = 4,
    train_split: float = 0.7,
    val_split: float = 0.15
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    np.random.seed(42)
    indices = np.random.permutation(len(manifest))
    n = len(manifest)
    n_train = max(1, int(n * train_split))
    n_val = max(1, int(n * val_split))

    train_manifest = [manifest[i] for i in indices[:n_train]]
    val_manifest = [manifest[i] for i in indices[n_train:n_train + n_val]]
    test_manifest = [manifest[i] for i in indices[n_train + n_val:]]
    if not test_manifest:
        test_manifest = val_manifest

    train_ds = DementiaDataset(train_manifest)
    val_ds = DementiaDataset(val_manifest)
    test_ds = DementiaDataset(test_manifest)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=dementia_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=dementia_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=dementia_collate_fn)

    return train_loader, val_loader, test_loader
