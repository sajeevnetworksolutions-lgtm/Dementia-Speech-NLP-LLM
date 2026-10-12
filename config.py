"""
Project Configuration for Early Dementia & Cognitive Decline Screening System.
M.Tech Dissertation: Acoustic NLP + Fine-Tuned LLM (WavLM + LLaMA-3 QLoRA)
"""
import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "saved_models"
LOGS_DIR = PROJECT_ROOT / "logs"
SAMPLES_DIR = PROJECT_ROOT / "sample_data"

for d in [DATA_DIR, MODELS_DIR, LOGS_DIR, SAMPLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Audio Processing Configuration
AUDIO_CONFIG = {
    "sample_rate": 16000,
    "max_duration_sec": 120.0,      # 2-minute screening audio
    "min_duration_sec": 3.0,
    "vad_threshold": 0.45,           # Voice activity threshold
    "pause_min_duration_sec": 0.25,  # Micro-hesitation threshold (250ms)
    "n_fft": 512,
    "hop_length": 160,               # 10ms frame rate at 16kHz
    "n_mels": 80,
    "f_min": 50,
    "f_max": 7600,
}

# WavLM Acoustic Encoder Configuration
WAVLM_CONFIG = {
    "model_name": "microsoft/wavlm-base-plus",
    "large_model_name": "microsoft/wavlm-large",
    "embedding_dim": 1024,
    "sampling_rate": 16000,
    "freeze_feature_extractor": True,
    "use_paralinguistics": True,
}

# LLaMA-3 / Clinical NLP Configuration
LLM_CONFIG = {
    "model_name": "meta-llama/Meta-Llama-3-8B-Instruct",
    "fallback_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "hidden_dim": 4096,
    "projected_dim": 768,
    "max_seq_length": 512,
    # QLoRA Hyperparameters
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "quantization_bits": 4,
}

# Temporal Cross-Modal Attention Configuration
CROSS_ATTN_CONFIG = {
    "audio_dim": 1024,
    "text_dim": 768,
    "fused_dim": 512,
    "num_heads": 8,
    "dropout": 0.1,
    "feedforward_dim": 1024,
}

# Multi-Task Learning Objective
MULTI_TASK_CONFIG = {
    "cls_weight": 1.0,               # Loss weight for binary dementia classification
    "reg_weight": 0.5,               # Loss weight for continuous MMSE regression
    "mmse_min": 0.0,
    "mmse_max": 30.0,
}

# Clinical MMSE Severity Stratification
MMSE_THRESHOLDS = {
    "Normal": (26.0, 30.0),
    "Mild Cognitive Impairment (MCI)": (21.0, 25.9),
    "Mild Dementia": (15.0, 20.9),
    "Moderate to Severe Dementia": (0.0, 14.9),
}

# Training Defaults
TRAIN_CONFIG = {
    "batch_size": 8,
    "learning_rate": 2e-4,
    "weight_decay": 1e-4,
    "epochs": 20,
    "warmup_ratio": 0.1,
    "device": "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
}
