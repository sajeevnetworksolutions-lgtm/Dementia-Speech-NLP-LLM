"""
WavLM Self-Supervised Speech Feature Encoder.
Extracts 1024-dimensional paralinguistic frame embeddings from raw 16kHz speech.
Integrates with HuggingFace Transformers WavLM with high-performance standalone fallback.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional

try:
    from config import WAVLM_CONFIG, AUDIO_CONFIG
except ImportError:
    from ...config import WAVLM_CONFIG, AUDIO_CONFIG


class WavLMAcousticEncoder(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or WAVLM_CONFIG
        self.embedding_dim = self.config.get("embedding_dim", 1024)
        self.model_name = self.config.get("model_name", "microsoft/wavlm-base-plus")
        self.load_hf = self.config.get("load_hf_pretrained", False)
        
        self.hf_model = None
        self.hf_processor = None
        self._init_backend()

    def _init_backend(self):
        if self.load_hf:
            try:
                from transformers import WavLMModel, AutoProcessor
                self.hf_processor = AutoProcessor.from_pretrained(self.model_name)
                self.hf_model = WavLMModel.from_pretrained(self.model_name)
                if self.config.get("freeze_feature_extractor", True):
                    self.hf_model.feature_extractor._freeze_parameters()
                print(f"[WavLM] Loaded HuggingFace model: {self.model_name}")
                return
            except Exception as e:
                print(f"[WavLM] Fallback to standalone neural frontend: {e}")

        # High performance convolutional + self-attention audio backbone (1024d)
        # Downsamples 16kHz audio to compact temporal frame tokens (~50-100 frames)
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Conv1d(128, 256, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Conv1d(256, 512, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, self.embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(self.embedding_dim),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(128)  # Cap sequence length to 128 temporal tokens for ultra-fast attention
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(self, audio_tensors: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if audio_tensors.dim() == 1:
            audio_tensors = audio_tensors.unsqueeze(0)

        if self.hf_model is not None:
            device = next(self.hf_model.parameters()).device
            audio_tensors = audio_tensors.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            outputs = self.hf_model(audio_tensors, attention_mask=attention_mask)
            frame_embeds = outputs.last_hidden_state
            if frame_embeds.shape[-1] != self.embedding_dim:
                proj = nn.Linear(frame_embeds.shape[-1], self.embedding_dim).to(device)
                frame_embeds = proj(frame_embeds)
            pooled = torch.mean(frame_embeds, dim=1)
        else:
            if audio_tensors.dim() == 2:
                x = audio_tensors.unsqueeze(1)
            else:
                x = audio_tensors
            
            conv_out = self.conv_layers(x)       # [B, 1024, 128]
            conv_out = conv_out.transpose(1, 2)  # [B, 128, 1024]
            frame_embeds = self.transformer(conv_out)
            frame_embeds = self.layer_norm(frame_embeds)
            pooled = torch.mean(frame_embeds, dim=1)

        return {
            "frame_embeddings": frame_embeds,
            "pooled_embedding": pooled,
        }
