"""
Temporal Cross-Modal Attention Module.
Temporally aligns acoustic hesitation frames (WavLM 1024d) with linguistic tokens (LLaMA 768d).
Produces a 512-dimensional synchronized cognitive representation vector for multi-task diagnosis.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

try:
    from config import CROSS_ATTN_CONFIG
except ImportError:
    from ...config import CROSS_ATTN_CONFIG


class TemporalCrossModalAttention(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or CROSS_ATTN_CONFIG
        self.audio_dim = self.config.get("audio_dim", 1024)
        self.text_dim = self.config.get("text_dim", 768)
        self.fused_dim = self.config.get("fused_dim", 512)
        self.num_heads = self.config.get("num_heads", 8)
        self.dropout_rate = self.config.get("dropout", 0.1)

        self.audio_proj = nn.Linear(self.audio_dim, self.fused_dim)
        self.text_proj = nn.Linear(self.text_dim, self.fused_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.fused_dim,
            num_heads=self.num_heads,
            dropout=self.dropout_rate,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(self.fused_dim * 2, self.fused_dim),
            nn.Sigmoid()
        )

        self.ffn = nn.Sequential(
            nn.Linear(self.fused_dim, self.config.get("feedforward_dim", 1024)),
            nn.GELU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(self.config.get("feedforward_dim", 1024), self.fused_dim),
        )

        self.norm1 = nn.LayerNorm(self.fused_dim)
        self.norm2 = nn.LayerNorm(self.fused_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        proj_audio = self.audio_proj(audio_features)
        proj_text = self.text_proj(text_features)

        attn_out, attn_weights = self.cross_attn(
            query=proj_text,
            key=proj_audio,
            value=proj_audio,
            key_padding_mask=audio_mask,
            need_weights=True
        )

        gate_val = self.gate(torch.cat([proj_text, attn_out], dim=-1))
        fused_seq = self.norm1(proj_text + self.dropout(gate_val * attn_out + (1.0 - gate_val) * proj_text))
        fused_seq = self.norm2(fused_seq + self.dropout(self.ffn(fused_seq)))

        if text_mask is not None:
            mask_expanded = text_mask.unsqueeze(-1)
            pooled = (fused_seq * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            pooled = torch.mean(fused_seq, dim=1)

        return pooled, attn_weights
