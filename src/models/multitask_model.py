"""
Unified Multi-Task Dementia & Cognitive Decline Screening Network.
End-to-end PyTorch architecture integrating:
  1. WavLM Acoustic Speech Encoder (1024-dim)
  2. Clinical Linguistic Embedding Projector (768-dim)
  3. Temporal Cross-Modal Attention Layer (512-dim)
  4. Handcrafted Clinical Biomarker Fusion Highway
  5. Dual Multi-Task Heads: Binary Dementia Classifier & Continuous MMSE Regressor
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional

from ..audio.acoustic_encoder import WavLMAcousticEncoder
from ..nlp.llama_reasoner import ClinicalLlamaReasoner
from .cross_attention import TemporalCrossModalAttention

try:
    from config import WAVLM_CONFIG, LLM_CONFIG, CROSS_ATTN_CONFIG, MULTI_TASK_CONFIG
except ImportError:
    from ...config import WAVLM_CONFIG, LLM_CONFIG, CROSS_ATTN_CONFIG, MULTI_TASK_CONFIG


class DementiaScreeningNet(nn.Module):
    def __init__(
        self,
        audio_config: Optional[Dict] = None,
        llm_config: Optional[Dict] = None,
        cross_attn_config: Optional[Dict] = None,
        num_biomarkers: int = 16
    ):
        super().__init__()
        self.audio_encoder = WavLMAcousticEncoder(audio_config or WAVLM_CONFIG)
        self.llm_reasoner = ClinicalLlamaReasoner(llm_config or LLM_CONFIG)
        self.cross_attention = TemporalCrossModalAttention(cross_attn_config or CROSS_ATTN_CONFIG)

        fused_dim = (cross_attn_config or CROSS_ATTN_CONFIG).get("fused_dim", 512)

        self.biomarker_highway = nn.Sequential(
            nn.Linear(num_biomarkers, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        total_rep_dim = fused_dim + 64

        self.classifier_head = nn.Sequential(
            nn.Linear(total_rep_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )

        self.regressor_head = nn.Sequential(
            nn.Linear(total_rep_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        audio_tensors: torch.Tensor,
        text_features: torch.Tensor,
        biomarkers: Optional[torch.Tensor] = None,
        audio_mask: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        if audio_tensors.dim() <= 2 and audio_tensors.shape[-1] > 1024:
            audio_out = self.audio_encoder(audio_tensors)
            audio_embeds = audio_out["frame_embeddings"]
        else:
            audio_embeds = audio_tensors

        fused_vector, attn_weights = self.cross_attention(
            audio_features=audio_embeds,
            text_features=text_features,
            audio_mask=audio_mask,
            text_mask=text_mask
        )

        batch_size = fused_vector.shape[0]
        if biomarkers is None:
            biomarkers = torch.zeros((batch_size, 16), device=fused_vector.device)

        bio_rep = self.biomarker_highway(biomarkers)
        combined_rep = torch.cat([fused_vector, bio_rep], dim=-1)

        cls_logits = self.classifier_head(combined_rep)
        dementia_prob = torch.sigmoid(cls_logits)

        mmse_normalized = self.regressor_head(combined_rep)
        mmse_score = mmse_normalized * 30.0

        return {
            "cls_logits": cls_logits,
            "dementia_prob": dementia_prob,
            "mmse_score": mmse_score,
            "attention_weights": attn_weights,
            "latent_vector": combined_rep,
        }
