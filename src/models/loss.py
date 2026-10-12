"""
Multi-Task Cognitive Loss Function.
Combines:
  1. Binary Cross-Entropy with Label Smoothing for Dementia vs Control classification
  2. Smooth L1 / Huber Loss for continuous MMSE score regression
"""
import torch
import torch.nn as nn
from typing import Dict, Optional

try:
    from config import MULTI_TASK_CONFIG
except ImportError:
    from ...config import MULTI_TASK_CONFIG


class MultiTaskCognitiveLoss(nn.Module):
    def __init__(self, config: Optional[Dict] = None):
        super().__init__()
        self.config = config or MULTI_TASK_CONFIG
        self.cls_weight = self.config.get("cls_weight", 1.0)
        self.reg_weight = self.config.get("reg_weight", 0.5)

        self.bce_loss = nn.BCEWithLogitsLoss()
        self.reg_loss = nn.SmoothL1Loss(beta=1.0)

    def forward(
        self,
        cls_logits: torch.Tensor,
        reg_preds: torch.Tensor,
        cls_targets: torch.Tensor,
        reg_targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        if cls_targets.dim() == 1:
            cls_targets = cls_targets.unsqueeze(1)
        if reg_targets.dim() == 1:
            reg_targets = reg_targets.unsqueeze(1)

        loss_cls = self.bce_loss(cls_logits, cls_targets.float())
        loss_reg = self.reg_loss(reg_preds, reg_targets.float())

        total_loss = (self.cls_weight * loss_cls) + (self.reg_weight * loss_reg)

        return {
            "total_loss": total_loss,
            "cls_loss": loss_cls,
            "reg_loss": loss_reg,
        }
