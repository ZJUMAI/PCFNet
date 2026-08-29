"""Loss functions used by PET/CT classifiers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as functional


class FocalLoss(nn.Module):
    """Binary focal loss implemented on two-class logits."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        cross_entropy = functional.cross_entropy(logits, targets, reduction="none")
        probability = torch.exp(-cross_entropy)
        alpha_t = torch.where(
            targets == 1,
            torch.as_tensor(self.alpha, device=logits.device),
            torch.as_tensor(1.0 - self.alpha, device=logits.device),
        )
        return (alpha_t * (1.0 - probability).pow(self.gamma) * cross_entropy).mean()
