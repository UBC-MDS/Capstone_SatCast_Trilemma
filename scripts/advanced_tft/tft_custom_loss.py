"""
tft_custom_loss.py

Defines a custom loss function combining MAE, std penalty, and shape deviation penalty.
"""

import torch
from pytorch_forecasting.metrics import MultiHorizonMetric

class MAEWithStdPenalty(MultiHorizonMetric):
    """
    Custom multi-horizon loss combining MAE with std and deviation shape penalties.
    """
    def __init__(self, std_weight=1.0, de_weight=1.0, clip_weight_std=None, clip_weight_dev=None, reduction="mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.std_weight = std_weight
        self.de_weight = de_weight
        self.clip_weight_std = clip_weight_std
        self.clip_weight_dev = clip_weight_dev

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_pred_point = self.to_prediction(y_pred)
        base_loss = torch.abs(y_pred_point - target)
        mae = base_loss.mean(dim=1, keepdim=True)

        pred_std = torch.std(y_pred_point, dim=1, keepdim=True)
        true_std = torch.std(target, dim=1, keepdim=True)
        std_penalty = torch.abs(pred_std - true_std)
        w_std = mae / (std_penalty + 1e-8)
        if self.clip_weight_std is not None:
            w_std = w_std.clamp(max=self.clip_weight_std)

        pred_mean = y_pred_point.mean(dim=1, keepdim=True)
        true_mean = target.mean(dim=1, keepdim=True)
        pred_dev = y_pred_point - pred_mean
        true_dev = target - true_mean
        dev_error = torch.abs(pred_dev - true_dev)
        w_dev = base_loss / (dev_error + 1e-8)
        if self.clip_weight_dev is not None:
            w_dev = w_dev.clamp(max=self.clip_weight_dev)

        return base_loss + self.std_weight * w_std * std_penalty + self.de_weight * w_dev * dev_error
