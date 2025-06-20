# tft_custom_loss.py
# author: Ximin Xu
# date: 2025-06-18

"""
Defines a custom multi-horizon loss function for the Temporal Fusion Transformer (TFT),
designed to enhance sensitivity to temporal volatility and prediction shape fidelity in
Bitcoin transaction fee forecasting.

This module includes:
1. A base MAE loss between predictions and ground truth values.
2. A standard deviation penalty to align predicted and actual volatility.
3. A shape deviation penalty to match the temporal fluctuation patterns.
4. Configurable weights and optional clipping to stabilize training dynamics.

Usage:
Used during TFT initialization via:
    loss = MAEWithStdPenalty(std_weight=1.0, de_weight=1.0)
"""


import torch
from pytorch_forecasting.metrics import MultiHorizonMetric


class MAEWithStdPenalty(MultiHorizonMetric):
    """
    Custom multi-horizon loss combining MAE, standard deviation penalty, and deviation shape penalty.

    Parameters:
    ----------
    std_weight : float
        Weight for the standard deviation penalty term.
    de_weight : float
        Weight for the shape deviation penalty term.
    clip_weight_std : float or None
        Optional upper bound for standard deviation penalty weight.
    clip_weight_dev : float or None
        Optional upper bound for shape deviation penalty weight.
    reduction : str
        Specifies reduction mode ('mean' or 'none') for final loss.
    """

    def __init__(self, std_weight=1.0, de_weight=1.0, clip_weight_std=None, clip_weight_dev=None, reduction="mean", **kwargs):
        super().__init__(reduction=reduction, **kwargs)
        self.std_weight = std_weight
        self.de_weight = de_weight
        self.clip_weight_std = clip_weight_std
        self.clip_weight_dev = clip_weight_dev

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute the composite loss.

        Parameters:
        ----------
        y_pred : torch.Tensor
            Model predictions (batch_size x time_steps).
        target : torch.Tensor
            Ground truth values (batch_size x time_steps).

        Returns:
        -------
        torch.Tensor
            Computed loss tensor.
        """
        # Convert quantile outputs to point forecasts
        y_pred_point = self.to_prediction(y_pred)

        # Base MAE loss
        base_loss = torch.abs(y_pred_point - target)
        mae = base_loss.mean(dim=1, keepdim=True)

        # Standard deviation penalty: penalize variance mismatch
        pred_std = torch.std(y_pred_point, dim=1, keepdim=True)
        true_std = torch.std(target, dim=1, keepdim=True)
        std_penalty = torch.abs(pred_std - true_std)
        w_std = mae / (std_penalty + 1e-8)  # Weight inversely to std error
        if self.clip_weight_std is not None:
            w_std = w_std.clamp(max=self.clip_weight_std)

        # Shape deviation penalty: penalize mismatch in fluctuation patterns
        pred_mean = y_pred_point.mean(dim=1, keepdim=True)
        true_mean = target.mean(dim=1, keepdim=True)
        pred_dev = y_pred_point - pred_mean
        true_dev = target - true_mean
        dev_error = torch.abs(pred_dev - true_dev)
        w_dev = base_loss / (dev_error + 1e-8)  # Weight inversely to deviation error
        if self.clip_weight_dev is not None:
            w_dev = w_dev.clamp(max=self.clip_weight_dev)

        # Final composite loss
        return base_loss + self.std_weight * w_std * std_penalty + self.de_weight * w_dev * dev_error