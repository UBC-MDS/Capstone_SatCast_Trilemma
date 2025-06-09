import numpy as np
import os
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    mean_squared_error
)

def std_diff(y_pred, y_true):
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    return np.abs(pred_std - true_std)

def dev_error_component(y_pred, y_true):
    pred_dev = y_pred - np.mean(y_pred)
    true_dev = y_true - np.mean(y_true)
    return np.abs(pred_dev - true_dev)

def custom_loss_eval(y_pred, y_true, std_weight=1.0, 
                     de_weight=1.0, clip_weight_std=None, clip_weight_dev=None):
    base_loss = np.abs(y_pred - y_true)
    mae = base_loss.mean()

    # Std penalty
    std_penalty = std_diff(y_pred, y_true)
    w_std = mae / (std_penalty + 1e-8)
    if clip_weight_std is not None:
        w_std = np.minimum(w_std, clip_weight_std)

    # Deviation penalty
    dev_error = dev_error_component(y_pred, y_true)
    w_dev = base_loss / (dev_error + 1e-8)
    if clip_weight_dev is not None:
        w_dev = np.minimum(w_dev, clip_weight_dev)

    # Total weighted loss
    total_loss = base_loss + std_weight * w_std * std_penalty + de_weight * w_dev * dev_error
    return total_loss.mean()

def eval_metrics(y_pred, y_true, std_weight=1.0, de_weight=1.0, 
                               clip_weight_std=None, clip_weight_dev=None):
    """
    Computes evaluation metrics for a forecast and returns as a DataFrame.

    Args:
        y_pred (np.ndarray or pd.Series): Forecasted values
        y_true (np.ndarray or pd.Series): True values
        std_weight (float): Weight for standard deviation penalty in custom loss
        de_weight (float): Weight for deviation error penalty in custom loss
        clip_weight_std (float or None): Optional clip for std weight
        clip_weight_dev (float or None): Optional clip for deviation weight

    Returns:
        pd.DataFrame: Single-row DataFrame with evaluation metrics
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_penalty_value = std_diff(y_pred, y_true)
    dev_error = dev_error_component(y_pred, y_true).mean()
    custom_loss = custom_loss_eval(y_pred, y_true, std_weight, de_weight, clip_weight_std, clip_weight_dev)

    metrics = {
        "custom_loss": custom_loss,
        "std_diff": std_penalty_value,
        "dev_error": dev_error,
        "mae": mae,
        "mape": mape,
        "rmse": rmse
    }

    return pd.DataFrame([metrics]).T.rename(columns={0: 'value'}).round(4)