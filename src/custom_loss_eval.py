import numpy as np
import os
import sys
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, 
    mean_absolute_percentage_error, 
    mean_squared_error
)

def std_diff(y_pred, y_true):
    """
    Compute the absolute difference between the standard deviations of 
    predicted and true values.

    Parameters
    ----------
    y_pred : array-like
        Predicted values.
    y_true : array-like
        Actual target values.

    Returns
    -------
    float
        Absolute difference of standard deviations.
    """
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    return np.abs(pred_std - true_std)

def dev_error_component(y_pred, y_true):
    """
    Compute the element-wise absolute difference in deviations from the mean 
    between predicted and true values.

    Parameters
    ----------
    y_pred : array-like
        Predicted values.
    y_true : array-like
        Actual target values.

    Returns
    -------
    np.ndarray
        Array of absolute differences in deviations from the mean.
    """
    pred_dev = y_pred - np.mean(y_pred)
    true_dev = y_true - np.mean(y_true)
    return np.abs(pred_dev - true_dev)

def custom_loss_eval(y_pred, y_true, std_weight=1.0, de_weight=1.0):
    """
    Custom loss function that penalizes:
    - Standard prediction error (MAE)
    - Standard deviation mismatch between prediction and ground truth
    - Deviation error from global mean

    Parameters
    ----------
    y_pred : array-like
        Predicted values.
    y_true : array-like
        Actual target values.
    std_weight : float, optional
        Weight for standard deviation mismatch penalty. Default is 1.0.
    de_weight : float, optional
        Weight for deviation-from-mean penalty. Default is 1.0.

    Returns
    -------
    float
        Weighted custom loss score.
    """
    base_loss = np.abs(y_pred - y_true)
    mae = base_loss.mean()

    std_penalty = std_diff(y_pred, y_true)
    w_std = mae / (std_penalty + 1e-8)
    if std_weight is not None:
        w_std = std_weight

    dev_error = dev_error_component(y_pred, y_true)
    w_dev = base_loss / (dev_error + 1e-8)
    if de_weight is not None:
        w_dev = de_weight

    total_loss = base_loss + std_weight * std_penalty + de_weight * dev_error
    return total_loss.mean()

def eval_metrics(y_pred, y_true, std_weight=1.0, de_weight=1.0):
    """
    Computes evaluation metrics for model forecast vs. actual data.

    Metrics:
    --------
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    - RMSE: Root Mean Squared Error
    - std_diff: Absolute standard deviation difference
    - dev_error: Mean deviation-from-mean mismatch
    - custom_loss: Weighted aggregate loss combining all the above

    Parameters
    ----------
    y_pred : array-like
        Predicted values.
    y_true : array-like
        Actual target values.
    std_weight : float, optional
        Weight for standard deviation penalty. Default is 1.0.
    de_weight : float, optional
        Weight for deviation error penalty. Default is 1.0.

    Returns
    -------
    pd.DataFrame
        Transposed single-row DataFrame of evaluation metrics.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_penalty_value = std_diff(y_pred, y_true)
    dev_error = dev_error_component(y_pred, y_true).mean()
    custom_loss = custom_loss_eval(y_pred, y_true, std_weight, de_weight)

    metrics = {
        "custom_loss": custom_loss,
        "std_diff": std_penalty_value,
        "dev_error": dev_error,
        "mae": mae,
        "mape": mape,
        "rmse": rmse
    }

    return pd.DataFrame([metrics]).T.rename(columns={0: 'value'}).round(4)
