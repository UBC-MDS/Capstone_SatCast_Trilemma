# mae_with_std_penalty_np.py

import numpy as np

def mae_with_std_penalty_np(
    y_pred,
    y_true,
    std_weight=1.0,
    de_weight=1.0
):
    """
    Compute MAE with additional penalties:
    - std_penalty: penalizes difference in standard deviation between predictions and ground truth.
    - deviation error: penalizes shape mismatch in distribution after mean-centering.

    Parameters:
    -----------
    y_pred : np.ndarray
        Predicted values.
    y_true : np.ndarray
        Ground truth values.
    std_weight : float
        Weight for standard deviation penalty.
    de_weight : float
        Weight for deviation shape penalty.


    Returns:
    --------
    float
        Final loss value.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    base_loss = np.abs(y_pred - y_true)
    mae = base_loss.mean()

    # Std penalty
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    std_penalty = np.abs(pred_std - true_std)

    # Deviation penalty
    pred_dev = y_pred - np.mean(y_pred)
    true_dev = y_true - np.mean(y_true)
    dev_error = np.abs(pred_dev - true_dev)

    total_loss = base_loss + std_weight * std_penalty + de_weight * dev_error
    return total_loss.mean()
