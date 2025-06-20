# predict_hwes.py
# author: Ximin Xu
# date: 2025-06-18

"""
Script to run inference using a pretrained Holt-Winters Exponential Smoothing (HWES) model
for Bitcoin transaction fee forecasting.

This script performs the following steps:
1. Loads a serialized HWES model from a pickle file.
2. Forecasts the next N steps, where N is the length of the provided test set.
3. Assembles a forecast DataFrame with predicted and actual values.
4. Computes evaluation metrics including MAE, RMSE, MAPE, and a custom volatility-aware loss.

Usage:
------
Used by the full-model analysis pipeline (e.g., `analysis.py`) to evaluate HWES baseline performance.

Dependencies:
    - pandas
    - pickle
    - custom_loss_eval.eval_metrics
"""

import pickle
import pandas as pd
from custom_loss_eval import eval_metrics

def predict_hwes(df_test, model_path):
    """
    Run inference using a Holt-Winters Exponential Smoothing model and return forecast results and metrics.

    Parameters
    ----------
    df_test : pd.DataFrame
        DataFrame containing the test portion of the time series, including timestamps and true target values.
    model_path : str or Path
        Path to the saved HWES model pickle file.

    Returns
    -------
    df_forecast : pd.DataFrame
        DataFrame with columns: ['timestamp', 'y_pred', 'y_true', 'series_id']
        Contains forecasted and actual values for 'recommended_fee_fastestFee'.
    metrics : dict
        Dictionary of evaluation metrics including custom loss, MAE, RMSE, and MAPE.
    """
    # Step 1: Load HWES model from pickle
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Step 2: Forecast as many steps as the test set contains
    y_pred = model.forecast(len(df_test))

    # Step 3: Construct forecast DataFrame
    df_forecast = pd.DataFrame({
        "timestamp": df_test["timestamp"].values,
        "y_pred": y_pred,
        "y_true": df_test["recommended_fee_fastestFee"].values,
        "series_id": "recommended_fee_fastestFee"
    })

    # Step 4: Evaluate forecast accuracy using custom metrics
    metrics = eval_metrics(df_forecast["y_pred"], df_forecast["y_true"])

    return df_forecast, metrics
