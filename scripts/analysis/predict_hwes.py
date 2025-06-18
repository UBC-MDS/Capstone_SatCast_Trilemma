# scripts/analysis/predict_hwes.py

"""
Forecast Bitcoin transaction fees using a pretrained Holt-Winters Exponential Smoothing (HWES) model.

This script loads a HWES model from disk, uses it to forecast the next `n` time steps
(equal to the length of the test set), and returns a DataFrame containing the predicted
and actual values for the 'recommended_fee_fastestFee' series, along with evaluation metrics.

Dependencies:
- pandas
- pickle
- Custom module: eval_metrics from custom_loss_eval
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
