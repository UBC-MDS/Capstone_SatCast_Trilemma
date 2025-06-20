# sarima_predict.py
# author: Ximin Xu
# date: 2025-06-18

"""
Forecast Bitcoin transaction fees using a pretrained SARIMA model.

This script loads a serialized SARIMA model from disk, forecasts over the given test
horizon, and returns a DataFrame with predicted and actual values for the 
'recommended_fee_fastestFee' series. The forecasted values are exponentiated to reverse 
the log1p transformation applied during training.

Dependencies:
- pandas, numpy, pickle
- Custom module: eval_metrics from custom_loss_eval
"""

import pickle
import pandas as pd
import numpy as np
from custom_loss_eval import eval_metrics

def predict_sarima(df_test, model_path):
    """
    Run inference using a SARIMA model and return forecast results and evaluation metrics.

    Parameters
    ----------
    df_test : pd.DataFrame
        DataFrame containing timestamps and true target values for testing.
    model_path : str or Path
        Path to the saved SARIMA model pickle file.

    Returns
    -------
    df_forecast : pd.DataFrame
        DataFrame with columns: ['timestamp', 'y_pred', 'y_true', 'series_id']
        Contains forecasted and actual values for 'recommended_fee_fastestFee'.
    metrics : dict
        Dictionary of evaluation metrics including custom loss, MAE, RMSE, and MAPE.
    """
    # Step 1: Load SARIMA model from pickle file
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # Step 2: Forecast over the same horizon as the test set
    fh = list(range(1, len(df_test) + 1))
    y_pred_log = model.predict(fh=fh)

    # Step 3: Invert the log1p transformation (if applied during training)
    y_pred = np.expm1(y_pred_log)

    # Step 4: Construct forecast DataFrame
    df_forecast = pd.DataFrame({
        "timestamp": df_test["timestamp"].values,
        "y_pred": y_pred.values,
        "y_true": df_test["recommended_fee_fastestFee"].values,
        "series_id": "recommended_fee_fastestFee"
    })

    # Step 5: Evaluate forecast accuracy
    metrics = eval_metrics(df_forecast["y_pred"], df_forecast["y_true"])

    return df_forecast, metrics
