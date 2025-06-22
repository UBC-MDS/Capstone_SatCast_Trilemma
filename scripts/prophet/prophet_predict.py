# prophet_predict.py
# author: Ximin Xu
# date: 2025-06-18

"""
Forecast Bitcoin transaction fees using a pretrained Prophet model.

This script loads a serialized Prophet model, generates forecasts for a given test set horizon,
and returns a DataFrame containing predictions and actual values for the
'recommended_fee_fastestFee' series. Forecasted values are exponentiated to reverse
log1p transformation applied during training.

Dependencies:
- prophet
- pandas, numpy
- Custom module: eval_metrics from custom_loss_eval
"""

import pandas as pd
import numpy as np
from prophet.serialize import model_from_json

from pathlib import Path
import sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "scripts" / "prophet"))

from custom_loss_eval import eval_metrics

def predict_prophet(df_test, model_path):
    """
    Run inference using a Prophet model and return forecast results and evaluation metrics.

    Parameters
    ----------
    df_test : pd.DataFrame
        DataFrame containing test timestamps and true target values.
    model_path : str or Path
        Path to the saved Prophet model JSON file.

    Returns
    -------
    df_forecast : pd.DataFrame
        DataFrame with columns: ['timestamp', 'y_pred', 'y_true', 'series_id']
        Contains forecasted and actual values for 'recommended_fee_fastestFee'.
    metrics : dict
        Dictionary of evaluation metrics including custom loss, MAE, RMSE, and MAPE.
    """
    # Step 1: Load the Prophet model from JSON file
    with open(model_path, "r") as f:
        model = model_from_json(f.read())

    # Step 2: Create future dataframe extending into the forecast horizon
    future = model.make_future_dataframe(periods=len(df_test), freq='15min')

    # Step 3: Predict using the Prophet model
    forecast = model.predict(future)

    # Step 4: Extract forecasted values, reversing log1p transformation
    y_pred = np.expm1(forecast.iloc[-len(df_test):]["yhat"])

    # Step 5: Construct forecast DataFrame
    df_forecast = pd.DataFrame({
        "timestamp": df_test["timestamp"].values,
        "y_pred": y_pred.values,
        "y_true": df_test["recommended_fee_fastestFee"].values,
        "series_id": "recommended_fee_fastestFee"
    })

    # Step 6: Evaluate metrics
    metrics = eval_metrics(df_forecast["y_pred"], df_forecast["y_true"])

    return df_forecast, metrics
