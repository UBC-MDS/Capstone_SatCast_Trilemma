# scripts/analysis/predict_xgboost.py

"""
Forecast Bitcoin transaction fees using a pretrained XGBoost model.

This script constructs lag features from the full dataset, loads a trained XGBoost model,
predicts future fee values for the 'recommended_fee_fastestFee' series, and returns
a forecast DataFrame along with evaluation metrics.

Dependencies:
- pandas, joblib
- Custom modules: create_lag_features_fast, data_split, eval_metrics
"""

import pandas as pd
import joblib
from XGBoost import create_lag_features_fast, data_split
from custom_loss_eval import eval_metrics

def predict_xgboost(df_full, df_test, model_path):
    """
    Run inference using an XGBoost model and return forecast results and evaluation metrics.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full historical dataset used for feature construction.
    df_test : pd.DataFrame
        Test set with timestamps and true target values.
    model_path : str or Path
        Path to the saved XGBoost model file (.joblib).

    Returns
    -------
    df_forecast : pd.DataFrame
        DataFrame with columns: ['timestamp', 'y_pred', 'y_true', 'series_id']
        Contains forecasted and actual values for 'recommended_fee_fastestFee'.
    metrics : dict
        Dictionary of evaluation metrics including custom loss, MAE, RMSE, and MAPE.
    """
    # Step 1: Create lag features from the full dataset
    lags = range(1, 193)
    df_xgb_full = create_lag_features_fast(df_full, 'recommended_fee_fastestFee', lags)

    # Step 2: Split data to isolate X_test from the final segment (15 days)
    _, X_test, _, _ = data_split(df_xgb_full, 15)

    # Step 3: Load pretrained XGBoost model
    model = joblib.load(model_path)

    # Step 4: Predict using forecast horizon of same length as test set
    fh = list(range(1, len(df_test) + 1))
    y_pred = model.predict(fh=fh, X=X_test)

    # Step 5: Construct forecast DataFrame
    df_forecast = pd.DataFrame({
        "timestamp": df_test["timestamp"].values,
        "y_pred": y_pred.values,
        "y_true": df_test["recommended_fee_fastestFee"].values,
        "series_id": "recommended_fee_fastestFee"
    })

    # Step 6: Compute evaluation metrics
    metrics = eval_metrics(df_forecast["y_pred"], df_forecast["y_true"])

    return df_forecast, metrics
