# scripts/analysis/predict_prophet.py
import pandas as pd
import numpy as np
from prophet.serialize import model_from_json
from custom_loss_eval import eval_metrics

def predict_prophet(df_test, model_path):
    with open(model_path, "r") as f:
        model = model_from_json(f.read())

    future = model.make_future_dataframe(periods=len(df_test), freq='15min')
    forecast = model.predict(future)
    y_pred = np.expm1(forecast.iloc[-len(df_test):]["yhat"])

    df_forecast = pd.DataFrame({
        "timestamp": df_test["timestamp"].values,
        "y_pred": y_pred.values,
        "y_true": df_test["recommended_fee_fastestFee"].values,
        "series_id": "recommended_fee_fastestFee"
    })

    metrics = eval_metrics(df_forecast["y_pred"], df_forecast["y_true"])
    return df_forecast, metrics