# scripts/analysis/predict_hwes.py
import pickle
import pandas as pd
from custom_loss_eval import eval_metrics

def predict_hwes(df_test, model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    y_pred = model.forecast(len(df_test))
    df_forecast = pd.DataFrame({
        "timestamp": df_test["timestamp"].values,
        "y_pred": y_pred,
        "y_true": df_test["recommended_fee_fastestFee"].values,
        "series_id": "recommended_fee_fastestFee"
    })

    metrics = eval_metrics(df_forecast["y_pred"], df_forecast["y_true"])
    return df_forecast, metrics