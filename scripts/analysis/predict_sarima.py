# scripts/analysis/predict_sarima.py
import pickle
import pandas as pd
import numpy as np
from custom_loss_eval import eval_metrics

def predict_sarima(df_test, model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    fh = list(range(1, len(df_test) + 1))
    y_pred_log = model.predict(fh=fh)
    y_pred = np.expm1(y_pred_log)

    df_forecast = pd.DataFrame({
        "timestamp": df_test["timestamp"].values,
        "y_pred": y_pred.values,
        "y_true": df_test["recommended_fee_fastestFee"].values,
        "series_id": "recommended_fee_fastestFee"
    })

    metrics = eval_metrics(df_forecast["y_pred"], df_forecast["y_true"])
    return df_forecast, metrics