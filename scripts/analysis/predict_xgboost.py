import pandas as pd
import joblib
from XGBoost import create_lag_features_fast, data_split
from custom_loss_eval import eval_metrics

def predict_xgboost(df_full, df_test, model_path):
    lags = range(1, 193)
    df_xgb_full = create_lag_features_fast(df_full, 'recommended_fee_fastestFee', lags)
    _, X_test, _, _ = data_split(df_xgb_full, 15)

    model = joblib.load(model_path)
    fh = list(range(1, len(df_test) + 1))
    y_pred = model.predict(fh=fh, X=X_test)

    df_forecast = pd.DataFrame({
        "timestamp": df_test["timestamp"].values,
        "y_pred": y_pred.values,
        "y_true": df_test["recommended_fee_fastestFee"].values,
        "series_id": "recommended_fee_fastestFee"
    })

    metrics = eval_metrics(df_forecast["y_pred"], df_forecast["y_true"])
    return df_forecast, metrics