# scripts/analysis/predict_deepar.py
import pandas as pd
import numpy as np
from pytorch_forecasting import DeepAR
from custom_loss_eval import eval_metrics
from transform_fee_data_dl import transform_fee_data_dl
from split_series import split_series
from scale_series import scale_series
import lightning.pytorch as pl

def predict_deepar(df_full, model_path):
    df_dl = transform_fee_data_dl(df_full)
    _, df_valid = split_series(df_dl, 96)
    _, df_valid, _ = scale_series(_, df_valid)

    df_true = df_valid.query("time_idx > time_idx.max() - 96")
    model = DeepAR.load_from_checkpoint(model_path)
    model.eval()
    pl.seed_everything(42)
    pred = model.predict(df_valid, mode="raw", return_x=True, trainer_kwargs={"accelerator": "cpu"})
    y_pred = np.median(pred.output.prediction.detach().cpu().numpy(), axis=-1).flatten()

    df_forecast = pd.DataFrame({
        "timestamp": df_true["timestamp"],
        "y_pred": y_pred,
        "y_true": df_true["target"],
        "series_id": df_true["series_id"]
    }).query("series_id == 'recommended_fee_fastestFee'")

    metrics = eval_metrics(df_forecast["y_pred"], df_forecast["y_true"])
    return df_forecast, metrics