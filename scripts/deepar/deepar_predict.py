# deepar_predict.py
# author: Ximin Xu
# date: 2025-06-18
"""
Forecast Bitcoin transaction fees using a pretrained DeepAR model.

This script loads the full historical data, applies the necessary preprocessing steps,
loads a DeepAR model from checkpoint, performs prediction on the latest 96 time steps,
and returns a DataFrame containing the forecasted and actual values for the
'recommended_fee_fastestFee' series, along with evaluation metrics.

Dependencies:
- pytorch_forecasting
- lightning.pytorch
- pandas, numpy
- Custom modules: transform_fee_data_dl, split_series, scale_series, eval_metrics
"""

import pandas as pd
import numpy as np
from pytorch_forecasting import DeepAR
from custom_loss_eval import eval_metrics
from transform_fee_data_dl import transform_fee_data_dl
from split_series import split_series
from scale_series import scale_series
import lightning.pytorch as pl

def predict_deepar(df_full, model_path):
    """
    Run inference using a DeepAR model and return forecast results and evaluation metrics.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full historical fee dataset.
    model_path : str or Path
        Path to the trained DeepAR model checkpoint.

    Returns
    -------
    df_forecast : pd.DataFrame
        DataFrame with columns: ['timestamp', 'y_pred', 'y_true', 'series_id']
        Contains predictions for the last 96 time steps of 'recommended_fee_fastestFee'.
    metrics : dict
        Dictionary of evaluation metrics including custom loss, MAE, RMSE, and MAPE.
    """
    # Step 1: Transform data into DeepAR-compatible format
    df_dl = transform_fee_data_dl(df_full)

    # Step 2: Perform series-wise temporal split (using last 96 steps as validation)
    _, df_valid = split_series(df_dl, 96)

    # Step 3: Apply scaling to the validation set (ignore train set here)
    _, df_valid, _ = scale_series(_, df_valid)

    # Step 4: Filter to the most recent 96 time steps for evaluation
    df_true = df_valid.query("time_idx > time_idx.max() - 96")

    # Step 5: Load trained DeepAR model and set to evaluation mode
    model = DeepAR.load_from_checkpoint(model_path)
    model.eval()

    # Step 6: Set deterministic behavior for reproducibility
    pl.seed_everything(42)

    # Step 7: Run prediction on the validation set using raw mode
    pred = model.predict(
        df_valid, 
        mode="raw", 
        return_x=True, 
        trainer_kwargs={"accelerator": "cpu"}
    )

    # Step 8: Extract median prediction across quantiles
    y_pred = np.median(pred.output.prediction.detach().cpu().numpy(), axis=-1).flatten()

    # Step 9: Construct forecast DataFrame for the target series
    df_forecast = pd.DataFrame({
        "timestamp": df_true["timestamp"],
        "y_pred": y_pred,
        "y_true": df_true["target"],
        "series_id": df_true["series_id"]
    }).query("series_id == 'recommended_fee_fastestFee'")

    # Step 10: Evaluate metrics
    metrics = eval_metrics(df_forecast["y_pred"], df_forecast["y_true"])

    return df_forecast, metrics
