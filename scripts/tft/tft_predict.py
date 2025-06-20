# tft_predict.py
# author: Ximin Xu
# date: 2025-06-18

"""
Script to run inference using a pretrained Temporal Fusion Transformer (TFT) model
for Bitcoin transaction fee forecasting.

This script performs the following steps:
1. Preprocesses the full input dataset to match TFT input requirements.
2. Splits the data into training and validation sets, reserving the final 96 steps for evaluation.
3. Applies scaling transformations to the validation window.
4. Loads a saved TFT model checkpoint from disk.
5. Runs prediction on the final 96 time steps of 'recommended_fee_fastestFee'.
6. Assembles predictions and actuals into a forecast DataFrame.
7. Computes evaluation metrics including MAE, RMSE, MAPE, and a custom volatility-aware loss.

Usage:
    Called by `analysis.py` during the full-model evaluation stage.

Returns:
    - df_forecast: DataFrame with ['timestamp', 'y_pred', 'y_true', 'series_id'].
    - metrics: Dictionary with forecast accuracy scores.

Dependencies:
    - torch, pytorch_forecasting, lightning.pytorch, pandas, numpy
    - Custom modules: transform_fee_data_dl, split_series, scale_series, eval_metrics
"""


import torch
import pandas as pd
import numpy as np
from torch.serialization import add_safe_globals
from pytorch_forecasting import TemporalFusionTransformer
from custom_loss_eval import eval_metrics
from transform_fee_data_dl import transform_fee_data_dl
from split_series import split_series
from scale_series import scale_series
import sys
from pathlib import Path
import lightning.pytorch as pl

def predict_tft(df_full, model_path):
    """
    Run inference using a Temporal Fusion Transformer and return forecast results and metrics.

    Parameters
    ----------
    df_full : pd.DataFrame
        Full historical dataset with engineered features for model input.
    model_path : str or Path
        Path to the saved TFT model checkpoint (.pth or .pt file).

    Returns
    -------
    df_forecast : pd.DataFrame
        DataFrame with columns: ['timestamp', 'y_pred', 'y_true', 'series_id']
        Contains predictions and actual values for the last 96 time steps of 'recommended_fee_fastestFee'.
    metrics : dict
        Dictionary of evaluation metrics including custom loss, MAE, RMSE, and MAPE.
    """
    # Allow torch.load to deserialize TemporalFusionTransformer objects
    add_safe_globals([TemporalFusionTransformer])

    # Step 1: Transform input data into model-compatible format
    df_dl = transform_fee_data_dl(df_full)

    # Step 2: Split the data into train/validation sets using last 96 steps for validation
    _, df_valid = split_series(df_dl, 96)

    # Step 3: Apply scaling to the validation set
    _, df_valid, _ = scale_series(_, df_valid)

    # Step 4: Extract true values for final 96 time steps
    df_true = df_valid.query("time_idx > time_idx.max() - 96")

    # Step 5: Load the TFT model from disk
    project_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(project_root / "scripts" / "advanced_tft"))
    model = torch.load(model_path, map_location="cpu", weights_only=False)

    # Step 6: Ensure reproducibility
    pl.seed_everything(42)

    # Step 7: Generate raw predictions using GPU (if available)
    pred = model.predict(
        df_valid,
        mode="raw",
        return_index=True,
        return_x=True,
        trainer_kwargs={"accelerator": "gpu"}
    )

    # Step 8: Extract predictions from output
    y_pred = pred.output.prediction.detach().cpu().numpy().flatten()

    # Step 9: Construct forecast DataFrame for the target series
    df_forecast = pd.DataFrame({
        "timestamp": df_true["timestamp"],
        "y_pred": y_pred,
        "y_true": df_true["target"],
        "series_id": df_true["series_id"]
    }).query("series_id == 'recommended_fee_fastestFee'")

    # Step 10: Evaluate forecast accuracy using custom metrics
    metrics = eval_metrics(df_forecast["y_pred"], df_forecast["y_true"])

    return df_forecast, metrics
