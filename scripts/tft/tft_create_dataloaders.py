# tft_create_dataloaders.py
# author: Ximin Xu
# date: 2025-06-18
"""
Builds TimeSeriesDataSet objects and PyTorch DataLoaders for use with the Temporal Fusion Transformer (TFT) model.

This module performs the following:
1. Constructs a training `TimeSeriesDataSet` using the cleaned and feature-augmented input data.
2. Generates a validation dataset from the same configuration using `.from_dataset()`.
3. Automatically selects time-varying real covariates based on naming patterns.
4. Wraps both datasets into PyTorch DataLoaders with appropriate batching and multiprocessing setup.

Usage:
------
Used in a TFT training pipeline after data preparation and before model initialization:

    tft_ds, train_dl, val_dl = tft_make_dataloaders(df_train, df_valid, enc_len=672, pred_steps=96, batch_size=64)
"""



import os
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

def tft_make_dataloaders(df_train, df_valid, enc_len, pred_steps, batch_size):
    """
    Construct TimeSeriesDataSet and corresponding DataLoaders.

    Parameters:
    -----------
    df_train : pd.DataFrame
    df_valid : pd.DataFrame
    enc_len : int
        Length of encoder sequence.
    pred_steps : int
        Number of prediction steps.
    batch_size : int

    Returns:
    --------
    tft_ds : TimeSeriesDataSet
        Training dataset object.
    train_dl : DataLoader
        Training dataloader.
    val_dl : DataLoader
        Validation dataloader.
    """
    
    # All real covariates based on prefix
    real_covs = [c for c in df_train.columns if c.startswith(("mempool", "difficulty", "price"))]

    # Create training dataset
    tft_ds = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",  # Time index column (required)
        target="target",  # Target variable for prediction
        group_ids=["series_id"],  # Identifier for each time series instance
        min_encoder_length=enc_len // 2,  # Minimum history for padding samples
        max_encoder_length=enc_len,  # Full encoder context (7 days)
        min_prediction_length=1,
        max_prediction_length=pred_steps,  # Forecast horizon (1 day)
        static_categoricals=["series_id"],  # Per-series static info (e.g., ID)
        # Covariates known at prediction time (e.g., calendar)
        time_varying_known_reals=[
            "time_idx",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "minute_sin",
            "minute_cos",
        ] ,
        # Real-valued covariates not known at prediction time (e.g., mempool load)
        time_varying_unknown_reals=["target"] + real_covs,
        # Normalize target separately per series
        target_normalizer=GroupNormalizer(groups=["series_id"]),
        # Augmented features
        add_relative_time_idx=True,  # Add 0, 1, ..., N for time position
        add_target_scales=True,  # Add mean/std of target (per series)
        add_encoder_length=True,  # Add actual encoder length used
    )

    val_ds = TimeSeriesDataSet.from_dataset(tft_ds, df_valid, predict=True, stop_randomization=True)

    num_workers = min(4, os.cpu_count())
    train_dl = tft_ds.to_dataloader(train=True, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    val_dl = val_ds.to_dataloader(train=False, batch_size=batch_size * 10, shuffle=False, num_workers=num_workers, persistent_workers=True)

    return tft_ds, train_dl, val_dl
