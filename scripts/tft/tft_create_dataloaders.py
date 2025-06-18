"""
tft_create_dataloaders.py

Builds TimeSeriesDataSet objects and PyTorch DataLoaders for use with the Temporal Fusion Transformer (TFT) model.

Responsibilities:
-----------------
1. Constructs a training `TimeSeriesDataSet` from the preprocessed and lag-augmented training DataFrame.
2. Applies consistent configuration to generate a validation dataset using `.from_dataset()`.
3. Automatically identifies known and unknown real-valued covariates based on naming conventions.
4. Returns DataLoaders with proper batching and multiprocessing setup for training and evaluation.

Key Features:
-------------
- Uses GroupNormalizer for per-series normalization.
- Adds relative time index, target scales, and encoder length to improve learning.
- Handles multi-series setup using `series_id`.

Typical Usage:
--------------
Used in a TFT training pipeline after data preparation and before model initialization.

Returns:
--------
- `TimeSeriesDataSet` object (train config)
- `DataLoader` for training
- `DataLoader` for validation
"""


import os
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader
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

    # Lagged covariates are "known" because they reference past values
    known_lagged_covs = [
        c for c in df_train.columns
        if (
            "_lag_" in c and (
                c.startswith("mempool_") or
                c.startswith("difficulty_") or
                c.startswith("price_") or
                c.startswith("target")
            )
        )
    ]


    # Unlagged covariates from same sources are "unknown" at prediction time
    unknown_real_covs = [c for c in real_covs if c not in known_lagged_covs]

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
        ] + known_lagged_covs,
        # Real-valued covariates not known at prediction time (e.g., mempool load)
        time_varying_unknown_reals=["target"] + unknown_real_covs,
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
