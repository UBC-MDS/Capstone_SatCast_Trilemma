"""
deepar_create_dataloaders.py

Builds TimeSeriesDataSet objects and DataLoaders for training and validation.
"""

import os
from pytorch_forecasting import TimeSeriesDataSet
from torch.utils.data import DataLoader
from pytorch_forecasting.data import GroupNormalizer

def deepar_create_dataloaders(df_train, df_valid, enc_len, pred_steps, batch_size):
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
    train_ds : TimeSeriesDataSet
        Training dataset object.
    train_dl : DataLoader
        Training dataloader.
    val_dl : DataLoader
        Validation dataloader.
    """
    last_idx = df_valid.time_idx.max()  
    training_cutoff = last_idx - pred_steps
    TimeSeriesDataSet.verbose = True

    # Create training dataset
    train_ds = TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",  # Time index column (required)
        target="target",  # Target variable for prediction
        group_ids=["series_id"],  # Identifier for each time series instance
        max_encoder_length=enc_len,  # Full encoder context (7 days)
        min_encoder_length=enc_len // 2,
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
        ],
        # Real-valued covariates not known at prediction time (only target because deepar do not accept unmatched encoder and decoder)
        time_varying_unknown_reals=["target"] ,
        # Normalize target separately per series
        target_normalizer=GroupNormalizer(groups=["series_id"]),
    )

    val_ds = TimeSeriesDataSet.from_dataset(train_ds, df_valid, predict=True, stop_randomization=True, min_prediction_idx=training_cutoff + 1)

    num_workers = min(4, os.cpu_count())
    train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, shuffle=False, num_workers=num_workers, persistent_workers=True)
    val_dl = val_ds.to_dataloader(train=False, batch_size=batch_size * 10, shuffle=False, num_workers=num_workers, persistent_workers=True)

    return train_ds, train_dl, val_dl
