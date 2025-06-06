# split_and_scale.py
# Author: Ximin 
# Date: 2025-06-04
# Description: Split time series data into train/validation sets and apply standard scaling.

import pandas as pd
from sklearn.preprocessing import StandardScaler

def split_and_scale(df: pd.DataFrame, PRED_STEPS: int):
    """
    Splits the DataFrame into training and validation sets and applies Z-score normalization.

    The split reserves the last PRED_STEPS × 2 timesteps: one for validation and one for test.
    Only real-valued features (excluding specified time or identifier fields) are standardized.
    The scaler is fit on the training data and reused on the validation data.

    Parameters
    ----------
    df : pd.DataFrame
        Input time-indexed dataframe containing time series features and a 'time_idx' column.
    PRED_STEPS : int
        Number of steps to forecast (typically 96 for 24 hours of hourly data).

    Returns
    -------
    df_train : pd.DataFrame
        Scaled training dataset. The output df_train will be timestamp from beginning to the 2 days before the end.
    df_valid : pd.DataFrame
        Scaled validation dataset.The output df_valid will be time index from beginning to the 1 day before the end. The last day is for testing purpose.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object, useful for inverse transforming or scaling test data.
    """

    # Determine the last index to use for training, excluding prediction windows
    last_idx = df.time_idx.max() - PRED_STEPS
    training_cutoff = last_idx - PRED_STEPS

    # Split datasets based on cutoff indices
    df_train = df[df.time_idx <= training_cutoff].copy()
    df_valid = df[df.time_idx <= last_idx].copy()

    # Define excluded columns
    EXCLUDE = [
        "target", "time_idx", "hour", "minute", "day_of_week", "month",
        "series_id", "timestamp", "price_USD",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "month_sin", "month_cos", "minute_sin", "minute_cos",
    ]
    EXCLUDE_PREFIXES = ["mempool_fee_histogram_bin"]

    # Identify numeric columns to normalize (excluding unwanted ones)
    float_cols = df.select_dtypes("float64").columns
    num_cols = [
        col for col in float_cols
        if col not in EXCLUDE and not any(col.startswith(p) for p in EXCLUDE_PREFIXES)
    ]

    # Initialize and apply standard scaler
    scaler = StandardScaler()
    df_train.loc[:, num_cols] = scaler.fit_transform(df_train[num_cols])
    df_valid.loc[:, num_cols] = scaler.transform(df_valid[num_cols])

    return df_train, df_valid, scaler
