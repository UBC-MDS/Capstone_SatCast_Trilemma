"""
transform_fee_data_dl.py

This module performs:
- Reshaping of resampled data into long-format series
- Time-based feature extraction
- Cyclical encoding of time variables
- Time index creation
- Log transformation of histogram bins

Author: Ximin Xu
Date: 2025-06-04
"""

import pandas as pd
import numpy as np

def transform_fee_data_dl(df: pd.DataFrame, reference_time_str: str = "2025-03-05 02:00:00") -> pd.DataFrame:
    """
    Transform cleaned and resampled Bitcoin fee data into TFT-ready format.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned and resampled DataFrame from preprocessing step.
    reference_time_str : str, optional
        Timestamp string used to calculate relative time indices (default is "2025-03-05 02:00:00").

    Returns
    -------
    pd.DataFrame
        Long-format DataFrame with cyclical covariates and log-transformed histogram features.

    Raises
    ------
    ValueError
        If required columns are missing or input data is malformed.
    """

    # Required target columns for unpivoting
    fee_cols = [
        "recommended_fee_fastestFee",
        "recommended_fee_hourFee",
        "recommended_fee_halfHourFee",
        "recommended_fee_economyFee",
        "recommended_fee_minimumFee"
    ]

    # Check if required columns are present
    missing_cols = [col for col in fee_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required fee columns: {missing_cols}")

    # Ensure index contains timestamps
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must have a DatetimeIndex.")

    # Create long-format DataFrame for multi-series modeling
    df_long = df[fee_cols].copy()
    df_long["timestamp"] = df.index
    df_long = df_long.melt(id_vars=["timestamp"], var_name="series_id", value_name="target")

    # Merge with non-target features
    features = df.drop(columns=fee_cols).reset_index()
    df_long = df_long.merge(features, on="timestamp", how="left")

    # Sort and reset index
    df_long.sort_values(["series_id", "timestamp"], inplace=True)
    df_long.reset_index(drop=True, inplace=True)

    # Extract temporal features from timestamp
    df_long["hour"] = df_long["timestamp"].dt.hour
    df_long["minute"] = df_long["timestamp"].dt.minute
    df_long["day_of_week"] = df_long["timestamp"].dt.dayofweek
    df_long["month"] = df_long["timestamp"].dt.month

    # Create time_idx based on reference time
    try:
        reference_time = pd.Timestamp(reference_time_str)
    except Exception as e:
        raise ValueError(f"Invalid reference time: {reference_time_str} ({e})")

    df_long["time_idx"] = ((df_long["timestamp"] - reference_time) / pd.Timedelta(minutes=15)).astype(int)

    # Encode cyclical features
    df_long["hour_sin"] = np.sin(2 * np.pi * df_long["hour"] / 24)
    df_long["hour_cos"] = np.cos(2 * np.pi * df_long["hour"] / 24)
    df_long["day_of_week_sin"] = np.sin(2 * np.pi * df_long["day_of_week"] / 7)
    df_long["day_of_week_cos"] = np.cos(2 * np.pi * df_long["day_of_week"] / 7)
    df_long["month_sin"] = np.sin(2 * np.pi * df_long["month"] / 12)
    df_long["month_cos"] = np.cos(2 * np.pi * df_long["month"] / 12)
    df_long["minute_sin"] = np.sin(2 * np.pi * df_long["minute"] / 60)
    df_long["minute_cos"] = np.cos(2 * np.pi * df_long["minute"] / 60)

    # Drop raw time columns (since they are now encoded)
    df_long.drop(columns=["hour", "minute", "day_of_week", "month"], inplace=True)

    # Log1p-transform histogram bins if present
    hist_cols = [col for col in df_long.columns if col.startswith("mempool_fee_histogram_bin")]
    if hist_cols:
        df_long[hist_cols] = np.log1p(df_long[hist_cols])
    else:
        print("Warning: No histogram columns found to log-transform.")

    return df_long
