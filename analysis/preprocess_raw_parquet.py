"""
preprocess_raw_parquet.py

This module handles:
- Loading raw Bitcoin fee and mempool histogram data from a Parquet file
- Dropping uninformative columns
- Converting timestamps and setting the time index
- Resampling to a fixed interval
- Basic linear interpolation

Author: Ximin Xu
Date: 2025-06-04
"""

import os
import pandas as pd


def preprocess_raw_parquet(parquet_path: str) -> pd.DataFrame:
    """
    Load and clean raw Bitcoin fee + mempool data for further preprocessing.

    Parameters
    ----------
    parquet_path : str
        Path to the raw Parquet file.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with 15-minute uniform resampling and interpolated values.

    Raises
    ------
    FileNotFoundError
        If the given Parquet file does not exist.
    ValueError
        If required columns like 'timestamp' are missing or timestamp conversion fails.
    """

    # Check if the file exists
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    # Load the data
    try:
        df_raw = pd.read_parquet(parquet_path)
    except Exception as e:
        raise ValueError(f"Failed to read Parquet file: {e}")

    # Drop histogram bins only if they exist
    cols_to_drop = [
        'mempool_fee_histogram_bin_300_350', 'mempool_fee_histogram_bin_350_400',
        'mempool_fee_histogram_bin_400_450', 'mempool_fee_histogram_bin_450_500',
        'mempool_fee_histogram_bin_500_550', 'mempool_fee_histogram_bin_550_600',
        'mempool_fee_histogram_bin_600_650', 'mempool_fee_histogram_bin_650_700',
        'mempool_fee_histogram_bin_700_750', 'mempool_fee_histogram_bin_750_800',
        'mempool_fee_histogram_bin_800_850', 'mempool_fee_histogram_bin_850_900',
        'mempool_fee_histogram_bin_900_950', 'mempool_fee_histogram_bin_950_1000',
        'mempool_fee_histogram_bin_1000_plus'
    ]
    df_raw.drop(columns=[col for col in cols_to_drop if col in df_raw.columns], inplace=True)

    # Ensure 'timestamp' column exists
    if "timestamp" not in df_raw.columns:
        raise ValueError("Missing 'timestamp' column in input data.")

    # Convert timestamp and sort
    try:
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"], unit="s")
    except Exception as e:
        raise ValueError(f"Failed to convert timestamp column: {e}")

    df_raw.sort_values("timestamp", inplace=True)
    df_raw.set_index("timestamp", inplace=True)

    # Drop all price_* columns except 'price_USD'
    price_cols_to_drop = [col for col in df_raw.columns if col.startswith("price_") and col != "price_USD"]
    df_raw.drop(columns=price_cols_to_drop, inplace=True)

    # Handle empty or corrupted input
    if df_raw.empty:
        raise ValueError("DataFrame is empty after preprocessing. Check input data integrity.")

    # Resample and interpolate
    try:
        df_resampled = df_raw.resample("15min").mean().interpolate(method="linear")
    except Exception as e:
        raise ValueError(f"Error during resampling or interpolation: {e}")

    return df_resampled
