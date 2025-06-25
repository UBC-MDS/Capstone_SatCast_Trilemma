# tft_data_preparation.py
# author: Ximin Xu
# date: 2025-06-18
"""
Script to preprocess raw Bitcoin fee data for use with the Temporal Fusion Transformer (TFT)
forecasting model.

This script performs the following steps:
1. Loads a raw Parquet file and applies initial preprocessing and smoothing.
2. Transforms the data into a TFT-compatible tabular structure with time-varying features.
3. Splits the dataset into training and validation subsets, reserving the last `pred_steps` as hold-out.
4. Scales the numeric columns using a fitted standard scaler.

Usage:
    Called before constructing the `TimeSeriesDataSet` for TFT model training or inference.

Dependencies:
    - preprocess_raw_parquet
    - transform_fee_data_dl
    - split_series
    - scale_series
"""


import sys
from pathlib import Path

# Setup project root and import paths
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src" 
sys.path.insert(0, str(src_path))

from preprocess_raw_parquet import preprocess_raw_parquet
from transform_fee_data_dl import transform_fee_data_dl
from split_series import split_series
from scale_series import scale_series

def tft_prepare_data(parquet_path, pred_steps):
    """
    Load and process raw data for TFT modeling.

    Parameters:
    -----------
    parquet_path : str
        Path to the input .parquet file.
    pred_steps : int
        Number of steps to forecast.

    Returns:
    --------
    df : pd.DataFrame
        Full cleaned dataset.
    df_train : pd.DataFrame
        Scaled training subset.
    df_valid : pd.DataFrame
        Scaled validation subset.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object used on numerical features.
    """
    df = preprocess_raw_parquet(parquet_path)
    df = df.iloc[:-pred_steps]
    df = transform_fee_data_dl(df)
    df_train, df_valid = split_series(df, pred_steps)
    df_train, df_valid, scaler = scale_series(df_train, df_valid)
    return df, df_train, df_valid, scaler
