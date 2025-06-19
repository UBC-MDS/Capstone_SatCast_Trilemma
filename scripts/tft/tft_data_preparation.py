"""
tft_data_preparation.py

Handles preprocessing and feature engineering for Temporal Fusion Transformer (TFT) input data.

Responsibilities:
-----------------
1. Applies log transformations and target normalization.
2. Adds lag-based and rolling features to capture temporal dependencies.
3. Encodes time-based covariates and prepares final training-ready DataFrame.

Key Features:
-------------
- Includes fee rate smoothing and statistical aggregations.
- Automatically selects and constructs time-varying and static covariates.
- Ensures all required columns for TFT input format are generated.

Typical Usage:
--------------
Called before dataloader construction, this script transforms raw or pre-parqueted input data into the finalized DataFrame.

Returns:
--------
- Processed DataFrame ready for `TimeSeriesDataSet`.
"""

import sys
import os
import argparse
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
