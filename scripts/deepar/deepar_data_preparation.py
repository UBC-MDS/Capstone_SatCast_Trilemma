"""
deepar_data_preparation.py

Utility functions for loading, preprocessing, and scaling Bitcoin mempool & fee data 
for use with the DeepAR model.
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

def deepar_prepare_data(parquet_path, pred_steps):
    """
    Load and process raw data for deepAR modeling.

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
    df = df.iloc[:-96]
    df = transform_fee_data_dl(df)
    df_train, df_valid = split_series(df, pred_steps)
    df_train, df_valid, scaler = scale_series(df_train, df_valid)
    return df, df_train, df_valid, scaler
