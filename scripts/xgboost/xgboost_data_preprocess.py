# xgboost_data_preprocess.py
# author: Tengwei Wang
# date: 2025-06-18

"""
xgboost_data_preprocess.py

Script to preprocess raw Bitcoin fee data for XGBoost-based time series forecasting.

This script performs the following steps:
1. Loads a raw Parquet dataset and applies initial preprocessing logic.
2. Adds 48 hours (192 steps) of lagged features for the target variable.
3. Drops NA values and ensures proper alignment between features and forecast horizon.
4. Outputs a cleaned DataFrame ready for model training or optimization.

Usage:
    Called from XGBoost training and tuning scripts to provide model-ready inputs.

Dependencies:
    - preprocess_raw_parquet (for raw ingestion)
    - create_lag_features_fast (from xgboost_utils)

Returns:
    pd.DataFrame: Fully processed dataset with lag features and aligned timestamps.
"""


import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from preprocess_raw_parquet import preprocess_raw_parquet
src_path = project_root / "scripts" / "xgboost"
sys.path.insert(0, str(src_path))
from xgboost_utils import create_lag_features_fast

def data_preprocess(data_path):
    """
    Preprocess the dataset for optimizing the model. 

    Parameters:
    ----------
    data_path : str
        The path of training dataset. 

    Returns:
    -------
    pd.DataFrame
        Processed data.
    """
    df = preprocess_raw_parquet(data_path)
    df.dropna(inplace = True)
    lags = range(1, 193)  # 48 hours of 15-minute intervals
    df = create_lag_features_fast(df, 'recommended_fee_fastestFee', lags)
    df = df[:-96]
    return df
