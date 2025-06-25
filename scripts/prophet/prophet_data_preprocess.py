# prophet_data_preprocess.py
# author: Tengwei Wang
# date: 2025-06-18

"""
prophet_data_preprocess.py

Prepares Bitcoin fee data for Prophet model training.

Responsibilities:
-----------------
1. Loads and filters relevant columns for Prophet’s input format.
2. Converts timestamp to 'ds', fee target to 'y', and ensures proper types.
3. Splits into Prophet-ready training data and original target for holiday model logic.

Key Features:
-------------
- Standardizes data format required by Prophet.
- Supports selection of forecasting targets (e.g., fastestFee).
- Compatible with downstream optimization and training scripts.

Typical Usage:
--------------
Called by Prophet training or tuning pipelines to produce clean input for forecasting.

Returns:
--------
- `df` : Cleaned DataFrame with 'ds' and 'y' columns.
- `y_train` : Original untransformed target series.
"""

 
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from preprocess_raw_parquet import preprocess_raw_parquet
import numpy as np

def data_preprocess(df):
    """
    Preprocess the dataset for optimizing the Prophet model.

    This function:
    - Loads a DataFrame from the given path.
    - Renames columns to Prophet-required format (`ds`, `y`).
    - Returns both the processed DataFrame and the original target values.

    Parameters
    ----------
    df : str
        Path to the training dataset in Parquet format.

    Returns
    -------
    pd.DataFrame
        Prophet-ready DataFrame with 'ds' and 'y' columns.
    pd.Series
        Original untransformed target values (for use in holiday configuration).

    Example
    -------
    >>> df, y_train = data_preprocess("data/processed/fee_data.parquet")
    >>> df.head()
          ds     y
    0 2024-06-01  21.3
    1 2024-06-01  23.0
    """

    df_new = preprocess_raw_parquet(df)
    df_new.dropna(inplace = True)

    df_new = df_new.iloc[:-96]
    y_new = df_new["recommended_fee_fastestFee"]
    X_new = df_new.drop(columns = "recommended_fee_fastestFee")
    X_new = X_new.reset_index()
    X_new = X_new.drop(columns = "timestamp")

    # last 24h as test
    split_index = len(X_new) - 96

    X_train_new, X_test_new = X_new.iloc[:split_index], X_new.iloc[split_index:]
    y_train_new, y_test_new = y_new.iloc[:split_index], y_new.iloc[split_index:]

    df_prophet_new = y_train_new.reset_index()
    df_prophet_new = df_prophet_new.rename(columns={
        'timestamp': 'ds',
        'recommended_fee_fastestFee': 'y'
    })
    df_prophet_new['y'] = np.log1p(df_prophet_new['y'])
    
    return df_prophet_new,X_train_new, X_test_new, y_train_new, y_test_new
