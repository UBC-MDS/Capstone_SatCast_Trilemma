# xgboost_utils.py
# author: Tengwei Wang
# date: 2025-06-18

"""
Utility functions for feature engineering and data preparation to support
XGBoost-based Bitcoin fee forecasting at 15-minute resolution.

This module provides:
1. Fast generation of lag features for tree-based models.
2. Target-aligned splitting of features and labels for forecasting.
3. Preprocessing logic tailored for 24-hour ahead prediction.

Intended to support model pipelines using sktime + XGBoost regressors.

Functions:
    - create_lag_features_fast: Adds lagged versions of the target column.
    - data_split: Aligns and splits feature/target data for training and evaluation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_error as mean_absolute_error_sktime
from sktime.split import SlidingWindowSplitter, ExpandingWindowSplitter
from sklearn.model_selection import ParameterSampler

import sys
from pathlib import Path
# Add src to system path for custom module imports
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def data_split(df, interval):
    """
    Shift and split the time series dataset into aligned training and test sets.

    The input DataFrame must contain a column 'recommended_fee_fastestFee',
    which will be shifted to forecast 24 hours ahead. The rest are treated as features.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing features and target.
    interval : int
        The time interval in minutes (e.g., 15 for 15-minute data).

    Returns
    -------
    X_train : pd.DataFrame
        Feature matrix for training.
    X_test : pd.DataFrame
        Feature matrix for testing.
    y_train : pd.Series
        Target vector for training.
    y_test : pd.Series
        Target vector for testing.

    Example
    -------
    >>> X_train, X_test, y_train, y_test = data_split(df, 15)
    """
    y = df["recommended_fee_fastestFee"]  # target variable
    X = df.drop(columns="recommended_fee_fastestFee")  # all other features

    # Shift features back by 24 hours (96 steps for 15-min intervals)
    shift_steps = int(24 * (60 / interval))
    X = X.shift(periods=shift_steps)
    X = X.iloc[shift_steps:]  # drop initial NaNs
    y = y.loc[X.index]  # align y with shifted X

    # Split last 24 hours as test set
    split_index = int(len(X) - shift_steps)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test


def create_lag_features_fast(df, target_col, lags):
    """
    Generate lagged versions of the target variable as new features.

    Useful for tree-based models like XGBoost to encode time-dependence.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing all raw data.
    target_col : str
        The name of the target column to lag.
    lags : list or array-like
        List of lag steps (in units of time intervals) to create.

    Returns
    -------
    pandas.DataFrame
        Original DataFrame concatenated with lagged target features.

    Example
    -------
    >>> df = create_lag_features_fast(df, 'recommended_fee_fastestFee', [1, 2, 3])
    """
    # Create a list of shifted versions of the target column
    lagged_dfs = [
        df[[target_col]].shift(lag).rename(columns={target_col: f'{target_col}_lag_{lag}'})
        for lag in lags
    ]

    # Concatenate original df with all lagged versions
    return pd.concat([df] + lagged_dfs, axis=1)
