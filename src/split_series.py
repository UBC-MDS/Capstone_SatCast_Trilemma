# split_series.py
# Author: Ximin
# Date: 2025-06-07
# Description: Split time series data into training and validation sets.

import pandas as pd

def split_series(df: pd.DataFrame, PRED_STEPS: int):
    """
    Splits the DataFrame into training and validation sets based on prediction steps.

    Parameters
    ----------
    df : pd.DataFrame
        Input time-indexed dataframe containing a 'time_idx' column.
    PRED_STEPS : int
        Number of steps to forecast (e.g., 96 for 24 hours of hourly data).

    Returns
    -------
    df_train : pd.DataFrame
        DataFrame from the beginning to two prediction windows before the end.
    df_valid : pd.DataFrame
        DataFrame from the beginning to one prediction window before the end.
    """
    last_idx = df.time_idx.max()
    training_cutoff = last_idx - PRED_STEPS

    df_train = df[df.time_idx <= training_cutoff].copy()
    df_valid = df[df.time_idx <= last_idx].copy()

    return df_train, df_valid
