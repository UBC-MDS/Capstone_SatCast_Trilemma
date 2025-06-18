"""
split_series.py

Splits a time-indexed DataFrame into training and validation sets for forecasting.
"""
import pandas as pd

def split_series(df: pd.DataFrame, PRED_STEPS: int):
    """
    Split a time-indexed DataFrame into training and validation sets using prediction window size.

    The validation set includes the most recent `PRED_STEPS` data points,
    while the training set includes everything before that.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with a 'time_idx' column indicating temporal order.
    PRED_STEPS : int
        Number of time steps to forecast into the future (e.g., 96 for 1-day horizon).

    Returns
    -------
    df_train : pd.DataFrame
        DataFrame containing historical data up to (max_idx - PRED_STEPS).
    df_valid : pd.DataFrame
        DataFrame containing data up to max_idx (includes future targets).
    
    Example
    -------
    >>> df_train, df_valid = split_series(df, PRED_STEPS=96)
    """

    last_idx = df.time_idx.max()
    training_cutoff = last_idx - PRED_STEPS

    df_train = df[df.time_idx <= training_cutoff].copy()
    df_valid = df[df.time_idx <= last_idx].copy()

    return df_train, df_valid
