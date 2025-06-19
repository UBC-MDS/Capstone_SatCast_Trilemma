"""
hwes_extract_split.py

Utility function to validate and split the target fee series from a preprocessed DataFrame
for use in time series forecasting models such as HWES or SARIMA.
"""

import os
import pandas as pd

def hwes_extract_split(df, forecast_horizon=192):
    """
    Validate, extract, and split the target series from a preprocessed DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Preprocessed dataframe that contains a 'recommended_fee_fastestFee' column.
    forecast_horizon : int
        Number of time steps to reserve for the test set (default: 192).

    Returns:
    --------
    target_series : pd.Series
        The full target series extracted from the dataframe.
    train : pd.Series
        Training portion of the series (chronologically before test).
    test : pd.Series
        Testing portion of the series (last `forecast_horizon` steps).
    """
    # Validate no zero values in target
    zero_count = (df['recommended_fee_fastestFee'] == 0).sum()
    print(f"Number of zero values in 'recommended_fee_fastestFee': {zero_count}")

    # Extract and sort target series
    target_series = df['recommended_fee_fastestFee'].astype(float).sort_index()
    target_series.index.freq = pd.infer_freq(target_series.index)

    # Perform temporal train-test split
    train = target_series[:-forecast_horizon]
    test = target_series[-forecast_horizon:]

    return target_series, train, test