import os
import pandas as pd

def extract_split(df, forecast_horizon=192):
    """
    Validate, extract, and split the target series from a preprocessed DataFrame.

    Parameters:
        df (pd.DataFrame): Preprocessed DataFrame containing 'recommended_fee_fastestFee'.
        forecast_horizon (int): Number of rows to reserve for the test set. Default is 192.

    Returns:
        train (pd.Series): Training portion of the time series.
        test (pd.Series): Testing portion of the time series.
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