# scale_series.py
# Author: Ximin
# Date: 2025-06-07
# Description: Apply Z-score normalization to selected real-valued features.

import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_series(df_train: pd.DataFrame, df_valid: pd.DataFrame):
    """
    Applies standard scaling to selected real-valued features.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training set to fit the scaler.
    df_valid : pd.DataFrame
        Validation set to transform using the fitted scaler.

    Returns
    -------
    df_train_scaled : pd.DataFrame
        Scaled training set.
    df_valid_scaled : pd.DataFrame
        Scaled validation set.
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler object.
    """
    EXCLUDE = [
        "target", "time_idx", "hour", "minute", "day_of_week", "month",
        "series_id", "timestamp", "price_USD",
        "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
        "month_sin", "month_cos", "minute_sin", "minute_cos",
    ]
    EXCLUDE_PREFIXES = ["mempool_fee_histogram_bin"]

    float_cols = df_train.select_dtypes("float64").columns
    num_cols = [
        col for col in float_cols
        if col not in EXCLUDE and not any(col.startswith(p) for p in EXCLUDE_PREFIXES)
    ]

    scaler = StandardScaler()
    df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
    df_valid[num_cols] = scaler.transform(df_valid[num_cols])

    return df_train, df_valid, scaler


