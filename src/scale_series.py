"""
scale_series.py

Applies standard scaling to real-valued features in training and validation datasets.
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler

def scale_series(df_train: pd.DataFrame, df_valid: pd.DataFrame):
    """
    Apply Z-score normalization (standard scaling) to selected numeric features.

    This function:
    - Excludes timestamp, calendar encodings, ID fields, and histogram bins.
    - Applies scaling only to float64 columns that are not explicitly excluded.
    - Uses the training set to fit the scaler, then applies it to both sets.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training DataFrame to fit the scaler on.
    df_valid : pd.DataFrame
        Validation DataFrame to transform using the fitted scaler.

    Returns
    -------
    df_train_scaled : pd.DataFrame
        Scaled training DataFrame (in-place modified).
    df_valid_scaled : pd.DataFrame
        Scaled validation DataFrame (in-place modified).
    scaler : sklearn.preprocessing.StandardScaler
        The fitted StandardScaler object (can be reused on test data).

    Example
    -------
    >>> df_train_scaled, df_valid_scaled, scaler = scale_series(df_train, df_valid)
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


