import pandas as pd

def add_lag_features(df: pd.DataFrame, column: str, lag_step: int) -> pd.DataFrame:
    """
    Add a single lag feature column to the DataFrame for a specified column and lag step.
    NaN values introduced by shifting are filled with the median of the original column.

    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing the data.
    column : str
        The name of the column for which to create the lag feature.
    lag_step : int
        The lag step (number of time steps to shift the column by).

    Returns
    -------
    pd.DataFrame
        A copy of the original DataFrame with one additional lagged column,
        named as "<column>_lag_<lag_step>".

    Example
    -------
    >>> df_with_lag = add_lag_feature(df, "target", 1)
    """
    df_lagged = df.copy()
    lag_col_name = f"{column}_lag_{lag_step}"
    df_lagged[lag_col_name] = df_lagged.groupby("series_id")[column].shift(lag_step)

    # Fill NaNs in lag column with the median of the original column
    median_value = df_lagged[column].median()
    df_lagged[lag_col_name] = df_lagged[lag_col_name].fillna(median_value)

    return df_lagged
