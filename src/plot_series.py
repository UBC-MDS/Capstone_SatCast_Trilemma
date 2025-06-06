# plot_series.py
# Author: Ximin 
# Date: 2025-06-04
# Description:  visualizes the performance of a forecasting model by plotting the actual values (`y_true`) 
# and predicted values (`y_pred`) over time for a given `series_id`

import matplotlib.pyplot as plt
import pandas as pd

def plot_series(df_eval: pd.DataFrame, sid, ax=None):
    """
    Plot actual vs. predicted values for a specific time series.

    This function visualizes the performance of a forecasting model by plotting the actual values (`y_true`)
    and predicted values (`y_pred`) over time for a given `series_id`.

    Parameters
    ----------
    df_eval : pd.DataFrame
        DataFrame containing time series forecast results. Must include the following columns:
        - 'series_id': identifier for each series
        - 'timestamp': datetime values corresponding to predictions
        - 'y_true': actual observed values
        - 'y_pred': predicted values (e.g.,forecast)
    sid : str or int
        The series_id for which to generate the plot.
    ax : matplotlib.axes.Axes, optional
        A matplotlib Axes object to plot on. If None, a new figure and axes will be created.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The matplotlib Axes object containing the plot.

    Raises
    ------
    ValueError
        If required columns are missing or if the specified `sid` is not present in the DataFrame.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>> df_eval = pd.read_csv("evaluation_results.csv")  # Must include required columns
    >>> plot_series(df_eval, sid="recommended_fee_fastestFee")
    >>> plt.show()
    """
    # Validate input DataFrame
    required_cols = {"series_id", "timestamp", "y_true", "y_pred"}
    missing_cols = required_cols - set(df_eval.columns)
    if missing_cols:
        raise ValueError(f"Input DataFrame is missing required columns: {missing_cols}")

    # Filter the data
    subset = df_eval[df_eval["series_id"] == sid]
    if subset.empty:
        raise ValueError(f"No data found for series_id '{sid}'")

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 5))
    else:
        fig = ax.figure

    # Plot actual vs. predicted values
    ax.plot(subset["timestamp"], subset["y_true"], label="Actual", color="black")
    ax.plot(subset["timestamp"], subset["y_pred"], label="Forecast", color="blue")

    # Axis labels and title
    ax.set_title(f"Series {sid} – Forecast vs Actual", fontsize=14)
    ax.set_xlabel("Timestamp", fontsize=12)
    ax.set_ylabel("Transaction Fee (sats/vB)", fontsize=12)

    # Improve appearance
    ax.grid(True)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    return ax
