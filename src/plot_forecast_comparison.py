"""
plot_forecast_comparison.py

Visualizes and compares forecasts from two models against actual values for a given series.
"""
import matplotlib.pyplot as plt

def plot_forecast_comparison(df1, label1, df2, label2, sid="recommended_fee_fastestFee", color1="blue", color2="green"):
    """
    Plot two forecast model outputs alongside actual values for a specific time series.

    This function compares the predicted values from two different models against
    the true target values for a given `series_id`.

    Parameters
    ----------
    df1 : pd.DataFrame
        First forecast result DataFrame. Must include columns:
        ['series_id', 'timestamp', 'y_true', 'y_pred'].
    label1 : str
        Label for the forecast line from df1 (e.g., "Forecast (HWES)").
    df2 : pd.DataFrame
        Second forecast result DataFrame. Must include the same columns as df1.
    label2 : str
        Label for the forecast line from df2 (e.g., "Forecast (SARIMA)").
    sid : str, optional
        The series_id to filter and plot. Default is "recommended_fee_fastestFee".
    color1 : str, optional
        Line color for df1's forecast. Default is "blue".
    color2 : str, optional
        Line color for df2's forecast. Default is "green".

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib Figure object containing the plot.
    ax : matplotlib.axes.Axes
        The matplotlib Axes object with the plotted content.

    Raises
    ------
    ValueError
        If the specified `series_id` is missing in one or both DataFrames.

    Example
    -------
    >>> fig, ax = plot_forecast_comparison(
    ...     df1=forecast_df_hwes,
    ...     label1="HWES",
    ...     df2=forecast_df_sarima,
    ...     label2="SARIMA",
    ...     sid="recommended_fee_fastestFee"
    ... )
    >>> plt.show()
    """
    # Subset both DataFrames by series_id
    subset1 = df1[df1["series_id"] == sid]
    subset2 = df2[df2["series_id"] == sid]

    if subset1.empty or subset2.empty:
        raise ValueError(f"No data found for series_id '{sid}' in one or both dataframes.")

    fig, ax = plt.subplots(figsize=(12, 4))

    # Plot actual values from df1 (assumed identical across both)
    ax.plot(subset1["timestamp"], subset1["y_true"], label="Actual", color="black")

    # Plot df1 forecast
    ax.plot(subset1["timestamp"], subset1["y_pred"], label=label1, color=color1)

    # Plot df2 forecast
    ax.plot(subset2["timestamp"], subset2["y_pred"], label=label2, color=color2)

    # Final formatting
    ax.set_title(f"{label1} and {label2} vs Actual", fontsize=14)
    ax.set_xlabel("Timestamp", fontsize=12)
    ax.set_ylabel("Transaction Fee (sats/vB)", fontsize=12)
    ax.grid(True)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()

    return fig, ax
