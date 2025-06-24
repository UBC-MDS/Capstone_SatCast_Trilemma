# eda_utils.py
# Author: [Your Name]
# Date: 2025-06-23

"""
Utility functions for the EDA notebook in the Bitcoin transaction fee forecasting project.

This script provides helper functions to streamline visualization and analysis during exploratory data analysis (EDA).
These functions are called directly in the EDA notebook to maintain a clean, modular workflow.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose

def plot_feature_group(df, prefix, max_cols=3, figsize=(16, 10)):
    """
    Plots the distribution (histogram with KDE) of all features in the DataFrame that start with a given prefix.

    Parameters:
    ----------
    df : pd.DataFrame
        The DataFrame containing the feature columns to be plotted.
    prefix : str
        The prefix string used to filter column names.
    max_cols : int, optional (default=3)
        Maximum number of plots per row.
    figsize : tuple, optional (default=(16, 10))
        Size of the entire figure.

    Returns:
    -------
    None
        Displays the matplotlib figure inline.
    Example
    -------
    >>> plot_feature_group(df, prefix="mempool_")
    """
    group_cols = [col for col in df.columns if col.startswith(prefix)]
    n = len(group_cols)
    rows = (n + max_cols - 1) // max_cols
    fig, axes = plt.subplots(rows, max_cols, figsize=figsize, squeeze=False)
    
    for i, col in enumerate(group_cols):
        ax = axes[i // max_cols][i % max_cols]
        sns.histplot(df[col].dropna(), kde=True, ax=ax)
        ax.set_title(col)
        ax.set_ylabel("Frequency")
    for j in range(i+1, rows * max_cols):
        fig.delaxes(axes[j // max_cols][j % max_cols])

    fig.suptitle(f"Distribution of {prefix} Features", fontsize=16)
    fig.tight_layout()
    plt.show()


def plot_decomposition_custom(series, model, period, title, save_path=None, figsize=(10, 6)):
    """
    Decompose a time series and plot trend, seasonality, and residuals.

    Parameters
    ----------
    series : pd.Series
        Time series to decompose (indexed by datetime).
    model : str
        Decomposition model: 'additive' or 'multiplicative'.
    period : int
        Seasonal period (e.g., 288 = 1 day at 5-min intervals, 96 = 1 day at 15-min).
    title : str
        Plot title.
    save_path : str or None, optional
        File path to save the figure. If None, figure is not saved.
    figsize : tuple, optional
        Size of the figure.

    Returns
    -------
    None
        Displays the decomposition plot.
    Example
    -------
    >>> plot_decomposition_custom(df["fastestFee"], model="additive", period=288, title="Daily Seasonality")
    """
    series = series.copy()
    series.name = ""
    
    result = seasonal_decompose(series, model=model, period=period)
    fig = result.plot()
    fig.set_size_inches(figsize)

    # Format x-axis (weekly ticks, 'Jun 01' format)
    for ax in fig.axes:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

    # Label first and last subplot
    fig.axes[0].set_ylabel("Distribution")
    fig.axes[-1].set_xlabel("Date")
    fig.suptitle(title, y=1.02)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_group_correlation(
        df: pd.DataFrame,
        group_name: str,
        prefix: str,
        exclude_prefixes: list[str] | None = None,
        label_map: dict[str, str] | None = None,
        save_path: str | None = None):
    """
    Draws a Pearson-correlation heatmap for all columns whose names start
    with *prefix* (after optional exclusions).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing all candidate features.
    group_name : str
        Label for the feature-group, used in the plot title.
    prefix : str
        String prefix to select columns, e.g. ``"mempool_"``.
    exclude_prefixes : list[str], optional
        List of prefixes to leave out (e.g. histogram vectors).
    label_map : dict[str, str], optional
        Maps raw column names to human-readable labels.
        Anything *not* in this dict will be auto-formatted.
    save_path : str, optional
        If given, figure is saved to this path (dpi=300); otherwise only shown.

    Returns
    -------
    None
        Displays (and optionally saves) the heatmap.

    Notes
    -----
    * Auto-label fallback removes the common *prefix* and converts
      ``mempool_blocks_blockVSize`` → ``Block VSize``.
    * Correlation uses pair-wise complete observations (Pandas ``.corr()``).

    Examples
    --------
    >>> label_map = {"mempool_blocks_nTx": "Next-Block #Tx"}
    >>> plot_group_correlation(
    ...     df, "mempool_blocks", "mempool_blocks_",
    ...     exclude_prefixes=["mempool_blocks_hist_"],
    ...     label_map=label_map,
    ...     save_path="../img/mempool_blocks_corr.png")
    """
    
    cols = [c for c in df.columns if c.startswith(prefix)]
    if exclude_prefixes:
        cols = [c for c in cols if not any(c.startswith(ex) for ex in exclude_prefixes)]
    if len(cols) < 2:
        print(f"Skipping '{group_name}': <2 usable columns.")
        return

    corr = df[cols].dropna().corr()

    # generate readable name
    def pretty(col: str) -> str:
        if label_map and col in label_map:
            return label_map[col]
        clean = col[len(prefix):]          # strip common prefix
        clean = clean.replace("_", " ")    # underscores → space
        return clean.title()               # capitalise words

    corr = corr.rename(index=pretty, columns=pretty)

    # plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f",
                cmap="coolwarm", center=0, square=True,
                cbar_kws=dict(shrink=.8))
    nice_title = group_name.replace("_", " ").title()
    plt.title(f"{nice_title} Feature Correlation")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
