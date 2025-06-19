"""
compute_metrics_per_series.py

Standalone script to compute MAE, RMSE, MAPE, and custom loss per series_id
from a df_eval pandas dataframe.
"""

import pandas as pd
import numpy as np
from custom_loss_eval import custom_loss_eval

def compute_metrics(df_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Compute evaluation metrics (MAE, RMSE, MAPE, and custom loss) for each series_id.

    The custom loss combines MAE with penalties on prediction standard deviation
    and shape deviation to assess temporal stability.

    Parameters
    ----------
    df_eval : pd.DataFrame
        Evaluation results with columns ['series_id', 'y_true', 'y_pred'].

    Returns
    -------
    pd.DataFrame
        A DataFrame with one row per series_id and columns:
        ['series_id', 'MAE', 'RMSE', 'MAPE', 'Custom Loss (MAE+STD+Dev)'],
        sorted by ascending custom loss.

    Example
    -------
    >>> df_metrics = compute_metrics(df_eval)
    >>> df_metrics.head()
    """

    metrics_per_sid = (
        df_eval.groupby("series_id")
        .apply(
            lambda g: pd.Series({
                "MAE": np.abs(g.y_pred - g.y_true).mean(),
                "RMSE": np.sqrt(((g.y_pred - g.y_true) ** 2).mean()),
                "MAPE": np.abs((g.y_pred - g.y_true) / g.y_true).mean(),
                "Custom Loss (MAE+STD+Dev)": custom_loss_eval(
                    g.y_pred.values,
                    g.y_true.values,
                    std_weight=1.0,
                    de_weight=1.0
                ),
            })
        )
        .reset_index()
        .sort_values("Custom Loss (MAE+STD+Dev)")
    )
    return metrics_per_sid