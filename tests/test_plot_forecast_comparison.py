"""
test_plot_forecast_comparison.py

Unit tests for plot_forecast_comparison.py
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "src" 
sys.path.insert(0, str(src_path))
from plot_forecast_comparison import plot_forecast_comparison
from matplotlib.figure import Figure

def test_plot_forecast_comparison_success():
    """
    Test that plot_forecast_comparison successfully generates a Figure object
    when given valid input DataFrames for two models with matching series_id.
    Ensures normal functionality of the visualization under expected use.
    """
    df_common = pd.DataFrame({
        "series_id": ["recommended_fee_fastestFee"] * 4,
        "timestamp": pd.date_range("2023-01-01", periods=4, freq="h"),
        "y_true": [10, 20, 30, 40],
        "y_pred": [11, 19, 31, 39]
    })
    fig, ax = plot_forecast_comparison(df_common, "Model1", df_common, "Model2")
    assert isinstance(fig, Figure)

def test_plot_forecast_comparison_fail_missing_series():
    """
    Test that plot_forecast_comparison raises a ValueError when the specified
    series_id does not exist in either of the input DataFrames.
    Validates error handling for missing or mismatched series.
    """
    df1 = pd.DataFrame({
        "series_id": ["a"],
        "timestamp": [pd.Timestamp("2023-01-01")],
        "y_true": [1],
        "y_pred": [1]
    })
    df2 = pd.DataFrame({
        "series_id": ["b"],
        "timestamp": [pd.Timestamp("2023-01-01")],
        "y_true": [1],
        "y_pred": [1]
    })
    try:
        plot_forecast_comparison(df1, "Model1", df2, "Model2", sid="recommended_fee_fastestFee")
        assert False, "Expected ValueError due to missing series"
    except ValueError:
        assert True

def test_plot_forecast_comparison_edge_one_point():
    """
    Test that plot_forecast_comparison works when the input DataFrames contain only one timestamp.
    Ensures the function is robust to edge cases with minimal data.
    """
    df = pd.DataFrame({
        "series_id": ["recommended_fee_fastestFee"],
        "timestamp": [pd.Timestamp("2023-01-01 00:00:00")],
        "y_true": [100],
        "y_pred": [105]
    })
    fig, ax = plot_forecast_comparison(df, "M1", df, "M2")
    assert isinstance(fig, Figure)