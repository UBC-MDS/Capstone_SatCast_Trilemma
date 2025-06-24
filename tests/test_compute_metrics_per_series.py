"""
test_compute_metrics_per_series.py

Unit tests for compute_metrics_per_series.py
"""


import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "src" 
sys.path.insert(0, str(src_path))
from compute_metrics_per_series import compute_metrics


def test_compute_metrics_success():
    """
    Test that compute_metrics successfully computes metrics
    when given a well-formed input DataFrame with multiple series.
    Verifies that the function returns a DataFrame and includes expected columns.
    """
    df_eval = pd.DataFrame({
        "series_id": ["s1"] * 4 + ["s2"] * 4,
        "y_true": [1, 2, 3, 4, 2, 4, 6, 8],
        "y_pred": [1.1, 2.1, 2.9, 4.0, 2.1, 3.9, 6.1, 7.8]
    })
    result = compute_metrics(df_eval)
    assert isinstance(result, pd.DataFrame)
    assert "MAE" in result.columns

def test_compute_metrics_fail_missing_column():
    """
    Test that compute_metrics raises an AttributeError when 'y_pred' column is missing.
    This ensures the function properly validates required inputs if attribute-style access is used.
    """
    df_eval = pd.DataFrame({
        "series_id": ["s1"] * 4,
        "y_true": [1, 2, 3, 4]
    })
    with pytest.raises(AttributeError, match="y_pred"):
        compute_metrics(df_eval)

def test_compute_metrics_edge_single_point():
    """
    Test compute_metrics behavior on a minimal edge case with a single data point.
    Ensures that metrics like MAE return correctly (should be zero when prediction is perfect).
    """
    df_eval = pd.DataFrame({
        "series_id": ["s1"],
        "y_true": [1],
        "y_pred": [1]
    })
    result = compute_metrics(df_eval)
    assert result.shape[0] == 1
    assert result["MAE"].iloc[0] == 0.0