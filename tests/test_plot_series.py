import numpy as np
import pandas as pd

import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "src" 
sys.path.insert(0, str(src_path))
from plot_series import plot_series
import pytest
from datetime import datetime


def test_plot_series_success():
    df = pd.DataFrame({
        "series_id": ["A"] * 5,
        "timestamp": pd.date_range("2025-01-01", periods=5, freq="15min"),
        "y_true": np.arange(5),
        "y_pred": np.arange(5) + 1
    })
    ax = plot_series(df, sid="A")
    assert ax.get_title().startswith("Series A")

def test_plot_series_fail_missing_cols():
    df = pd.DataFrame({
        "series_id": ["A"],
        "timestamp": [datetime.now()]
    })
    with pytest.raises(ValueError, match="missing"):
        plot_series(df, sid="A")

def test_plot_series_fail_unknown_sid():
    df = pd.DataFrame({
        "series_id": ["B"],
        "timestamp": [datetime.now()],
        "y_true": [10],
        "y_pred": [12]
    })
    with pytest.raises(ValueError, match="No data found"):
        plot_series(df, sid="A")