"""
test_custom_loss_eval.py

Unit tests for custom_loss_eval.py
"""

import numpy as np
import pandas as pd

import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "src" 
sys.path.insert(0, str(src_path))
from custom_loss_eval import eval_metrics,custom_loss_eval


# Tests
def test_eval_metrics_success():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1.1, 2.1, 3.1, 4.1])
    result = eval_metrics(y_pred, y_true)
    assert isinstance(result, pd.DataFrame)
    assert set(result.index) == {'custom_loss', 'std_diff', 'dev_error', 'mae', 'mape', 'rmse'}

def test_eval_metrics_fail_length_mismatch():
    y_true = np.array([1, 2, 3])
    y_pred = np.array([1, 2, 3, 4])
    try:
        eval_metrics(y_pred, y_true)
        assert False, "Expected ValueError due to length mismatch"
    except ValueError:
        assert True

def test_eval_metrics_edge_all_zeros():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    result = eval_metrics(y_pred, y_true)
    assert np.isfinite(result.loc["custom_loss"].value)


def test_custom_loss_eval_success():
    y_true = np.array([1, 2, 3, 4])
    y_pred = np.array([1.1, 1.9, 3.0, 4.2])
    result = custom_loss_eval(y_pred, y_true)
    assert isinstance(result, float)
    assert result > 0

def test_custom_loss_eval_fail_shape():
    try:
        custom_loss_eval(np.array([1, 2]), np.array([1, 2, 3]))
        assert False, "Expected broadcasting error due to shape mismatch"
    except ValueError:
        assert True

def test_custom_loss_eval_edge_constant():
    y_true = np.array([2, 2, 2, 2])
    y_pred = np.array([2, 2, 2, 2])
    result = custom_loss_eval(y_pred, y_true)
    assert np.isclose(result, 0.0)


