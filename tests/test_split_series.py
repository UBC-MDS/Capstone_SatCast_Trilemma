
import numpy as np
import pandas as pd

import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "src" 
sys.path.insert(0, str(src_path))
from split_series import split_series


def test_split_series_success():
    df = pd.DataFrame({"time_idx": list(range(100))})
    df_train, df_valid = split_series(df, PRED_STEPS=10)
    assert df_train["time_idx"].max() == 89
    assert df_valid["time_idx"].max() == 99

def test_split_series_edge_zero_pred():
    df = pd.DataFrame({"time_idx": list(range(10))})
    df_train, df_valid = split_series(df, PRED_STEPS=0)
    assert df_train.equals(df)
    assert df_valid.equals(df)

def test_split_series_minimal_input():
    df = pd.DataFrame({"time_idx": [0]})
    df_train, df_valid = split_series(df, PRED_STEPS=1)
    assert df_train.empty
    assert df_valid.shape[0] == 1