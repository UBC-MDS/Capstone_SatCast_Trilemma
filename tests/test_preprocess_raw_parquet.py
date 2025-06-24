"""
test_preprocess_raw_parquet.py

Unit tests for preprocess_raw_parquet.py
"""


import pytest
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "src" 
sys.path.insert(0, str(src_path))
from preprocess_raw_parquet import preprocess_raw_parquet
file_path = project_root / "data" / "raw" / "mar_5_may_12.parquet"
from unittest.mock import patch

def test_preprocess_raw_parquet_file_not_exist():
    """
    Test that preprocess_raw_parquet raises FileNotFoundError when the provided file path does not exist.
    Confirms that missing file errors are surfaced clearly and early.
    """
    with pytest.raises(FileNotFoundError):
        preprocess_raw_parquet("a/a.parquet")

def test_preprocess_raw_parquet_success():
    """
    Test that preprocess_raw_parquet successfully reads a valid parquet file and returns a non-empty DataFrame.
    Verifies the core functionality of data loading and initial validation.
    """
    df = preprocess_raw_parquet(file_path)
    assert df.shape[0] != 0


def test_preprocess_raw_parquet_read_failure():
    """
    Test that preprocess_raw_parquet raises a ValueError when parquet read operation fails due to I/O or format issues.
    Uses mocking to simulate a failure in pandas.read_parquet, ensuring robust error handling.
    """
    with patch("preprocess_raw_parquet.pd.read_parquet") as mock_read:
        mock_read.side_effect = Exception("parquet read failed")
        with pytest.raises(ValueError):
            preprocess_raw_parquet(file_path)
