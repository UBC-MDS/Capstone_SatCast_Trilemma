
import pandas as pd
import numpy as np
import pytest
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "src" 
sys.path.insert(0, str(src_path))
from transform_fee_data_dl import transform_fee_data_dl

@pytest.fixture
def sample_df():
    idx = pd.date_range("2025-01-01", periods=10, freq="15min")
    df = pd.DataFrame({
        "recommended_fee_fastestFee": np.random.rand(10),
        "recommended_fee_hourFee": np.random.rand(10),
        "recommended_fee_halfHourFee": np.random.rand(10),
        "recommended_fee_economyFee": np.random.rand(10),
        "recommended_fee_minimumFee": np.random.rand(10),
        "price_USD": np.random.rand(10),
    }, index=idx)
    df.index.name = "timestamp"
    return df

def test_transform_fee_data_dl_success(sample_df):
    result = transform_fee_data_dl(sample_df)
    assert "series_id" in result.columns
    assert "target" in result.columns
    assert "hour_sin" in result.columns

def test_transform_fee_data_dl_missing_column(sample_df):
    df = sample_df.drop(columns=["recommended_fee_hourFee"])
    with pytest.raises(ValueError, match="Missing required fee"):
        transform_fee_data_dl(df)

def test_transform_fee_data_dl_invalid_index(sample_df):
    df = sample_df.reset_index()
    with pytest.raises(ValueError, match="DatetimeIndex"):
        transform_fee_data_dl(df)

def test_transform_fee_data_dl_invalid_reference_time(sample_df):
    with pytest.raises(ValueError, match="Invalid reference"):
        transform_fee_data_dl(sample_df, reference_time_str="not-a-date")