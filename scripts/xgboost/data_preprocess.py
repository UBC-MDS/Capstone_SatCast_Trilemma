import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from preprocess_raw_parquet import preprocess_raw_parquet
from XGBoost import create_lag_features_fast
import numpy as np

def data_preprocess(data_path):
    """
    Preprocess the dataset for optimizing the model. 

    Parameters:
    ----------
    data_path : str
        The path of training dataset. 

    Returns:
    -------
    pd.DataFrame
        Processed data.
    """
    df = preprocess_raw_parquet(data_path)
    df.dropna(inplace = True)
    lags = range(1, 193)  # 48 hours of 15-minute intervals
    df = create_lag_features_fast(df, 'recommended_fee_fastestFee', lags)
    return df
