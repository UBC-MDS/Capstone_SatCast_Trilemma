# prophet_data_preprocess.py
# author: Tengwei Wang
# date: 2025-06-18

# Preprocess data for prophet model. 
 
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))
from preprocess_raw_parquet import preprocess_raw_parquet
import numpy as np

def data_preprocess(df):
    """
    Preprocess the dataset for optimizing the model. 

    Parameters:
    ----------
    df : str
        The path of training dataset. 

    Returns:
    -------
    pd.DataFrame
        Processed data.
    """
    df_new = preprocess_raw_parquet(df)
    df_new.dropna(inplace = True)

    df_new = df_new.iloc[:-96]
    y_new = df_new["recommended_fee_fastestFee"]
    X_new = df_new.drop(columns = "recommended_fee_fastestFee")
    X_new = X_new.reset_index()
    X_new = X_new.drop(columns = "timestamp")

    # last 24h as test
    split_index = len(X_new) - 96

    X_train_new, X_test_new = X_new.iloc[:split_index], X_new.iloc[split_index:]
    y_train_new, y_test_new = y_new.iloc[:split_index], y_new.iloc[split_index:]

    df_prophet_new = y_train_new.reset_index()
    df_prophet_new = df_prophet_new.rename(columns={
        'timestamp': 'ds',
        'recommended_fee_fastestFee': 'y'
    })
    df_prophet_new['y'] = np.log1p(df_prophet_new['y'])
    
    return df_prophet_new,y_train_new
