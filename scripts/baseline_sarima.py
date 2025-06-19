"""
baseline_sarima.py

This script performs the full SARIMA pipeline including:
1. Loading and preprocessing raw mempool data
2. Training/testing data split
3. Model training with SARIMA (using log-transformed values)
4. Forecasting and inverse transformation
5. Saving predictions and evaluation metrics

The trained model is saved as a pickle file, and all intermediate results are written to CSV.

Usage:
    python scripts/baseline_sarima.py
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import warnings

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Setup
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import project utilities
from src.preprocess_raw_parquet import preprocess_raw_parquet
from src.save_csv_data import save_csv_data
from src.read_csv_data import read_csv_data
from src.save_model import save_model
from src.custom_loss_eval import *

# Configurations
FORECAST = 96  # 1 day = 96 steps (15min)
INPUT_DATA = "./data/raw/mar_5_may_12.parquet"
DATA_DIR = './data/processed/sarima'
RESULTS_DIR = './results'
MODEL_FROM = './results/models/sarima_final_model.pkl'

if __name__ == '__main__':

    ## ---------Step 1: Load and preprocess data---------
    df = preprocess_raw_parquet(INPUT_DATA)

    # Extract target series
    y = df['recommended_fee_fastestFee']
    y = y.astype(float)
    y = y.iloc[:-96]  # remove spike day

    ## ---------Step 2: Train/test split---------
    y_train, y_test = temporal_train_test_split(y, test_size=FORECAST)

    # Save the processed and split data
    os.makedirs(DATA_DIR, exist_ok=True)
    save_csv_data(y.to_frame(name='recommended_fee_fastestFee'), os.path.join(DATA_DIR, 'full_series.csv'), index=True)
    save_csv_data(y_train.to_frame(name='train'), os.path.join(DATA_DIR, 'train.csv'), index=True)
    save_csv_data(y_test.to_frame(name='test'), os.path.join(DATA_DIR, 'test.csv'), index=True)

    ## ---------Step 3: Train SARIMA model---------
    y_train_log = np.log1p(y_train)
    forecaster = ARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1, 96))
    forecaster.fit(y_train_log)

    # Save the final trained model
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)

    save_model(forecaster, MODEL_FROM)
    print(f"✅ SARIMA model saved to {MODEL_FROM}")

    ## ---------Step 4: Forecast---------
    fh = list(range(1, FORECAST + 1))
    y_pred_log = forecaster.predict(fh=fh)

    # Convert to pandas series, expm1, assign timestamp
    y_pred = np.expm1(y_pred_log)

    forecast_df = pd.DataFrame({'timestamp': y_test.index, 'forecast': y_pred.values})

    save_csv_data(forecast_df, os.path.join(RESULTS_DIR, 'tables', 'sarima_forecast.csv'), index=True)
    print(f"✅ SARIMA forecast saved to sarima_forecast.csv")



    ## ---------Step 5: Evaluate---------
    eval_results = eval_metrics(y_pred, y_test)
    save_csv_data(eval_results, os.path.join(RESULTS_DIR, 'tables', 'sarima_eval_results.csv'), index=True)
    print(f"✅ SARIMA evaluation saved to sarima_eval_results.csv")
