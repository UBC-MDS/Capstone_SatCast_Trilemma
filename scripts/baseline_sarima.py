# baseline_sarima.py
# author: Yajing Liu
# date: 2025-06-18

"""
Script to train a SARIMA baseline for Bitcoin fee forecasting.

This script performs the following steps:
1. Loads raw mempool fee data and applies basic cleaning.
2. Splits the log-transformed series into training and
   24-hour evaluation windows.
3. Fits a SARIMA(1,0,1)(1,0,1,96) model on the training data.
4. Saves the trained model to
   ``results/models/sarima_final_model.pkl`` and stores the
   processed train/test splits for downstream analysis.

Usage:
    python scripts/baseline_sarima.py --parquet-path data/raw/mar_5_may_12.parquet
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
import warnings
import click
from pathlib import Path

from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import ARIMA

# Setup
warnings.filterwarnings("ignore")
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import project utilities
from src.preprocess_raw_parquet import preprocess_raw_parquet
from src.custom_loss_eval import *

# Configurations
FORECAST = 96  # 1 day = 96 steps (15min)
from pathlib import Path

# Setup project root
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]

# Directories
DATA_DIR = project_root / "data" / "processed" / "sarima"
RESULTS_DIR = project_root / "results"
MODEL_FROM = RESULTS_DIR / "models"/ "sarima_final_model.pkl"


@click.command()
@click.option(
    "--parquet-path",
    type=click.Path(exists=True),
    default="./data/raw/mar_5_may_12.parquet",
    help="Path to input Parquet file"
)
def main(parquet_path):
    """Run SARIMA pipeline for Bitcoin fee forecasting."""

    # Step 1: Load and preprocess data
    df = preprocess_raw_parquet(parquet_path)

    # Extract target series
    y = df['recommended_fee_fastestFee'].astype(float)
    y = y.iloc[:-FORECAST]  # remove spike day

    # Step 2: Train/test split
    y_train, y_test = temporal_train_test_split(y, test_size=FORECAST)

    # Save the processed and split data
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    y.to_frame(name='recommended_fee_fastestFee').to_csv(DATA_DIR / 'full_series.csv', index=True)
    y_train.to_frame(name='train').to_csv(DATA_DIR / 'train.csv', index=True)
    y_test.to_frame(name='test').to_csv(DATA_DIR / 'test.csv', index=True)


    # Step 3: Train SARIMA model on log-transformed data
    print("Fitting SARIMA model...")
    y_train_log = np.log1p(y_train)
    forecaster = ARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1, 96))
    forecaster.fit(y_train_log)

    # Step 4: Save model
    (RESULTS_DIR / 'models').mkdir(parents=True, exist_ok=True)
    with open(MODEL_FROM, 'wb') as f:
        pickle.dump(forecaster, f)


    print(f"✅ SARIMA model saved to {MODEL_FROM}")


if __name__ == '__main__':
    main()
