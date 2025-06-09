"""
advanced_tft.py

Main orchestration script to run full TFT forecasting pipeline.
"""

import sys
import os
import argparse
from pathlib import Path

# Setup project root and import paths
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "scripts" / "deepar"
sys.path.insert(0, str(src_path))

# Custom module imports
from deepar_data_preparation import deepar_prepare_data
from deepar_create_dataloaders import deepar_create_dataloaders
from deepar_train_model import train_deepar_model

# Argument parser
parser = argparse.ArgumentParser(description="Run full deepAR forecasting pipeline.")
parser.add_argument(
    "--parquet_path",
    type=str,
    default=str(project_root / "data" / "raw" / "mar_5_may_12.parquet"),
    help="Path to the input parquet file (default: mar_5_may_12.parquet in data/raw/)"
)
args = parser.parse_args()

# Config
PARQUET_PATH = Path(args.parquet_path)
ENC_LEN = 672
PRED_STEPS = 96
BATCH_SIZE = 32

if __name__ == "__main__":
    # Step 1 ── Data
    df, df_train, df_valid, scaler = deepar_prepare_data(PARQUET_PATH, PRED_STEPS)

    # Step 2 ── Dataloaders
    deepar_ds, train_dl, val_dl = deepar_create_dataloaders(df_train, df_valid, ENC_LEN, PRED_STEPS, BATCH_SIZE)

    # Step 3 ── Training
    train_deepar_model(deepar_ds, train_dl, val_dl)

    print("deepar Bitcoin fee forecasting pipeline completed.")
