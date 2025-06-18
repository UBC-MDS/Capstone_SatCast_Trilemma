"""
advanced_tft.py

Main orchestration script to run the full Temporal Fusion Transformer (TFT) forecasting pipeline
for Bitcoin transaction fee prediction.

This script performs the following steps:
1. Loads and preprocesses raw fee data from a Parquet file.
2. Constructs training and validation datasets and corresponding dataloaders.
3. Defines a custom loss function that penalizes both MAE and prediction instability (std/dev).
4. Trains a TFT model using the specified configuration and logs progress.

Usage:
    python advanced_tft.py --parquet_path <optional_path_to_parquet_file>

Dependencies:
    - The script expects supporting modules in scripts/advanced_tft/
    - Requires the Lightning-based training loop, loss functions, and dataloader constructors.
"""


import sys
import os
import argparse
from pathlib import Path

# Setup project root and import paths
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "scripts" / "tft"
sys.path.insert(0, str(src_path))

# Custom module imports
from tft_data_preparation import tft_prepare_data
from tft_create_dataloaders import tft_make_dataloaders
from tft_custom_loss import MAEWithStdPenalty
from tft_train_model import tft_train_model

# Argument parser
parser = argparse.ArgumentParser(description="Run full TFT forecasting pipeline.")
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
FALLBACK_CKPT = project_root / "analysis" / "saved_models" / "best-model-epoch=16-val_loss=0.7079.ckpt"

if __name__ == "__main__":
    # Step 1 ── Data
    df, df_train, df_valid, scaler = tft_prepare_data(PARQUET_PATH, PRED_STEPS)

    # Step 2 ── Dataloaders
    tft_ds, train_dl, val_dl = tft_make_dataloaders(df_train, df_valid, ENC_LEN, PRED_STEPS, BATCH_SIZE)

    # Step 3 ── Loss
    loss_fn = MAEWithStdPenalty(std_weight=1.0, de_weight=1.0, clip_weight_std=10.0, clip_weight_dev=10.0)

    # Step 4 ── Training
    model, trainer = tft_train_model(tft_ds, train_dl, val_dl, loss_fn)

    print("TFT Bitcoin fee forecasting pipeline completed.")
