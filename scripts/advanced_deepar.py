# advanced_deepar.py
# author: Ximin Xu
# date: 2025-06-18

"""
advanced_deepar.py

Script to train a DeepAR model for Bitcoin transaction-fee forecasting.

This script performs the following steps:
1. Loads raw 15-minute fee data from a Parquet file and creates train/validation splits.
2. Scales the series, serializes the processed datasets, and logs dataset statistics.
3. Builds PyTorch datasets and dataloaders in encoder/decoder format.
4. Trains the DeepAR model with validation-based early stopping and
   saves the best checkpoint to disk.

Typical Usage:
-------------
1. Run with sample data:
    python scripts/advanced_deepar.py --parquet-path data/raw/sample_8_days.parquet

2. [~6 hours] Run with full dataset:
    python scripts/advanced_deepar.py --parquet-path data/raw/mar_5_may_12.parquet
"""

import sys
from pathlib import Path
import click

# Setup project root and import paths
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "scripts" / "deepar"
sys.path.insert(0, str(src_path))

# Custom module imports
from deepar_data_preparation import deepar_prepare_data
from deepar_create_dataloaders import deepar_create_dataloaders
from deepar_train_model import train_deepar_model

# Config
ENC_LEN = 672
PRED_STEPS = 96
BATCH_SIZE = 32

@click.command()
@click.option(
    "--parquet-path",
    type=click.Path(exists=True),
    default=str(project_root / "data" / "raw" / "mar_5_may_12.parquet"),
    help="Path to the input Parquet file (default: mar_5_may_12.parquet in data/raw/)"
)
def main(parquet_path):
    parquet_path = Path(parquet_path)

    # Step 1 ── Data
    df, df_train, df_valid, scaler = deepar_prepare_data(parquet_path, PRED_STEPS)

    # Save processed data to deepar folder
    processed_dir = project_root / "data" / "processed" / "deepar"
    processed_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(processed_dir / "df_full.csv", index=False)
    df_train.to_csv(processed_dir / "df_train.csv", index=False)
    df_valid.to_csv(processed_dir / "df_valid.csv", index=False)

    # Step 2 ── Dataloaders
    deepar_ds, train_dl, val_dl = deepar_create_dataloaders(df_train, df_valid, ENC_LEN, PRED_STEPS, BATCH_SIZE)

    # Step 3 ── Training and save best checkpoint
    train_deepar_model(deepar_ds, train_dl, val_dl)

    print("DeepAR Bitcoin fee forecasting pipeline completed.")

if __name__ == "__main__":
    main()
