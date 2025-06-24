# advanced_tft.py
# author: Ximin Xu
# date: 2025-06-18

"""
advanced_tft.py

Script to train a Temporal Fusion Transformer (TFT) for
multi-horizon Bitcoin fee forecasting.

This script performs the following steps:
1. Converts raw fee data plus exogenous features into a
   PyTorch-friendly dataset and persists the splits.
2. Builds dataloaders with the specified encoder length,
   prediction horizon, and batch size.
3. Defines a custom MAE + volatility penalty loss.
4. Trains the TFT model with early stopping and learning-rate
   scheduling.
5. Saves the final model checkpoint to ``results/models``.

Usage:
    python scripts/advanced_tft.py --parquet-path data/raw/mar_5_may_12.parquet
"""


import sys
from pathlib import Path
import torch
import click

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

# Constants
ENC_LEN = 672
PRED_STEPS = 96
BATCH_SIZE = 32

@click.command()
@click.option(
    "--parquet-path", 
    type=click.Path(exists=True), 
    default=str(project_root / "data" / "raw" / "mar_5_may_12.parquet"), 
    help="Path to input parquet file."
)
def main(parquet_path):
    """Run full TFT pipeline on Bitcoin fee data."""
    
    # Step 1 ── Data
    df, df_train, df_valid, scaler = tft_prepare_data(Path(parquet_path), PRED_STEPS)

    # Save processed data
    processed_dir = project_root / "data" / "processed" / "tft"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "df_full.csv", index=False)
    df_train.to_csv(processed_dir / "df_train.csv", index=False)
    df_valid.to_csv(processed_dir / "df_valid.csv", index=False)

    # Step 2 ── Dataloaders
    tft_ds, train_dl, val_dl = tft_make_dataloaders(df_train, df_valid, ENC_LEN, PRED_STEPS, BATCH_SIZE)

    # Step 3 ── Loss
    loss_fn = MAEWithStdPenalty(
        std_weight=1.0, 
        de_weight=1.0, 
        clip_weight_std=10.0, 
        clip_weight_dev=10.0
    )

    # Step 4 ── Training
    model, trainer = tft_train_model(tft_ds, train_dl, val_dl, loss_fn)

    # Step 5 ── Save model
    model_save_dir = project_root / "results" / "models"/ "temp_models"
    model_save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model, model_save_dir / "best-model-tft-full.pt")
    print(f"✅ Full model saved at: {model_save_dir / 'best-model-tft-full.pt'}")

    print("✅ TFT Bitcoin fee forecasting pipeline completed.")

if __name__ == '__main__':
    main()
