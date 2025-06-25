# advanced_prophet.py
# author: Tengwei Wang
# date: 2025-06-18

"""
advanced_prophet.py

Script to preprocess data, fine-tune, and train a Prophet model
for Bitcoin fee forecasting.

This script performs the following steps:
1. Converts raw Parquet data into the Prophet-friendly (ds, y) format
   and stores the processed file for reproducibility.
2. (Optional) Runs grid search cross-validation to optimise key Prophet
   hyperparameters; skip with --skip-optimization.
3. Fits the best (or existing) configuration on the full training set.
4. Saves the trained model to JSON so it can be re-loaded with
   ``prophet.serialize.model_from_json``.

Typical Usage:
-------------
1. Skip optimization (use saved config):
    python scripts/advanced_prophet.py \
        --parquet-path data/raw/mar_5_may_12.parquet \
        --skip-optimization

2. [~3-4 hours] Full Optimization (train from scratch):
    python scripts/advanced_prophet.py \
        --parquet-path data/raw/mar_5_may_12.parquet
"""


import sys
from pathlib import Path
import click
from prophet.serialize import model_to_json

# Setup paths
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "scripts" / "prophet"
sys.path.insert(0, str(src_path))

# Custom imports
from prophet_model_optimization import model_optimization
from prophet_data_preprocess import data_preprocess
from prophet_model_training import prophet_model_training

@click.command()
@click.option(
    '--parquet-path', 
    type=click.Path(exists=True), 
    default=str(project_root / "data" / "raw" / "mar_5_may_12.parquet"),
    help="Path to training Parquet file"
)
@click.option(
    '--skip-optimization',
    is_flag=True,
    default=False,
    help="Skip hyperparameter tuning and use existing config."
)
def main(parquet_path, skip_optimization):
    # Step 1: Preprocess raw data from parquet
    df_processed, y_train = data_preprocess(parquet_path)

    # Save processed inputs for reproducibility
    output_dir = project_root / "data" / "processed" / "prophet"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(output_dir / "df_processed.csv", index=False)
    y_train.to_frame(name="y").to_csv(output_dir / "y_train.csv", index=False)

    result_dir = project_root / "results" / "models"
    result_dir.mkdir(parents=True, exist_ok=True)

    # Step 2: Hyperparameter optimization
    if not skip_optimization:
        model_optimization(df_processed, y_train, str(result_dir))
    else:
        print("Skipping hyperparameter optimization. Ensure a config already exists in the result directory.")

    # Step 3: Train final model
    model = prophet_model_training(df_processed, y_train, str(result_dir))

    # Step 4: Save the model to disk
    model_path = result_dir / "prophet_model.json"
    with open(model_path, 'w') as fout:
        fout.write(model_to_json(model))

    print(f"Prophet model saved to: {model_path}")

if __name__ == '__main__':
    main()
