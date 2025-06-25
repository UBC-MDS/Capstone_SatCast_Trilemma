# baseline_xgboost.py
# author: Tengwei Wang
# date: 2025-06-18

"""
Script to build an XGBoost baseline (with exogenous features) for
Bitcoin fee forecasting.

This script performs the following steps:
1. Loads raw fee data, engineers lags/covariates, and
   writes the processed dataset to ``data/processed/xgboost``.
2. Samples hyperparameter sets with random search and trains
   an XGBoost regressor on each fold (expanding or sliding window).
3. Selects the best configuration based on custom loss and MAE.
4. Fits the best model on the entire training window and
   saves it to ``results/models/xgboost.pkl``.

Usage:
    python scripts/baseline_xgboost.py \
      --parquet-path data/raw/mar_5_may_12.parquet \
         [--skip-optimization]
"""

import sys
from pathlib import Path
import click
import joblib
import json

# Set up module path for importing local project modules
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "scripts" / "xgboost"
sys.path.insert(0, str(src_path))

# Import preprocessing, optimization, and training routines
from xgboost_model_optimization import optimization
from xgboost_model_training import build_random_search_cv
from xgboost_data_preprocess import data_preprocess

@click.command()
@click.option(
    "--parquet-path",
    type=click.Path(exists=True),
    default="./data/raw/mar_5_may_12.parquet",
    help="Path to input Parquet file"
)
@click.option(
    '--skip-optimization',
    is_flag=True,
    default=False,
    help="Skip hyperparameter tuning and use existing config."
)
def main(parquet_path, skip_optimization):
    interval = 15

    # Step 1: Load and preprocess dataset
    print("Loading and preprocessing data...")
    df = data_preprocess(parquet_path)

    # Save processed data
    processed_dir = project_root / "data" / "processed" / "xgboost"
    processed_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(processed_dir / "df_processed.csv", index=False)

    # Step 2: Generate hyperparameter search space + train/test split
    print("Preparing training data and search space...")
    random_search, X_train, y_train = optimization(df, interval)
    X_train.to_csv(processed_dir / "X_train.csv", index=False)
    y_train.to_frame(name="y").to_csv(processed_dir / "y_train.csv", index=False)

    # Step 3: Train model (with or without tuning)
    print("Training model (skip optimization =", skip_optimization, ")...")
    result_dir = project_root / "results" / "models"
    result_dir.mkdir(parents=True, exist_ok=True)
    best_model, best_params, best_score = build_random_search_cv(
      X_train, y_train,
      param_dist=random_search,
      n_iter=8,
      n_folds=5,               # number of CV folds
      horizon=96,              # forecast horizon (e.g., 24 hours if 15-min steps)
      optimize=not skip_optimization
   )
    # Step 4: Save model
    file_path = result_dir / "xgboost.pkl"
    joblib.dump(best_model, file_path)
    print("Best model saved to:", file_path)


if __name__ == '__main__':
    main()
