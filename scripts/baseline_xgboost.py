# baseline_xgboost.py
# author: Tengwei Wang
# date: 2025-06-18

"""
baseline_xgboost.py

Command-line interface (CLI) entry point to optimize and train an XGBoost forecaster.

Responsibilities:
-----------------
1. Loads and preprocesses fee data.
2. Runs randomized hyperparameter optimization.
3. Fits and saves the best-performing model to disk.

Key Features:
-------------
- CLI support via `click` for easy pipeline execution.
- Configurable forecast horizon (`interval`) and file paths.
- Modular structure supports integration into batch jobs or notebooks.

Typical Usage:
--------------
$ python baseline_xgboost.py --data-path "data/processed/fee_data.parquet" --result "results/xgb" --interval 15

This script will:
    - preprocess the data
    - optimize hyperparameters
    - train and save the best XGBoost forecaster
"""

import sys
from pathlib import Path
import click

# Set up module path for importing local project modules
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "scripts" / "xgboost"
sys.path.insert(0, str(src_path))

# Import preprocessing, optimization, and training routines
from xgboost_model_optimization import optimization
from xgboost_model_training import train_and_save_best_model
from xgboost_data_preprocess import data_preprocess

@click.command()
@click.option('--data-path', type=str, help="Path to training data")
@click.option('--result', type=str, help="Path to save model")
@click.option('--interval', type=int, default=15, help="Forecast horizon length (in steps)")

def main(data_path, result, interval):
    # Step 1: Load and preprocess dataset from provided path
    df = data_preprocess(data_path)
    
    # Step 2: Perform random search to find the best hyperparameters
    random_search, X_train, y_train = optimization(df, interval)
    
    # Step 3: Train final model using the best parameters and save it
    train_and_save_best_model(random_search, X_train, y_train, result, interval)

if __name__ == '__main__':
    main()
