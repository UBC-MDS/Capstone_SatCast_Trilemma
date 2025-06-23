# xgboost_window.py
# author: Ximin Xu
# date: 2025-06-18
"""
Runs XGBoost back-tests on Bitcoin-fee data using **weekly windowing
schemes**, matching `sarima_window.py` and `hwes_window.py` in interface.

Modes:
------
1. **Reverse expanding window** – growing training windows that move backwards from the final test day.
2. **Weekly expanding window** – walk-forward: growing train window, predict next day (96 steps).
3. **Weekly sliding window** – fixed-size train window moves 1 week forward per fold.

Workflow
--------
1. Load preprocessed 15-min resampled Parquet.
2. Generate folds based on mode.
3. For each fold:
   a. Build supervised lag features
   b. Apply best hyperparameters from saved JSON
   c. Predict next 96 steps
   d. Evaluate MAE, RMSE, MAPE, custom loss
4. Save fold-wise metrics to:
   - expanding_window_reverse_weekly_predictions.csv
   - expanding_window_weekly_predictions.csv
   - sliding_window_weekly_predictions.csv

Example usage:
1. expanding window
python scripts/experimentation/xgboost_window.py \
  --parquet-path ./data/raw/mar_5_may_12.parquet \
  --mode expanding

2. sliding window
python scripts/experimentation/xgboost_window.py \
  --parquet-path ./data/raw/mar_5_may_12.parquet \
  --mode sliding

3. reverse expanding window
python scripts/experimentation/xgboost_window.py \
  --parquet-path ./data/raw/mar_5_may_12.parquet \
  --mode reverse
"""

import sys
import json
import click
import numpy as np
import pandas as pd
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.base import clone
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter
from sktime.forecasting.compose import make_reduction

# === Paths and Imports ===
project_root = Path(__file__).resolve().parent.parent.parent

# Add module paths
sys.path.append(str(project_root / "scripts" / "xgboost"))
from xgboost_data_preprocess import data_preprocess  

sys.path.append(str(project_root / "src"))
from custom_loss_eval import eval_metrics             

# Constants
RESULTS_DIR = project_root / "results" / "tables" / "xgboost"
BEST_PARAMS_PATH = project_root / "results" / "models" / "xgb_best_params_cv.json"
FORECAST = 96                    # 96 steps = 1 day @ 15min
WEEKLY = FORECAST * 7            # 672 steps = 1 week

# === Fold Generation Function ===
def get_folds(y: pd.Series, mode: str):
    """
    Generate training/testing folds based on selected windowing mode.

    Parameters
    ----------
    y : pd.Series
        Target time series.
    mode : str
        One of ['reverse', 'expanding', 'sliding'].

    Returns
    -------
    list of (train_idx, test_idx) tuples
    """
    fh = list(range(1, FORECAST + 1))  # relative horizon (next 96 steps)
    n_obs = len(y)

    if mode == "reverse":
        # Reverse expanding: always test on final 96 points, grow training set backwards
        test_end = n_obs
        test_start = test_end - FORECAST
        folds = []

        for i in range(1, (test_start // WEEKLY) + 1):
            train_start = test_start - i * WEEKLY
            train_idx = list(range(train_start, test_start))
            test_idx = list(range(test_start, test_end))
            folds.append((train_idx, test_idx))

        return folds

    elif mode == "expanding":
        splitter = ExpandingWindowSplitter(
            initial_window=WEEKLY,
            step_length=WEEKLY,
            fh=fh,
        )
    elif mode == "sliding":
        splitter = SlidingWindowSplitter(
            window_length=WEEKLY,
            step_length=WEEKLY,
            fh=fh,
        )
    else:
        raise ValueError("Mode must be one of: reverse, expanding, sliding")

    # Return list of train-test index pairs
    return [(train.tolist(), test.tolist()) for train, test in splitter.split(y)]


# === Run Backtests ===
def run_xgboost_fixed(df: pd.DataFrame, folds, results_path: Path):
    """
    Run XGBoost forecasting across folds using best hyperparameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with target and features.
    folds : list of (train_idx, test_idx)
        Precomputed CV folds.
    results_path : Path
        Output path for saving results.
    """
    # Load best hyperparameters (from random search or tuning)
    print(f"🔧 Loading best hyperparameters from: {BEST_PARAMS_PATH}")
    with open(BEST_PARAMS_PATH) as f:
        raw_params = json.load(f)
    best_params = {k.replace("estimator__", ""): v for k, v in raw_params.items()}
    
    print("Best parameters loaded:", best_params)

    # Extract target and features
    y = df["recommended_fee_fastestFee"]
    X = df.drop(columns=["recommended_fee_fastestFee"])
    fh = ForecastingHorizon(np.arange(1, FORECAST + 1), is_relative=True)

    # Create base recursive forecaster
    base_model = XGBRegressor(**best_params)
    forecaster = make_reduction(base_model, window_length=FORECAST, strategy="recursive")

    results = []
    print(f"Running backtests on {len(folds)} folds...")

    # Loop over all folds
    for i, (train_idx, test_idx) in enumerate(folds, 1):
        print(f"\nFold {i} | Train range: {train_idx[0]}–{train_idx[-1]} | Test range: {test_idx[0]}–{test_idx[-1]}")

        # Slice train and test sets
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        # Clone forecaster and fit
        model = clone(forecaster)
        model.fit(y_train, X=X_train)

        print("Predicting...")
        y_pred = model.predict(fh=fh, X=X_test)

        # Evaluate fold
        metrics = eval_metrics(y_pred.values, y_test.values).T
        metrics["fold"] = i
        results.append(metrics)

    # Concatenate and save metrics
    metrics_df = pd.concat(results)
    metrics_df = metrics_df.set_index("fold")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(results_path)
    print(f"\nSaved fold metrics → {results_path.resolve()}")


# === CLI Interface ===
@click.command()
@click.option("--parquet-path", type=str, required=True, help="Path to preprocessed Parquet")
@click.option("--mode", type=click.Choice(["reverse", "expanding", "sliding"]), required=True)
def main(parquet_path, mode):
    """
    Main CLI entrypoint.
    """
    # Preprocess and get folds
    df = data_preprocess(parquet_path)
    y = df["recommended_fee_fastestFee"][:-96]
    folds = get_folds(y, mode)

    # Choose output filename by mode
    filename_map = {
        "reverse": "expanding_window_reverse_weekly_predictions.csv",
        "expanding": "expanding_window_weekly_predictions.csv",
        "sliding": "sliding_window_weekly_predictions.csv",
    }
    results_path = RESULTS_DIR / filename_map[mode]

    # Run evaluation
    run_xgboost_fixed(df, folds, results_path)


# === Script Entrypoint ===
if __name__ == "__main__":
    main()