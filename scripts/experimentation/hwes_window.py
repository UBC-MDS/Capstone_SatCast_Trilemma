# hwes_window.py
# author: Jenny Zhang
# date: 2025-06-18
"""
hwes_window.py

Runs Holt-Winters Exponential Smoothing (HWES) back-tests on Bitcoin-fee data 
using **custom-defined weekly windowing schemes**, since HWES does not natively 
support expanding/sliding windows.

Modes:
------
1. **Reverse expanding window** – fixes the *last* 24 hours (96 steps) as test set
   and expands training data backward in 7-day blocks, moving one week earlier 
   in each fold.

2. **Weekly expanding window** – standard walk-forward CV: starts with one week of data,
   adds a week in each fold, and always predicts the next day (96 steps).

3. **Weekly sliding window** – rolls a fixed 7-day training window forward by 1 week 
   per fold, always predicting the next day.

Workflow
--------
1. Load the 15-minute-resampled Parquet file created in the preprocessing phase.
2. Define window folds based on the selected mode.
3. For each fold:
   a. Fit a Holt-Winters model using `ExponentialSmoothing`  
   b. Predict the next 96 time steps (1 day)  
   c. Score against the test set using `eval_metrics()` → MAE, RMSE, MAPE, custom loss, etc.
4. Aggregate fold-wise metrics into a tidy DataFrame (`fold` as index).
5. Write results to a CSV at table folder

Key Features
------------
- **Daily horizon** = 96 × 15-minute intervals (1 day).
- Uses `statsmodels`' ExponentialSmoothing (supports additive/multiplicative trend/seasonality).
- Fold generation is handled manually since statsmodels lacks native splitters.
- Automatically saves one of:
  - `expanding_window_reverse_weekly_predictions.csv`  
  - `expanding_window_weekly_predictions.csv`  
  - `sliding_window_weekly_predictions.csv`

Typical Usage
-------------
1. Reverse expanding:
python scripts/experimentation/hwes_window.py \
  --data ./data/raw/mar_5_may_12.parquet \
  --mode reverse

2. Weekly expanding: 
python scripts/experimentation/hwes_window.py \
  --data ./data/raw/mar_5_may_12.parquet \
  --mode expanding

3. Weekly sliding:
python scripts/experimentation/hwes_window.py \
  --data ./data/raw/mar_5_may_12.parquet \
  --mode sliding
"""
  
import os
import sys
import click
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Suppress warnings to keep output clean
warnings.filterwarnings("ignore")

# Set project root path and append source directory to sys.path for imports
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root / "src"))

# Import helper functions
from preprocess_raw_parquet import preprocess_raw_parquet
from custom_loss_eval import eval_metrics

# Constants
DAILY = 96  # 24 hours × 4 (15-min intervals) = 96 steps
WEEK = 96 * 7  # 1 week of 15-min intervals
RESULTS_DIR = project_root / "results"  # Root for saving results


def get_folds(y, mode):
    """
    Generate train/test splits based on windowing strategy.

    Parameters:
    -----------
    y : pd.Series - full time series
    mode : str - one of ['reverse', 'expanding', 'sliding']

    Returns:
    --------
    list of (train_idx, test_idx) tuples
    """

    if mode == "reverse":
        # Fixed final day as test set, training expands backwards
        test_end = len(y)
        test_start = test_end - DAILY
        train_end = test_start
        n_folds = train_end // WEEK  # number of full weeks
        folds = []
        for i in range(1, n_folds + 1):
            train_start = max(0, train_end - i * WEEK)
            folds.append((list(range(train_start, train_end)), list(range(test_start, test_end))))
        return folds

    elif mode == "expanding":
        # Train grows weekly, test always the next day
        folds = []
        for i in range(1, (len(y) - WEEK - DAILY) // WEEK + 1):
            train_end = WEEK + (i - 1) * WEEK
            train_idx = list(range(train_end))
            test_idx = list(range(train_end, train_end + DAILY))
            folds.append((train_idx, test_idx))
        return folds

    elif mode == "sliding":
        # Fixed window slides 1 week forward each fold
        folds = []
        for i in range(0, (len(y) - WEEK - DAILY) // WEEK):
            train_start = i * WEEK
            train_end = train_start + WEEK
            test_start = train_end
            test_end = test_start + DAILY
            folds.append((list(range(train_start, train_end)), list(range(test_start, test_end))))
        return folds

    else:
        raise ValueError("Invalid mode. Choose from reverse, expanding, sliding.")


def run_hwes_cv(y, folds, results_path, mode, trend, seasonal, damped, periods):
    """
    Run HWES on each fold and evaluate performance.

    Parameters:
    -----------
    y : pd.Series - full series
    folds : list - index splits
    results_path : Path - output CSV path
    mode : str - windowing mode
    trend, seasonal, damped, periods : HWES parameters

    Saves:
    -------
    A CSV of per-fold metrics.
    """
    all_results = []

    for i, (train_idx, test_idx) in enumerate(folds):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        try:
            # Fit HWES model
            model = ExponentialSmoothing(
                y_train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=periods if seasonal else None,
                damped_trend=damped
            )
            fit = model.fit(optimized=True, use_brute=True)

            # Forecast next day (96 steps)
            y_pred = fit.forecast(DAILY)

            # Evaluate
            result = eval_metrics(y_pred, y_test).T
            result["fold"] = i + 1
            all_results.append(result)

            # Print progress
            print(f"{mode.capitalize()} Fold {i + 1} — {y.index[train_idx[0]].date()} to {y.index[train_idx[-1]].date()}")

        except Exception as e:
            print(f"Fold {i + 1} failed: {e}")

    # Aggregate all fold results
    df = pd.concat(all_results)
    df.set_index("fold", inplace=True)

    # Ensure parent directory exists
    os.makedirs(results_path.parent, exist_ok=True)

    # Save results
    df.to_csv(results_path)
    print(f"Results saved to {results_path}")


@click.command()
@click.option('--data', type=str, required=True, help="Path to raw data")
@click.option('--mode', type=click.Choice(['reverse', 'expanding', 'sliding']), required=True, help="Windowing strategy")
def main(data, mode):
    """
    CLI entry point for HWES forecasting experiment.

    Parameters:
    -----------
    data : str - Parquet path
    mode : str - windowing strategy
    """
    # Load and reindex time series
    y = preprocess_raw_parquet(data)['recommended_fee_fastestFee'].astype(float).asfreq("15min")

    # Load best HWES parameters from prior random search
    cv_result_path = RESULTS_DIR / "tables" / "hwes" / "hwes_cv_results.csv"
    hyperparam_matrix = pd.read_csv(cv_result_path)
    best_trend = hyperparam_matrix.loc[0, 'trend']
    best_seasonal = hyperparam_matrix.loc[0, 'seasonal']
    best_damped = hyperparam_matrix.loc[0, 'damped']

    # Generate CV folds
    folds = get_folds(y, mode)

    # Choose filename based on mode
    filename_map = {
        "reverse": "expanding_window_reverse_weekly_predictions.csv",
        "expanding": "expanding_window_weekly_predictions.csv",
        "sliding": "sliding_window_weekly_predictions.csv"
    }
    results_path = RESULTS_DIR / "tables" / "hwes" / filename_map[mode]

    # Run cross-validation and save results
    run_hwes_cv(y, folds, results_path, mode, best_trend, best_seasonal, best_damped, DAILY)


if __name__ == "__main__":
    main()