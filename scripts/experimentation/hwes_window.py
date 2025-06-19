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
5. Write results to a CSV at the path specified via `--results`.

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
  --results ./results/tables/hwes_experimentation_rev_expand_win_results.csv \
  --mode reverse

2. Weekly expanding: 
python scripts/experimentation/hwes_window.py \
  --data ./data/raw/mar_5_may_12.parquet \
  --results ./results/tables/hwes_experimentation_expand_win_results.csv \
  --mode expanding

3. Weekly sliding:
python scripts/experimentation/hwes_window.py \
  --data ./data/raw/mar_5_may_12.parquet \
  --results ./results/tables/hwes_experimentation_slide_win_results.csv \
  --mode sliding
"""
  
import os
import sys
import click
import warnings
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.preprocess_raw_parquet import preprocess_raw_parquet
from src.custom_loss_eval import eval_metrics
from src.read_csv_data import read_csv_data

# Constants
DAILY = 96  # 96 steps = 24 hours at 15-min intervals
WEEK = 96 * 7
RESULTS_DIR = './results'


def get_folds(y, mode):
    """
    Generate training and testing index pairs for cross-validation based on the selected mode.

    Parameters:
    -----------
    y : pd.Series
        The full time series to be split.
    mode : str
        One of ['reverse', 'expanding', 'sliding']:
            - 'reverse': test set is fixed to the final day, training expands backward.
            - 'expanding': walk-forward with cumulative weekly training.
            - 'sliding': fixed-size weekly training sliding forward.

    Returns:
    --------
    list of tuples:
        Each tuple contains (train_idx, test_idx), both lists of integer indices.
    """
    if mode == "reverse":
        test_end = len(y)
        test_start = test_end - DAILY
        train_end = test_start
        n_folds = train_end // WEEK
        folds = []
        for i in range(1, n_folds + 1):
            train_start = max(0, train_end - i * WEEK)
            folds.append((list(range(train_start, train_end)), list(range(test_start, test_end))))
        return folds

    elif mode == "expanding":
        folds = []
        for i in range(1, (len(y) - WEEK - DAILY) // WEEK + 1):
            train_end = WEEK + (i - 1) * WEEK
            train_idx = list(range(train_end))
            test_idx = list(range(train_end, train_end + DAILY))
            folds.append((train_idx, test_idx))
        return folds

    elif mode == "sliding":
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
    Run Holt-Winters Exponential Smoothing on each cross-validation fold.

    Parameters:
    -----------
    y : pd.Series
        Full target time series with 15-minute frequency.
    folds : list of tuples
        Output from `get_folds()`, containing (train_idx, test_idx) pairs.
    results_path : str
        File path to save the per-fold evaluation results as CSV.
    mode : str
        One of ['reverse', 'expanding', 'sliding'], used for progress tracking.

    Output:
    -------
    A CSV file saved to `results_path` containing fold-wise metrics:
        - MAE, RMSE, MAPE, custom loss, etc.
    """
    all_results = []

    for i, (train_idx, test_idx) in enumerate(folds):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        try:
            model = ExponentialSmoothing(
                y_train,
                trend=trend,
                seasonal=seasonal,
                seasonal_periods=periods if seasonal else None,
                damped_trend=damped
            )
            fit = model.fit(optimized=True, use_brute=True)
            y_pred = fit.forecast(DAILY)

            result = eval_metrics(y_pred, y_test).T
            result["fold"] = i + 1
            all_results.append(result)

            print(f"✅ {mode.capitalize()} Fold {i + 1} — {y.index[train_idx[0]].date()} to {y.index[train_idx[-1]].date()}")

        except Exception as e:
            print(f"❌ Fold {i + 1} failed: {e}")

    df = pd.concat(all_results)
    df.set_index("fold", inplace=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.to_csv(results_path)
    print(f"✅ Results saved to {results_path}")


@click.command()
@click.option('--data', type=str, required=True, help="Path to raw data")
@click.option('--results', type=str, required=True, help="Path to save results CSV")
@click.option('--mode', type=click.Choice(['reverse', 'expanding', 'sliding']), required=True, help="Windowing strategy")
def main(data, results, mode):
    """
    CLI entry point for running HWES cross-validation.

    Parameters:
    -----------
    data : str
        Path to the input Parquet file containing the time series.
    results : str
        Path to save the evaluation results CSV.
    mode : str
        One or more of ['reverse', 'expanding', 'sliding'] indicating fold strategy.
    """
    y = preprocess_raw_parquet(data)['recommended_fee_fastestFee'].astype(float).asfreq("15min")

    hyperparam_matrix = read_csv_data(os.path.join(RESULTS_DIR, 'tables', 'hwes_cv_results.csv'))
    best_trend, best_seasonal, best_damped = hyperparam_matrix.iloc[0][['trend', 'seasonal', 'damped']]

    folds = get_folds(y, mode)
    run_hwes_cv(y, folds, results, mode, best_trend, best_seasonal, best_damped, DAILY)


if __name__ == "__main__":
    main()
