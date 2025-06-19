"""
sarima_window.py

Runs SARIMA back-tests on Bitcoin-fee data with **one of three weekly
windowing schemes**:

1. **Reverse expanding window** – keeps the *last* 24 h as a fixed test set and
   trains on an ever-growing window that moves *backwards* in 1-week chunks.
2. **Weekly expanding window** – classic walk-forward CV that starts with one
   week of data, adds a week each fold, and always predicts the *next* day.
3. **Weekly sliding window** – rolls a fixed-length 7-day training window
   forward by 1 week, forecasting the following day each time.

Workflow
--------
1. Load the 15-minute-resampled Parquet created in the preprocessing step.
2. Build the fold indices for the chosen windowing *mode*.
3. For each fold  
   a. log-transform the training slice  
   b. fit SARIMA(1,0,1)(1,0,1,96)  
   c. predict the next 96 steps (24 h) and invert the log  
   d. score with `eval_metrics()` → custom_loss, MAE, RMSE, MAPE, etc.
4. Concatenate fold metrics into a tidy DataFrame (`fold` = row index).
5. Write the CSV to the path supplied by `--results`.

Key Features
------------
- **Horizon** 1 day = 96 × 15 min.
- **No baseline** stored (median baseline used only for debugging; omit to
  simplify output).
- **Single script**; choose the splitter with `--mode`
  (`reverse|expanding|sliding`).
- Saves one of  
  `expanding_window_reverse_weekly_predictions.csv`  
  `expanding_window_weekly_predictions.csv`  
  `sliding_window_weekly_predictions.csv`
  depending on the mode you pass.

Typical Usage
-------------
1. Reverse weekly expanding (fixed last-day test):

python scripts/experimentation/sarima_window.py \
  --data ./data/processed/sarima/preprocessed_sarima_15min.parquet \
  --results ./results/tables/sarima/expanding_window_reverse_weekly_predictions.csv \
  --mode reverse
  
  
2. Weekly expanding:

python scripts/experimentation/sarima_window.py \
  --data ./data/processed/sarima/preprocessed_sarima_15min.parquet \
  --results ./results/tables/sarima/expanding_window_weekly_predictions.csv \
  --mode expanding
  
3. Weekly sliding:

python scripts/experimentation/sarima_window.py \
  --data ./data/processed/sarima/preprocessed_sarima_15min.parquet \
  --results ./results/tables/sarima/sliding_window_weekly_predictions.csv \
  --mode sliding
"""


import os
import sys
import click
import warnings
import pandas as pd
import numpy as np
from sktime.forecasting.arima import ARIMA
from sktime.forecasting.model_selection import ExpandingWindowSplitter, SlidingWindowSplitter

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.custom_loss_eval import eval_metrics


def get_folds(y, mode):
    """
    Return a list of (train_indices, test_indices) for the selected windowing mode.

    Parameters
    ----------
    y : pd.Series
        Target time series (assumed 15-min frequency).
    mode : str
        One of 'reverse', 'expanding', 'sliding'.

    Returns
    -------
    folds : list of (train_idx, test_idx)
        Each fold is a tuple of index lists for training and testing.
    """
    fh = list(range(1, 97))  # 1-day horizon
    if mode == "reverse":
        # Fixed test: final day
        test_end = len(y)
        test_start = test_end - 96
        train_end = test_start
        n_folds = train_end // (96 * 7)
        folds = []
        for i in range(1, n_folds + 1):
            train_start = max(0, train_end - i * 96 * 7)
            folds.append((list(range(train_start, train_end)), list(range(test_start, test_end))))
        return folds
    elif mode == "expanding":
        splitter = ExpandingWindowSplitter(initial_window=96 * 7, step_length=96 * 7, fh=fh)
    elif mode == "sliding":
        splitter = SlidingWindowSplitter(window_length=96 * 7, step_length=96 * 7, fh=fh)
    else:
        raise ValueError("Invalid mode. Choose from reverse, expanding, sliding.")
    return list(splitter.split(y))


def run_sarima_cv(y, folds, results_path, mode):
    """
    Trains and evaluates SARIMA models using cross-validation folds.

    Parameters:
    -----------
    y : pd.Series
        Target time series (15-minute frequency), excluding the final spike day.

    folds : list of (train_idx, test_idx)
        List of index pairs defining training and test windows for cross-validation.

    results_path : str
        Path to the output CSV file for saving evaluation metrics.

    mode : str
        One of {'reverse', 'expanding', 'sliding'}, used for printing progress context.

    Output:
    -------
    - A CSV file saved to `results_path` containing per-fold metrics such as MAE, RMSE, etc.
    """
    fh = list(range(1, 97))
    all_results = []

    for i, (train_idx, test_idx) in enumerate(folds):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        model = ARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1, 96))
        model.fit(np.log1p(y_train))
        y_pred = np.expm1(model.predict(fh=fh))

        result = eval_metrics(y_pred, y_test).T
        result["fold"] = i + 1
        all_results.append(result)

        print(f"✅ {mode.capitalize()} Fold {i + 1} — {y.index[train_idx[0]].date()} to {y.index[train_idx[-1]].date()}")

    df = pd.concat(all_results)
    df.set_index("fold", inplace=True)
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    df.to_csv(results_path)
    print(f"\n✅ Results saved to {results_path}")


@click.command()
@click.option('--data', type=str, required=True, help="Path to preprocessed data")
@click.option('--results', type=str, required=True, help="Path to save results CSV")
@click.option('--mode', type=click.Choice(['reverse', 'expanding', 'sliding']), required=True, help="Which window mode to run")
def main(data, results, mode):
    """
    Entry point for running SARIMA cross-validation.

    Parameters
    ----------
    data : str
        Path to the preprocessed 15-minute interval Parquet file.

    results : str
        Path to save the resulting evaluation metrics CSV.

    mode : {'reverse', 'expanding', 'sliding'}
        Type of windowing strategy to use for cross-validation.
    """
    y = pd.read_parquet(data)['recommended_fee_fastestFee'].iloc[:-96].astype(float).asfreq("15min")
    folds = get_folds(y, mode)
    run_sarima_cv(y, folds, results, mode)


if __name__ == "__main__":
    main()
