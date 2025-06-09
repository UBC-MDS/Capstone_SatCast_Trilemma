# sliding_window_weekly_train.py
# author: Yajing Liu
# date: 2025-06-08

# Usage:
# python scripts/sarima/sliding_window_weekly_train.py \
#     --data="./data/processed/sarima/preprocessed_sarima_15min.parquet" \
#     --results="./results/tables/sarima/sliding_window_weekly_predictions.csv"

import click
import os
import sys
import pandas as pd
import numpy as np
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.forecasting.arima import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import mae_with_std_penalty, mae_with_std_dev_penalty_np

@click.command()
@click.option('--data', type=str, default="./data/processed/sarima/preprocessed_sarima_15min.parquet", help="Path to preprocessed Parquet file")
@click.option('--results', type=str, default="./results/tables/sarima/sliding_window_weekly_predictions.csv", help="Path to save results CSV")
def main(data, results):
    """
    Performs SARIMA sliding window forecasting (weekly) and saves evaluation metrics.
    """

    # Load data
    df_resampled = pd.read_parquet(data)
    y = df_resampled['recommended_fee_fastestFee'].iloc[:-96]  # Drop spike day
    y = y.astype(float)

    # Define forecasting horizon (1 day = 96 steps)
    fh = list(range(1, 96 + 1))

    # Sliding window splitter (weekly)
    sliding_cv_weekly = SlidingWindowSplitter(
        window_length=96 * 7,   # 1 week of training
        step_length=96 * 7,     # move window by 1 week
        fh=fh
    )

    # Initialize error containers
    rows = []

    # Run cross-validation
    for i, (train_idx, test_idx) in enumerate(sliding_cv_weekly.split(y)):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        train_start_time = y.index[train_idx[0]]
        train_end_time = y.index[train_idx[-1]]
        test_start_time = y.index[test_idx[0]]
        test_end_time = y.index[test_idx[-1]]

        # SARIMA Model
        y_train_log = np.log1p(y_train)
        forecaster = ARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1, 96))
        forecaster.fit(y_train_log)
        y_pred_log = forecaster.predict(fh=fh)
        y_pred = np.expm1(y_pred_log)

        # Baseline: median of training data
        baseline_value = y_train.median()
        y_pred_baseline = [baseline_value] * len(y_test)

        # Model metrics
        model_mae = mean_absolute_error(y_test, y_pred)
        model_rmse = mean_squared_error(y_test, y_pred) ** 0.5
        model_mape = mean_absolute_percentage_error(y_test, y_pred)
        model_mae_std_penalty = mae_with_std_penalty(y_test.values, y_pred, std_weight=1.0)
        model_mae_std_dev_penalty = mae_with_std_dev_penalty_np(y_test.values, y_pred, std_weight=1.0, de_weight=1.0)

        # Baseline metrics
        baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
        baseline_rmse = mean_squared_error(y_test, y_pred_baseline) ** 0.5
        baseline_mape = mean_absolute_percentage_error(y_test, y_pred_baseline)
        baseline_mae_std_penalty = mae_with_std_penalty(y_test.values, y_pred_baseline, std_weight=1.0)
        baseline_mae_std_dev_penalty = mae_with_std_dev_penalty_np(y_test.values, y_pred_baseline, std_weight=1.0, de_weight=1.0)

        # Store metrics
        rows.append({
            "week": i + 1,
            "train_start": train_start_time,
            "train_end": train_end_time,
            "test_start": test_start_time,
            "test_end": test_end_time,
            # Model
            "model_mae": model_mae,
            "model_rmse": model_rmse,
            "model_mape": model_mape,
            "model_mae_std_penalty": model_mae_std_penalty,
            "model_mae_std_dev_penalty": model_mae_std_dev_penalty,
            # Baseline
            "baseline_mae": baseline_mae,
            "baseline_rmse": baseline_rmse,
            "baseline_mape": baseline_mape,
            "baseline_mae_std_penalty": baseline_mae_std_penalty,
            "baseline_mae_std_dev_penalty": baseline_mae_std_dev_penalty
        })

        # Print fold result
        print(f"Week {i+1}")
        print(f"  Train: {train_start_time} to {train_end_time}")
        print(f"  Test : {test_start_time} to {test_end_time}")
        print(f"  Model     — MAE: {model_mae:.4f}, RMSE: {model_rmse:.4f}, MAPE: {model_mape:.2%}, MAE+STD Penalty: {model_mae_std_penalty:.4f}, MAE+STD+DEV Penalty: {model_mae_std_dev_penalty:.4f}")
        print(f"  Baseline  — MAE: {baseline_mae:.4f}, RMSE: {baseline_rmse:.4f}, MAPE: {baseline_mape:.2%}, MAE+STD Penalty: {baseline_mae_std_penalty:.4f}, MAE+STD+DEV Penalty: {baseline_mae_std_dev_penalty:.4f}")

    # Save results
    df_results = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(results), exist_ok=True)
    df_results.to_csv(results, index=False)

    print(f"\n✅ Results saved to {results}")

if __name__ == '__main__':
    main()
