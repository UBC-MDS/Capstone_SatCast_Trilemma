# expanding_window_daily_train.py
# author: Yajing Liu
# date: 2025-06-08

# Usage:
# python scripts/sarima/expanding_window_daily_train.py \
#     --data="./data/processed/sarima/preprocessed_sarima_15min.parquet" \
#     --output="./results/tables/sarima/expanding_window_daily_predictions.csv"

import click
import os
import sys
import pandas as pd
import numpy as np
import warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.utils import mae_with_std_penalty, mae_with_std_dev_penalty_np
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.forecasting.arima import AutoARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

@click.command()
@click.option('--data', type=str, required=True, help="Path to preprocessed data parquet")
@click.option('--output', type=str, required=True, help="Path to save csv predictions")
def main(data, output):
    # Load data
    df = pd.read_parquet(data)
    y = df['recommended_fee_fastestFee'].iloc[:-96]  # Drop final spike day

    fh = list(range(1, 97))
    expanding_cv = ExpandingWindowSplitter(initial_window=96*7, step_length=96, fh=fh)

    # Initialize lists
    model_mae_list, model_rmse_list, model_mape_list = [], [], []
    model_mae_std_penalty_list, model_mae_std_dev_penalty_list = [], []

    baseline_mae_list, baseline_rmse_list, baseline_mape_list = [], [], []
    baseline_mae_std_penalty_list, baseline_mae_std_dev_penalty_list = [], []

    # Loop through splits
    for i, (train_idx, test_idx) in enumerate(expanding_cv.split(y)):
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        y_train_log = np.log1p(y_train)
        forecaster = AutoARIMA(sp=96, suppress_warnings=True)
        forecaster.fit(y_train_log)
        y_pred = np.expm1(forecaster.predict(fh=fh))

        # Baseline
        baseline_value = y_train.median()
        y_pred_baseline = [baseline_value] * len(y_test)

        # Model metrics
        model_mae = mean_absolute_error(y_test, y_pred)
        model_rmse = mean_squared_error(y_test, y_pred) ** 0.5
        model_mape = mean_absolute_percentage_error(y_test, y_pred)
        model_mae_std_penalty = mae_with_std_penalty(y_test, y_pred, std_weight=1.0)
        model_mae_std_dev_penalty = mae_with_std_dev_penalty_np(y_test.values, y_pred, std_weight=1.0, de_weight=1.0)

        model_mae_list.append(model_mae)
        model_rmse_list.append(model_rmse)
        model_mape_list.append(model_mape)
        model_mae_std_penalty_list.append(model_mae_std_penalty)
        model_mae_std_dev_penalty_list.append(model_mae_std_dev_penalty)

        # Baseline metrics
        baseline_mae = mean_absolute_error(y_test, y_pred_baseline)
        baseline_rmse = mean_squared_error(y_test, y_pred_baseline) ** 0.5
        baseline_mape = mean_absolute_percentage_error(y_test, y_pred_baseline)
        baseline_mae_std_penalty = mae_with_std_penalty(y_test, y_pred_baseline, std_weight=1.0)
        baseline_mae_std_dev_penalty = mae_with_std_dev_penalty_np(y_test.values, y_pred_baseline, std_weight=1.0, de_weight=1.0)

        baseline_mae_list.append(baseline_mae)
        baseline_rmse_list.append(baseline_rmse)
        baseline_mape_list.append(baseline_mape)
        baseline_mae_std_penalty_list.append(baseline_mae_std_penalty)
        baseline_mae_std_dev_penalty_list.append(baseline_mae_std_dev_penalty)

        print(f"Fold {i+1} done.")

    # Save results to CSV
    df_results = pd.DataFrame({
        'model_mae': model_mae_list,
        'model_rmse': model_rmse_list,
        'model_mape': model_mape_list,
        'model_mae_std_penalty': model_mae_std_penalty_list,
        'model_mae_std_dev_penalty': model_mae_std_dev_penalty_list,
        'baseline_mae': baseline_mae_list,
        'baseline_rmse': baseline_rmse_list,
        'baseline_mape': baseline_mape_list,
        'baseline_mae_std_penalty': baseline_mae_std_penalty_list,
        'baseline_mae_std_dev_penalty': baseline_mae_std_dev_penalty_list,
    })

    os.makedirs(os.path.dirname(output), exist_ok=True)
    df_results.to_csv(output, index=False)
    print(f"✅ Results saved to {output}")

if __name__ == "__main__":
    main()
