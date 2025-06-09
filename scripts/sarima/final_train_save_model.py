# final_train_save_model.py
# author: Yajing Liu
# date: 2025-06-08

# Usage:
# python scripts/sarima/final_train_save_model.py \
#     --data="./data/processed/sarima/preprocessed_sarima_15min.parquet" \
#     --model="./results/models/sarima_final_model.pkl"

import click
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils import mae_with_std_penalty, mae_with_std_dev_penalty_np
from src.save_model import save_model

@click.command()
@click.option('--data', type=str, default="./data/processed/sarima/preprocessed_sarima_15min.parquet", help="Path to preprocessed Parquet file")
@click.option('--model', type=str, default="./results/models/sarima_final_model.pkl", help="Path to save trained SARIMA model (Pickle)")
@click.option('--results', type=str, default="./results/tables/sarima/final_train_metrics.csv", help="Path to save final metrics CSV")
def main(data, model, results):
    """
    Performs final SARIMA training on full data and saves model + evaluation metrics.
    """

    # Load data
    df_resampled = pd.read_parquet(data)
    y = df_resampled['recommended_fee_fastestFee'].iloc[:-96]
    y = y.astype(float)

    # Split into train/test (last 1 day)
    y_train, y_test = temporal_train_test_split(y, test_size=96)

    # SARIMA model
    y_train_log = np.log1p(y_train)
    forecaster = ARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1, 96))
    forecaster.fit(y_train_log)

    # Forecast
    fh = list(range(1, 97))
    y_pred_log = forecaster.predict(fh=fh)
    y_pred = np.expm1(y_pred_log)

    # Baseline
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

    # Print results
    print("=== SARIMA Model ===")
    print(f"MAE: {model_mae:.4f}")
    print(f"RMSE: {model_rmse:.4f}")
    print(f"MAPE: {model_mape:.2%}")
    print(f"MAE + STD Penalty: {model_mae_std_penalty:.4f}")
    print(f"MAE + STD + DEV Penalty: {model_mae_std_dev_penalty:.4f}")

    print("\n=== Baseline (Median) ===")
    print(f"MAE: {baseline_mae:.4f}")
    print(f"RMSE: {baseline_rmse:.4f}")
    print(f"MAPE: {baseline_mape:.2%}")
    print(f"MAE + STD Penalty: {baseline_mae_std_penalty:.4f}")
    print(f"MAE + STD + DEV Penalty: {baseline_mae_std_dev_penalty:.4f}")
    

    # Save model
    save_model(forecaster, model)
    print(f"\n✅ SARIMA model saved to {model}")


    # Save metrics
    os.makedirs(os.path.dirname(results), exist_ok=True)
    df_metrics = pd.DataFrame([{
        "model_mae": model_mae,
        "model_rmse": model_rmse,
        "model_mape": model_mape,
        "model_mae_std_penalty": model_mae_std_penalty,
        "model_mae_std_dev_penalty": model_mae_std_dev_penalty,
        "baseline_mae": baseline_mae,
        "baseline_rmse": baseline_rmse,
        "baseline_mape": baseline_mape,
        "baseline_mae_std_penalty": baseline_mae_std_penalty,
        "baseline_mae_std_dev_penalty": baseline_mae_std_dev_penalty
    }])

if __name__ == '__main__':
    main()
