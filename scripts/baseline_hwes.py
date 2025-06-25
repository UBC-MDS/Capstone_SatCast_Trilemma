# baseline_hwes.py
# author: Jenny Zhang
# date: 2025-06-18

"""
Script to build a Holt–Winters Exponential-Smoothing (HWES) baseline
for Bitcoin fee forecasting.

This script performs the following steps:
1. Loads and resamples raw Parquet data to 15-minute frequency.
2. Extracts the target series and splits it into training and
   48-hour test windows.
3. Executes a grid search with time-series cross-validation to tune
   trend, seasonality, and damping options.
4. Fits the best model on the full training window.
5. Serialises the fitted model to ``results/models/hwes_best_train.pkl``.

Usage:
    python scripts/baseline_hwes.py --parquet-path data/raw/mar_5_may_12.parquet
"""

import sys
import pickle
import pandas as pd
import warnings
import click
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")

# Setup project root
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]

# Add project src to path
sys.path.append(str(project_root))

# Import project modules
from src.preprocess_raw_parquet import preprocess_raw_parquet
from scripts.hwes.hwes_extract_split import hwes_extract_split
from scripts.hwes.hwes_cv_optimization import hwes_cv_optimization

# Configurations
FORECAST = 192  # 48 hours * (60 / 15 mins)
DAILY = 96      # 24 hours * (60 / 15 mins)
WINDOWS = 672 * 5  # 5 weeks of rolling windows
STEPS = 672     # Step size = 1 week

@click.command()
@click.option(
    "--parquet-path",
    type=click.Path(exists=True),
    default=str(project_root / "data" / "raw" / "mar_5_may_12.parquet"),
    help="Path to input Parquet file"
)
def main(parquet_path):
    # Step 1: Load and preprocess raw data
    df = preprocess_raw_parquet(parquet_path)

    # Step 2: Extract and split fee series
    fee_series, train, test = hwes_extract_split(df, forecast_horizon=FORECAST)

    # Step 2.5: Save processed data
    data_dir = project_root / "data" / "processed" / "hwes"
    data_dir.mkdir(parents=True, exist_ok=True)
    fee_series.to_csv(data_dir / "fee_series.csv", index=True)
    train.to_csv(data_dir / "train.csv", index=True)
    test.to_csv(data_dir / "test.csv", index=True)

    # Step 3: Cross-validated grid search for hyperparameters
    scaler = MinMaxScaler(feature_range=(1, 2))
    cv_results = hwes_cv_optimization(
        series=train,
        seasonal_periods=DAILY,
        horizon=DAILY,
        window_size=WINDOWS,
        step=STEPS,
        scaler=scaler
    )

    # Step 4: Save and load best parameters
    tables_dir = project_root / "results" / "tables"/ "hwes"
    
    tables_dir.mkdir(parents=True, exist_ok=True)
    cv_results_path = tables_dir / "hwes_cv_results.csv"
    cv_results.to_csv(cv_results_path, index=False)

    hyperparam_matrix = pd.read_csv(cv_results_path)

    if hyperparam_matrix.empty:
        print("HWES cross-validation failed. Not enough data or all configurations failed.")
        print("Try using a longer series like `mar_5_may_12.parquet` instead of the 8-day sample.")
        sys.exit(1)

    best_trend, best_seasonal, best_damped = hyperparam_matrix.iloc[0][['trend', 'seasonal', 'damped']]
    print(f"✅ Best HWES parameters: trend={best_trend}, seasonal={best_seasonal}, damped={best_damped}")

    # Step 5: Train final model
    final_model = ExponentialSmoothing(
        train,
        trend=best_trend,
        seasonal=best_seasonal,
        seasonal_periods=DAILY if best_seasonal else None,
        damped_trend=best_damped
    )
    final_fit = final_model.fit(optimized=True, use_brute=True)

    # Step 6: Save trained model
    model_path = project_root / "results" / "models" / "hwes_best_train.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(final_fit, f)
    print(f"✅ Final model saved to: {model_path}")

if __name__ == '__main__':
    main()
