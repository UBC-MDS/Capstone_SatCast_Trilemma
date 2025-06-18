# scripts/analysis/run_all_models.py

"""
Main script to run all forecasting models on Bitcoin fee data.

This script:
1. Loads the raw fee dataset and preprocesses it.
2. Runs inference using six different models:
   - Holt-Winters Exponential Smoothing (HWES)
   - SARIMA
   - XGBoost
   - Prophet
   - DeepAR
   - Temporal Fusion Transformer (TFT)
3. Saves forecast plots for each model to disk.
4. Aggregates evaluation metrics and exports them as a CSV table.

Ensure all model artifacts are stored in `results/models/`
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Setup paths
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from preprocess_raw_parquet import preprocess_raw_parquet
from plot_series import plot_series

# Add analysis scripts to path
sys.path.append(str(project_root / "scripts" / "analysis"))

# Import individual model predictors
from predict_hwes import predict_hwes
from predict_sarima import predict_sarima
from predict_xgboost import predict_xgboost
from predict_prophet import predict_prophet
from predict_deepar import predict_deepar
from predict_tft import predict_tft

# Define paths
DATA_PATH = project_root / "data" / "raw" / "mar_5_may_12.parquet"
MODEL_DIR = project_root / "results" / "models"
PLOT_DIR = project_root / "results" / "plots"
TABLE_DIR = project_root / "results" / "tables"

# Step 1: Load and preprocess dataset
df_full = preprocess_raw_parquet(DATA_PATH)
df_full = df_full[:-96]                  # Exclude final 96 points for test horizon
df_test = df_full.tail(96).copy()        # Use last 96 points as test set
df_test.reset_index(inplace=True)

# Step 2: Run all models and collect forecasts and metrics
forecasts = {}
metrics = {}

print("Running HWES...")
forecasts["HWES"], metrics["HWES"] = predict_hwes(df_test, MODEL_DIR / "hwes_best_train.pkl")

print("Running SARIMA...")
forecasts["SARIMA"], metrics["SARIMA"] = predict_sarima(df_test, MODEL_DIR / "sarima_final_model.pkl")

print("Running XGBoost...")
forecasts["XGBoost"], metrics["XGBoost"] = predict_xgboost(df_full, df_test, MODEL_DIR / "xgboost.pkl")

print("Running Prophet...")
forecasts["Prophet"], metrics["Prophet"] = predict_prophet(df_test, MODEL_DIR / "prophet_model.json")

print("Running DeepAR...")
forecasts["DeepAR"], metrics["DeepAR"] = predict_deepar(df_full, MODEL_DIR / "best_deepar_model_v5.ckpt")

print("Running TFT...")
forecasts["TFT"], metrics["TFT"] = predict_tft(df_full, MODEL_DIR / "best-model-tft-v5.pt")

# Step 3: Save forecast plots
print("Saving forecast plots...")
for name, df_forecast in forecasts.items():
    ax = plot_series(df_forecast, sid="recommended_fee_fastestFee")
    plt.title(f"{name} forecast vs actual")
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(PLOT_DIR / f"forecast_{name.lower()}.png")
    plt.close()

# Step 4: Save evaluation metrics
print("Saving evaluation metrics...")
metrics_df = pd.concat(metrics.values(), axis=1)
metrics_df.columns = metrics.keys()
metrics_df = metrics_df.round(4)
metrics_df.to_csv(TABLE_DIR / "all_model_metrics.csv")

# Summary
print("Forecasts and metrics saved.")
print(metrics_df)
