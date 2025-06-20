# analysis.py
# author: Ximin Xu
# date: 2025-06-18

"""
analysis.py

Script to run inference with all trained models and compare their
forecasting performance.

This script performs the following steps:
1. Loads the full pre-processed dataset and a 24-hour hold-out test set.
2. Runs inference for six models (HWES, SARIMA, XGBoost, Prophet,
   DeepAR, TFT) using their saved checkpoints.
3. Computes MAE, RMSE, MAPE, and a custom volatility-aware loss
   for each model.
4. Saves every model’s forecasts (Pickle) and the aggregated
   metric table (CSV) to ``results/tables``.
5. Optionally produces and stores forecast-vs-actual plots.

Usage:
    python scripts/analysis.py
"""


import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pickle

# Setup paths
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root / "src"))

from preprocess_raw_parquet import preprocess_raw_parquet
from plot_series import plot_series
from custom_loss_eval import eval_metrics 

# Add analysis scripts to path
sys.path.append(str(project_root / "scripts" / "hwes"))
from hwes_predict import predict_hwes

sys.path.append(str(project_root / "scripts" / "sarima"))
from sarima_predict import predict_sarima

sys.path.append(str(project_root / "scripts" / "xgboost"))
from xgboost_predict import predict_xgboost

sys.path.append(str(project_root / "scripts" / "prophet"))
from prophet_predict import predict_prophet

sys.path.append(str(project_root / "scripts" / "deepar"))
from deepar_predict import predict_deepar

sys.path.append(str(project_root / "scripts" / "tft"))
from tft_predict import predict_tft

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


# Add median baseline
print("Evaluating Median Baseline...")
test_median = df_full[:-96]["recommended_fee_fastestFee"].median()
test_median_forecast = pd.Series([test_median] * 96)
true_values = df_test["recommended_fee_fastestFee"].values
metrics["Median"] = eval_metrics(test_median_forecast.values, true_values)
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


with open(TABLE_DIR / "all_forecasts.pkl", "wb") as f:
    pickle.dump(forecasts, f)
# Summary
print("Forecasts and metrics saved.")
print(metrics_df)
