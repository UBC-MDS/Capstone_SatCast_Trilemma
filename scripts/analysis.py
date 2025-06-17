import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Setup paths
project_root = Path(__file__).resolve().parent.parent

print(project_root)
sys.path.append(str(project_root / "src"))
from preprocess_raw_parquet import preprocess_raw_parquet
from plot_series import plot_series
sys.path.append(str(project_root / "scripts" / "analysis"))
# Import individual model predictors
from predict_hwes import predict_hwes
from predict_sarima import predict_sarima
from predict_xgboost import predict_xgboost
from predict_prophet import predict_prophet
from predict_deepar import predict_deepar
from predict_tft import predict_tft

# Paths
DATA_PATH = project_root / "data" / "raw" / "mar_5_may_12.parquet"
MODEL_DIR = project_root / "results" / "models"
PLOT_DIR = project_root / "results" / "plots"
TABLE_DIR = project_root / "results" / "tables"

# Load and prepare data
df_full = preprocess_raw_parquet(DATA_PATH)
df_full = df_full[:-96]
df_test = df_full.tail(96).copy()
df_test.reset_index(inplace=True)

# Run all models
forecasts = {}
metrics = {}

forecasts["HWES"], metrics["HWES"] = predict_hwes(df_test, MODEL_DIR / "hwes_best_train.pkl")
forecasts["SARIMA"], metrics["SARIMA"] = predict_sarima(df_test, MODEL_DIR / "sarima_final_model.pkl")
forecasts["XGBoost"], metrics["XGBoost"] = predict_xgboost(df_full, df_test, MODEL_DIR / "xgboost.pkl")
forecasts["Prophet"], metrics["Prophet"] = predict_prophet(df_test, MODEL_DIR / "prophet_model.json")
forecasts["DeepAR"], metrics["DeepAR"] = predict_deepar(df_full, MODEL_DIR / "best_deepar_model_v5.ckpt")
forecasts["TFT"], metrics["TFT"] = predict_tft(df_full, MODEL_DIR / "best-model-tft-v5.pt")

# Save plots
for name, df_forecast in forecasts.items():
    ax = plot_series(df_forecast, sid="recommended_fee_fastestFee")
    plt.title(f"{name} forecast vs actual")
    plt.tight_layout()
    fig = ax.get_figure()
    fig.savefig(PLOT_DIR / f"forecast_{name.lower()}.png")
    plt.close()

# Save metrics
metrics_df = pd.concat(metrics.values(), axis=1)
metrics_df.columns = metrics.keys()
metrics_df = metrics_df.round(4)
metrics_df.to_csv(TABLE_DIR / "all_model_metrics.csv")

print("Forecasts and metrics saved.")
print(metrics_df)
