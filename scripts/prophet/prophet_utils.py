# prophet_utils.py
# author: Tengwei Wang
# date: 2025-06-18

"""
Utility functions for training, tuning, and evaluating Prophet models for Bitcoin fee forecasting.

This module includes:
- Hyperparameter optimization using Prophet's built-in cross-validation
- Standard Prophet model creation with or without custom holidays
- Evaluation over weekly windows (with or without external regressors)
- Visualization and scoring using custom and standard metrics
"""

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics
import json
import sys
from pathlib import Path

# Add src path for custom loss function import
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from custom_loss_eval import custom_loss_eval as mae_with_std_penalty_np
model_path = str(project_root / "results" / "models")

# === Model Optimization ===
def model_optimization(df, all_params):
    """
    Run grid search for Prophet model hyperparameters using cross-validation.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'ds' and 'y' columns (log-transformed target).
    all_params : list of dict
        List of Prophet hyperparameter combinations.

    Returns
    -------
    list of (dict, float)
        Each tuple contains the parameter set and average RMSE from CV.
    """
    results = []

    for i, params in enumerate(all_params):
        print(f"🔍 Testing parameter set {i+1}/{len(all_params)}: {params}")
        try:
            # Initialize model with custom seasonality
            m = Prophet(daily_seasonality=False, weekly_seasonality=False, **params)
            m.add_seasonality(name='hourly', period=1/24, fourier_order=6)
            m.add_seasonality(name='daily', period=1, fourier_order=8)
            m.add_seasonality(name='weekly', period=7, fourier_order=4)
            m.fit(df)

            # Perform CV
            df_cv = cross_validation(m, initial='7 days', period='1 day', horizon='1 day', parallel="processes")
            df_p = performance_metrics(df_cv)
            mean_rmse = df_p['rmse'].mean()
            results.append((params, mean_rmse))
        except Exception as e:
            print(f"Skipping due to error: {e}")
            continue

    if not results:
        raise ValueError("All hyperparameter sets failed. Try smaller horizon or more data.")

    return results

# === Model Creation ===
def create_model_new():
    """
    Create a Prophet model using best hyperparameters saved in JSON file.

    Returns
    -------
    Prophet
        Configured Prophet model.
    """
    with open(model_path + "/prophet_hyperparameter.json", "r") as f:
        params = json.load(f)

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        **params
    )
    model.add_seasonality(name='hourly', period=1/24, fourier_order=5)
    model.add_seasonality(name='daily', period=1, fourier_order=10)
    model.add_seasonality(name='weekly', period=24, fourier_order=5)
    return model

def create_model_new_holiday(y_train):
    """
    Create Prophet model with added congestion spike holidays.

    Parameters
    ----------
    y_train : pd.DataFrame
        Training data with 'recommended_fee_fastestFee' and timestamp index.

    Returns
    -------
    Prophet
        Prophet model with holiday events.
    """
    with open(model_path + "/prophet_best_params.json", "r") as f:
        params = json.load(f)

    # Identify congestion spikes
    s = y_train.reset_index()
    threshold = s['recommended_fee_fastestFee'].quantile(0.9)
    spike_times = s[s['recommended_fee_fastestFee'] > threshold]['timestamp']

    holidays = pd.DataFrame({
        'ds': pd.to_datetime(spike_times),
        'holiday': 'congestion_spike'
    }).drop_duplicates(subset='ds')

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        holidays=holidays,
        **params
    )
    model.add_seasonality(name='hourly', period=1/24, fourier_order=5)
    model.add_seasonality(name='daily', period=1, fourier_order=10)
    model.add_seasonality(name='weekly', period=24, fourier_order=5)
    return model

# === Forecast Result Printer ===
def get_result_new(df, y_test, y):
    """
    Evaluate and plot Prophet forecast.

    Parameters
    ----------
    df : pd.DataFrame
        Prophet forecast DataFrame.
    y_test : np.ndarray
        Actual values.
    y : pd.Series
        Original full series for computing baseline.
    """
    df.index = y.index
    y_pred = np.expm1(df.iloc[-96:]['yhat'])  # Un-log transform
    split_index = len(y) - 96
    y_train = y.iloc[:split_index]
    y_baseline = [y_train.median()] * len(y_test)

    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae_std = mae_with_std_penalty_np(y_pred, y_test)

    base_mae = mean_absolute_error(y_test, y_baseline)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
    base_mape = mean_absolute_percentage_error(y_test, y_baseline)
    base_mae_std = mae_with_std_penalty_np(y_baseline, y_test)

    # Print
    print(f"MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nMAPE: {mape:.4f}\nMAE+STD: {mae_std:.4f}")
    print(f"Base MAE: {base_mae:.4f}\nBase RMSE: {base_rmse:.4f}\nBase MAPE: {base_mape:.4f}\nBase MAE+STD: {base_mae_std:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_test, label="Actual", color="black")
    ax.plot(y_pred, label="Forecast", color="blue")
    ax.set_title("Forecast vs Actual")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Transaction Fee (sats/vB)")
    ax.grid(True)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# === Sliding Evaluation ===
def evaluate_model(df_new, weeks=10, holiday=0):
    """
    Evaluate Prophet with or without holiday logic over multiple weeks.

    Parameters
    ----------
    df_new : pd.DataFrame
        Data with 15-min resolution.
    weeks : int
        How many weeks to evaluate.
    holiday : int
        Whether to use spike-based holiday model.

    Returns
    -------
    metrics_per_week : list of dict
    avg_metrics : dict
    """
    metrics_per_week = []
    avg_mae = avg_rmse = avg_mape = avg_mae_std = 0
    base_avg_mae = base_avg_rmse = base_avg_mape = base_avg_mae_std = 0

    for i in range(weeks):
        df_sliding = df_new[i * 96 * 7: (i + 1) * 96 * 7]
        y = df_sliding["recommended_fee_fastestFee"]
        split_index = len(y) - 96
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        df_prophet = y_train.reset_index().rename(columns={'timestamp': 'ds', 'recommended_fee_fastestFee': 'y'})
        df_prophet['y'] = np.log1p(df_prophet['y'])

        model = create_model_new_holiday(y_train) if holiday else create_model_new()
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=96, freq='15min')
        forecast = model.predict(future)

        y_pred = np.expm1(forecast.iloc[-96:]["yhat"])
        y_pred.index = y_test.index
        y_baseline = [y_train.median()] * len(y_test)

        # Evaluate
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae_std = mae_with_std_penalty_np(y_pred, y_test)

        base_mae = mean_absolute_error(y_test, y_baseline)
        base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
        base_mape = mean_absolute_percentage_error(y_test, y_baseline)
        base_mae_std = mae_with_std_penalty_np(y_baseline, y_test)

        metrics_per_week.append({
            "mae": mae, "rmse": rmse, "mape": mape, "mae_std": mae_std,
            "base_mae": base_mae, "base_rmse": base_rmse,
            "base_mape": base_mape, "base_mae_std": base_mae_std
        })

        # Accumulate
        avg_mae += mae
        avg_rmse += rmse
        avg_mape += mape
        avg_mae_std += mae_std
        base_avg_mae += base_mae
        base_avg_rmse += base_rmse
        base_avg_mape += base_mape
        base_avg_mae_std += base_mae_std

    avg_metrics = {
        "avg_mae": avg_mae / weeks,
        "avg_rmse": avg_rmse / weeks,
        "avg_mape": avg_mape / weeks,
        "avg_mae_std": avg_mae_std / weeks,
        "base_avg_mae": base_avg_mae / weeks,
        "base_avg_rmse": base_avg_rmse / weeks,
        "base_avg_mape": base_avg_mape / weeks,
        "base_avg_mae_std": base_avg_mae_std / weeks
    }

    return metrics_per_week, avg_metrics

# === External Feature Evaluation ===
def evaluate_model_external(df_new, weeks=10):
    """
    Evaluate Prophet with external regressors over multiple weeks.

    Parameters
    ----------
    df_new: pd.DataFrame
        Full dataset with external variables.
    weeks: int
        Number of evaluation weeks.

    Returns
    -------
    metrics_per_week : list of dict
    avg_metrics : dict
    """
    metrics_per_week = []
    avg_mae = avg_rmse = avg_mape = avg_mae_std = 0
    base_avg_mae = base_avg_rmse = base_avg_mape = base_avg_mae_std = 0

    for i in range(weeks):
        df_sliding = df_new[i*7*96: (i+1)*7*96]

        X = df_sliding.drop(columns="recommended_fee_fastestFee")
        y = df_sliding["recommended_fee_fastestFee"]

        X = X.shift(96).dropna()
        y = y.loc[X.index]

        split_index = len(y) - 96
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

        df_prophet = y_train.reset_index().rename(columns={"timestamp": "ds", "recommended_fee_fastestFee": "y"})
        df_prophet['y'] = np.log1p(df_prophet['y'])

        model = create_model_new()
        for col in X_train.columns:
            df_prophet[col] = X_train[col].values
            model.add_regressor(col)

        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=96, freq='15min')
        for col in X.columns:
            future[col] = list(df_prophet[col]) + list(X_test[col])

        forecast = model.predict(future)
        y_pred_temp = np.expm1(forecast.iloc[-96:]["yhat"])
        y_pred_temp.index = y_test.index

        # Evaluate
        mae = mean_absolute_error(y_test, y_pred_temp)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_temp))
        mape = mean_absolute_percentage_error(y_test, y_pred_temp)
        mae_std = mae_with_std_penalty_np(y_pred_temp, y_test)

        y_baseline = [y_train.median()] * len(y_test)
        base_mae = mean_absolute_error(y_test, y_baseline)
        base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
        base_mape = mean_absolute_percentage_error(y_test, y_baseline)
        base_mae_std = mae_with_std_penalty_np(y_baseline, y_test)

        metrics_per_week.append({
            "mae": mae, "rmse": rmse, "mape": mape, "mae_std": mae_std,
            "base_mae": base_mae, "base_rmse": base_rmse,
            "base_mape": base_mape, "base_mae_std": base_mae_std
        })

        # Aggregate
        avg_mae += mae
        avg_rmse += rmse
        avg_mape += mape
        avg_mae_std += mae_std
        base_avg_mae += base_mae
        base_avg_rmse += base_rmse
        base_avg_mape += base_mape
        base_avg_mae_std += base_mae_std

    avg_metrics = {
        "avg_mae": avg_mae / weeks,
        "avg_rmse": avg_rmse / weeks,
        "avg_mape": avg_mape / weeks,
        "avg_mae_std": avg_mae_std / weeks,
        "base_avg_mae": base_avg_mae / weeks,
        "base_avg_rmse": base_avg_rmse / weeks,
        "base_avg_mape": base_avg_mape / weeks,
        "base_avg_mae_std": base_avg_mae_std / weeks
    }

    return metrics_per_week, avg_metrics

# === Final Model Evaluation ===
def evaluate_best_model(df_new):
    """
    Evaluate final Prophet model on full dataset (no sliding).

    Parameters
    ----------
    df_new: pd.DataFrame
        Full dataset.

    Returns
    -------
    list, dict
        Weekly metrics and averages.
    """
    y = df_new["recommended_fee_fastestFee"]
    split_index = len(y) - 96
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    df_prophet = y_train.reset_index().rename(columns={'timestamp': 'ds', 'recommended_fee_fastestFee': 'y'})
    df_prophet['y'] = np.log1p(df_prophet['y'])

    model = create_model_new_holiday(y_train)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=96, freq='15min')
    forecast = model.predict(future)

    y_pred_temp = np.expm1(forecast.iloc[-96:]["yhat"])
    y_pred_temp.index = y_test.index
    y_pred_temp.to_csv("prophet.csv", index=True)

    get_result_new(forecast, y_test, y)
