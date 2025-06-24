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
from prophet.diagnostics import cross_validation, performance_metrics
import json
import sys
from pathlib import Path

# Add src path for custom loss function import
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


model_path = str(project_root / "results" / "models")

# === Model Optimization ===
def model_optimization(df, y_train, all_params):
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
            m = create_model_new_holiday(y_train)
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
