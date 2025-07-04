# hwes_cv_optimization.py
# author: Jenny Zhang
# date: 2025-06-18
"""
Performs grid-based cross-validation for Holt-Winters Exponential Smoothing (HWES) to
identify the best combination of trend, seasonality, and damping settings for univariate
time series forecasting.

This script performs the following:
1. Defines sliding-window training and testing splits across the full time series.
2. Iterates over a parameter grid of HWES configurations.
3. Fits each model and evaluates it on the next 24-hour (96-step) forecast horizon.
4. Applies both standard metrics (MAE, RMSE, MAPE) and custom volatility-aware metrics.
5. Aggregates performance across all windows to rank configurations by average custom loss.

Usage:
------
Used during the HWES tuning phase before final model selection:

    cv_results = hwes_cv_optimization(series, seasonal_periods=96, horizon=96)
"""

import os
import sys
import warnings

import pandas as pd
import numpy as np
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Add project root to system path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# Import custom evaluation functions
from src.custom_loss_eval import std_diff, dev_error_component, custom_loss_eval

def hwes_cv_optimization(series, seasonal_periods=96, horizon=96, window_size=672, step=672, param_grid=None, scaler=None):
    """
    Construct Grid Search for HWES parameters using cross-validation. 
    GridSearchCV and SlidingWindowCV are not available in statsmodels, so we implement a custom solution.

    Parameters:
    -----------
    series : pd.Series
        Input time series to evaluate.
    seasonal_periods : int
        Number of time steps in one full seasonal cycle (e.g., 96 = 24h @ 15min intervals).
    horizon : int
        Forecast horizon (e.g., 96 = next 24 hours).
    window_size : int
        Size of the training window.
    step : int
        Step size to slide the window.
    param_grid : list of tuples, optional
        List of (trend, seasonal, damped) combinations to evaluate. If None, defaults are used.
    scaler : object, optional
        Scaler (e.g., MinMaxScaler) applied to each training window.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing average scores for each parameter combination,
        sorted by the custom loss metric.
    """
    # If no parameter grid is passed, generate default combinations
    if param_grid is None:
        trend_opts = ['add', 'mul', None]
        seasonal_opts = ['add', 'mul', None]
        damped_opts = [True, False]
        param_grid = [
            (t, s, d) for t, s, d in product(trend_opts, seasonal_opts, damped_opts)
            if not (t is None and d)  # Damped can't be True if there's no trend
        ]

    results = []  # Store aggregated results for each parameter combo
    total_windows = (len(series) - window_size - horizon) // step + 1  # Number of sliding windows

    for trend, seasonal, damped in param_grid:
        # Track metrics for each sliding window
        mae_scores, mape_scores, rmse_scores = [], [], []
        std_errors, dev_errors, custom_scores = [], [], []
        valid = True  # Flag to skip parameter set if one window fails

        for i in range(total_windows):
            print(f"Processing window {i + 1}/{total_windows} with params: {trend}, {seasonal}, {damped}")

            # Define train/test slices for this window
            start = i * step
            end = start + window_size
            test_end = end + horizon
            train = series.iloc[start:end]
            test = series.iloc[end:test_end]

            # Optionally scale training data to avoid numerical instability
            if scaler:
                scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
                min_val = scaled.min()
                shift = abs(min_val) + 1e-3 if min_val <= 0 else 0
                train = pd.Series(scaled + shift, index=train.index)

            try:
                # Suppress convergence warnings during fitting
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Initialize and fit the HWES model
                    model = ExponentialSmoothing(
                        train,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods if seasonal else None,
                        damped_trend=damped
                    )
                    fit = model.fit(optimized=True, use_brute=True)
                    forecast = fit.forecast(horizon)

                    # Evaluate forecasts using standard and custom metrics
                    mae = mean_absolute_error(test, forecast)
                    mape = mean_absolute_percentage_error(test, forecast)
                    rmse = np.sqrt(mean_squared_error(test, forecast))
                    std_error = std_diff(forecast, test)
                    dev_error = dev_error_component(forecast, test).mean()
                    custom_score = custom_loss_eval(forecast, test)

                    # Append scores
                    mae_scores.append(mae)
                    mape_scores.append(mape)
                    rmse_scores.append(rmse)
                    std_errors.append(std_error)
                    dev_errors.append(dev_error)
                    custom_scores.append(custom_score)

            except Exception as e:
                print(f"Exception in window {i + 1}: {e}")
                valid = False  # Skip this parameter combo entirely
                break

        # Only store results if all windows succeeded
        if valid and custom_scores:
            results.append({
                "trend": trend,
                "seasonal": seasonal,
                "damped": damped,
                "avg_mae": np.mean(mae_scores),
                "avg_mape": np.mean(mape_scores),
                "avg_rmse": np.mean(rmse_scores),
                "avg_std_penalty": np.mean(std_errors),
                "avg_dev_error": np.mean(dev_errors),
                "avg_custom_score": np.mean(custom_scores)
            })

    # Return results sorted by custom score (lower is better)
    if not results:
        return pd.DataFrame(columns=[
            "trend", "seasonal", "damped", "avg_mae", "avg_mape", "avg_rmse",
            "avg_std_penalty", "avg_dev_error", "avg_custom_score"
        ])

    # Otherwise, return sorted DataFrame
    cv_results = pd.DataFrame(results)
    return cv_results.sort_values("avg_custom_score")