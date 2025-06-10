import os
import sys
import warnings

import pandas as pd
import numpy as np
from itertools import product
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.custom_loss_eval import std_diff, dev_error_component, custom_loss_eval

def cv_optimization(series, seasonal_periods=96, horizon=96, window_size=672, step=672, param_grid=None, scaler=None):
    """
    Perform sliding window cross-validation for Holt-Winters Exponential Smoothing.
    
    Args:
        series (pd.Series): Time series to evaluate
        seasonal_periods (int): Seasonal period (e.g., 288 for 24h @ 5min)
        horizon (int): Forecasting horizon (e.g., 288 for next 24h)
        window_size (int): Size of each training window
        step (int): Step size to slide the window
        param_grid (list of tuples): List of (trend, seasonal, damped) combinations
        scaler: Optional scaler (must support fit_transform)
    
    Returns:
        pd.DataFrame: Sorted parameter results (trend, seasonal, damped, avg_mae, avg_rmse)
    """
    if param_grid is None:
        trend_opts = ['add', 'mul', None]
        seasonal_opts = ['add', 'mul', None]
        damped_opts = [True, False]
        param_grid = [
            (t, s, d) for t, s, d in product(trend_opts, seasonal_opts, damped_opts)
            if not (t is None and d)
        ]

    results = []
    total_windows = (len(series) - window_size - horizon) // step + 1

    for trend, seasonal, damped in param_grid:
        mae_scores, mape_scores, rmse_scores, std_errors, dev_errors, custom_scores, = [], [], [], [], [], []
        valid = True

        for i in range(total_windows):
            print(f"Processing window {i + 1}/{total_windows} with params: {trend}, {seasonal}, {damped}")
            start = i * step
            end = start + window_size
            test_end = end + horizon

            train = series.iloc[start:end]
            test = series.iloc[end:test_end]
            
            if scaler:
                scaled = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
                min_val = scaled.min()
                shift = abs(min_val) + 1e-3 if min_val <= 0 else 0
                train = pd.Series(scaled + shift, index=train.index)

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ExponentialSmoothing(
                        train,
                        trend=trend,
                        seasonal=seasonal,
                        seasonal_periods=seasonal_periods if seasonal is not None else None,
                        damped_trend=damped
                    )
                    
                    fit = model.fit(optimized=True, use_brute=True)
                    forecast = fit.forecast(horizon)

                    mae = mean_absolute_error(test, forecast)
                    mape = mean_absolute_percentage_error(test, forecast)
                    rmse = np.sqrt(mean_squared_error(test, forecast))
                    std_error = std_diff(forecast, test)
                    dev_error = dev_error_component(forecast, test).mean()
                    custom_score = custom_loss_eval(forecast, test)

                    mae_scores.append(mae)
                    mape_scores.append(mape)
                    rmse_scores.append(rmse)
                    std_errors.append(std_error)
                    dev_errors.append(dev_error)
                    custom_scores.append(custom_score)       
            
            except Exception as e:
                print(f"Exception in window {i + 1}: {e}")
                valid = False
                break

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

    cv_results = pd.DataFrame(results).sort_values("avg_custom_score")
    return cv_results