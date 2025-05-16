import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from itertools import product
import warnings

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def hwes_cross_val(series, seasonal_periods=288, horizon=288, window_size=2016, step=288, param_grid=None, scaler=None):
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
    timestamps = series.index
    total_windows = (len(series) - window_size - horizon) // step + 1

    for trend, seasonal, damped in param_grid:
        mae_scores, mape_scores, rmse_scores = [], [], []
        valid = True

        for i in range(total_windows):
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
                        seasonal_periods=seasonal_periods if seasonal else None,
                        damped_trend=damped,
                        # use_boxcox=True
                    )
                    fit = model.fit(optimized=True, use_brute=True)
                    forecast = fit.forecast(horizon)

                    mae = mean_absolute_error(test, forecast)
                    mape = mean_absolute_percentage_error(test, forecast)
                    rmse = np.sqrt(mean_squared_error(test, forecast))

                    mae_scores.append(mae)
                    mape_scores.append(mape)
                    rmse_scores.append(rmse)
            except Exception:
                valid = False
                break

        if valid and mae_scores:
            results.append({
                "trend": trend,
                "seasonal": seasonal,
                "damped": damped,
                "avg_mae": np.mean(mae_scores),
                "avg_mape": np.mean(mape_scores),
                "avg_rmse": np.mean(rmse_scores)
            })

    df_results = pd.DataFrame(results).sort_values("avg_mae")
    return df_results
