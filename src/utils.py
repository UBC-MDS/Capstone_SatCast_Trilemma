import os
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from itertools import product

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.custom_loss_eval import *

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def hwes_cross_val(series, seasonal_periods=288, horizon=288, window_size=2016, step=2016, param_grid=None, scaler=None):
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
        mae_scores, mape_scores, rmse_scores, std_errors, dev_errors, custom_scores, = [], [], [], [], [], []
        valid = True

        for i in range(total_windows):
            print(f"Processing window {i + 1}/{total_windows} for trend={trend}, seasonal={seasonal}, damped={damped}")
            # start = i * step
            # end = start + window_size
            # test_end = end + horizon

            # train = series.iloc[start:end]
            # test = series.iloc[end:test_end]
            end = window_size + i * step
            test_end = end + horizon

            train = series.iloc[:end]
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
                    std_penalty = std_penalty_component(forecast, test)
                    dev_error = dev_error_component(forecast, test)
                    custom_score = custom_loss_eval(forecast, test)

                    mae_scores.append(mae)
                    mape_scores.append(mape)
                    rmse_scores.append(rmse)
                    std_errors.append(std_penalty)
                    dev_errors.append(dev_error)
                    custom_scores.append(custom_score)       
            
            except Exception:
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

    df_results = pd.DataFrame(results).sort_values("avg_custom_score")
    return df_results

def hwes_train_test(
    series,
    seasonal_periods=288,
    horizon=288,
    window_size=2016,
    method='expanding',
    trend='mul',
    seasonal='mul',
    damped_trend=True,
    max_splits=None, 
    optimize_method=None
):
    """
    Perform cross-validation for Holt-Winters Exponential Smoothing (HWES) using expanding or sliding window.

    Parameters
    ----------
    series : pd.Series
        Time series data indexed by timestamp.
    seasonal_periods : int
        Number of periods in a full seasonal cycle.
    horizon : int
        Forecast horizon in time steps.
    window_size : int
        Size of the training window or expanding window.
    method : str
        'expanding' or 'sliding'.
    trend : str or None
        Trend component: 'add', 'mul', or None.
    seasonal : str or None
        Seasonal component: 'add', 'mul', or None.
    damped_trend : bool
        Whether to apply dampening.
    max_splits : int or None
        Maximum number of CV splits to run. None = all possible splits.
    optimize_method : str or None
        Optional method passed to `.fit()` to control optimizer (e.g., 'Nelder-Mead').

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['train_end', 'mae', 'rmse', 'mape']
    """
    errors = []
    series = series.sort_index()

    if method == 'expanding':
        split_points = range(window_size, len(series) - horizon, window_size)
    elif method == 'sliding':
        split_points = range(0, len(series) - window_size - horizon, horizon)
    else:
        raise ValueError("Method must be 'expanding' or 'sliding'")

    if max_splits is not None:
        split_points = list(split_points)[:max_splits]

    for split in split_points:
        if method == 'expanding':
            train = series.iloc[:split]
            test = series.iloc[split:split + horizon]
        else:
            train = series.iloc[split:split + window_size]
            test = series.iloc[split + window_size:split + window_size + horizon]

        try:
            if len(test) < horizon:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                warnings.simplefilter("ignore", RuntimeWarning)
                fit = ExponentialSmoothing(
                    train,
                    trend=trend,
                    seasonal=seasonal,
                    damped_trend=damped_trend,
                    seasonal_periods=seasonal_periods
                ).fit(optimized=True, method=optimize_method)
            
            forecast = fit.forecast(horizon)
            mae = mean_absolute_error(test, forecast)
            rmse = np.sqrt(mean_squared_error(test, forecast))
            mape = mean_absolute_percentage_error(test, forecast)
            errors.append((train.index[-1], mae, rmse, mape))

        except Exception:
            errors.append((train.index[-1], None, None, None))

    return pd.DataFrame(errors, columns=['train_end', 'mae', 'rmse', 'mape'])

# def hwes_train_test(
#     series,
#     seasonal_periods=288,
#     horizon=288,
#     window_size=2016,
#     method='expanding',
#     trend='mul',
#     seasonal='mult',
#     damped_trend=True
# ):
#     """
#     Perform cross-validation for Holt-Winters Exponential Smoothing (HWES) using expanding or sliding window.

#     Parameters
#     ----------
#     series : pd.Series
#         Time series data indexed by timestamp (e.g., Bitcoin transaction fees).
#     seasonal_periods : int, optional
#         Number of periods in a full seasonal cycle (default is 288 for daily seasonality in 5-minute data).
#     horizon : int, optional
#         Forecast horizon in number of time steps (e.g., 288 = next 24 hours if 5-min interval).
#     window_size : int, optional
#         Size of the training window or expanding window size.
#     method : {'expanding', 'sliding'}, optional
#         Cross-validation method:
#             - 'expanding': growing training window starting from window_size
#             - 'sliding' : fixed-size window moving forward by horizon each iteration
#     trend : {'add', 'mul', None}, optional
#         Trend component of HWES. 'add' = additive trend, 'mul' = multiplicative trend, None = no trend.
#     seasonal : {'add', 'mul', None}, optional
#         Seasonal component of HWES. Same options as trend.
#     damped_trend : bool, optional
#         Whether to dampen the trend component.

#     Returns
#     -------
#     pd.DataFrame
#         A DataFrame containing the following columns for each training/evaluation split:
#         - 'train_end' : Timestamp of the last point in the training set
#         - 'mae'       : Mean Absolute Error
#         - 'rmse'      : Root Mean Squared Error
#         - 'mape'      : Mean Absolute Percentage Error (0-1 scale)
#     """
#     errors = []
#     series = series.sort_index()

#     if method == 'expanding':
#         split_points = range(window_size, len(series) - horizon, window_size)
#     elif method == 'sliding':
#         split_points = range(0, len(series) - window_size - horizon, horizon)
#     else:
#         raise ValueError("Method must be 'expanding' or 'sliding'")

#     for split in split_points:
#         if method == 'expanding':
#             train = series.iloc[:split]
#             test = series.iloc[split:split + horizon]
#         else:
#             train = series.iloc[split:split + window_size]
#             test = series.iloc[split + window_size:split + window_size + horizon]

#         try:
#             if len(test) < horizon:
#                 continue

#             with warnings.catch_warnings():
#                 warnings.simplefilter("ignore", ConvergenceWarning)
#                 warnings.simplefilter("ignore", RuntimeWarning)
#                 model = ExponentialSmoothing(
#                     train,
#                     trend=trend,
#                     seasonal=seasonal,
#                     damped_trend=damped_trend,
#                     seasonal_periods=seasonal_periods
#                 ).fit()
#             forecast = model.forecast(horizon)
#             mae = mean_absolute_error(test, forecast)
#             rmse = np.sqrt(mean_squared_error(test, forecast))
#             mape = mean_absolute_percentage_error(test, forecast)
#             errors.append((train.index[-1], mae, rmse, mape))
#         except Exception:
#             errors.append((train.index[-1], None, None, None))

#     return pd.DataFrame(errors, columns=['train_end', 'mae', 'rmse', 'mape'])


def mae_with_std_penalty(y_true, y_pred, std_weight=1.0):
    """
    MAE + std penalty.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Model predictions.
    std_weight : float
        Weight for std penalty.

    Returns
    -------
    total_loss : float
        MAE + std penalty.
    """
    error = np.abs(y_true - y_pred)
    mae = np.mean(error)
    std_penalty = np.abs(np.std(y_true) - np.std(y_pred))
    return mae + std_weight * std_penalty



def mae_with_std_penalty_np(
    y_pred,
    y_true,
    std_weight=1.0,
    de_weight=1.0
):
    """
    Compute MAE with additional penalties:
    - std_penalty: penalizes difference in standard deviation between predictions and ground truth.
    - deviation error: penalizes shape mismatch in distribution after mean-centering.

    Parameters:
    -----------
    y_pred : np.ndarray
        Predicted values.
    y_true : np.ndarray
        Ground truth values.
    std_weight : float
        Weight for standard deviation penalty.
    de_weight : float
        Weight for deviation shape penalty.

    Returns:
    --------
    float
        Final loss value.
    """
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    base_loss = np.abs(y_pred - y_true)
    mae = base_loss.mean()

    # Std penalty
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    std_penalty = np.abs(pred_std - true_std)

    # Deviation penalty
    pred_dev = y_pred - np.mean(y_pred)
    true_dev = y_true - np.mean(y_true)
    dev_error = np.abs(pred_dev - true_dev)

    total_loss = base_loss + std_weight * std_penalty + de_weight * dev_error
    return total_loss.mean()

