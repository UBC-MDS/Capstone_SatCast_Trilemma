# xgboost_utils.py
# author: Tengwei Wang
# date: 2025-06-18

# Functions for xgboost model data preprocessing, training, evaluating and optimizing. 

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import ForecastingRandomizedSearchCV
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_error as mean_absolute_error_sktime
from sktime.split import SlidingWindowSplitter,ExpandingWindowSplitter

import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src" 
sys.path.insert(0, str(src_path))
from custom_loss_eval import custom_loss_eval as mae_with_std_penalty_np

def data_split(df,interval):
    """
    Split the data into training data and test data.

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to be split.
    interval : int
        The time interval between data points.

    Returns:
    -------
    pd.DataFrame
        Exogenous features of training data.
    pd.DataFrame
        Targets of training data.
    pd.DataFrame
        Exogenous features of test data.
    pd.DataFrame
        Targets of test data.

    Examples:
    --------
    >>> X_train, X_test, y_train, y_test = data_split(df,15)
    """
    y = df["recommended_fee_fastestFee"]
    X = df.drop(columns = "recommended_fee_fastestFee")
    shift_steps = int(24*(60/interval)) # shift 24h
    X = X.shift(periods=shift_steps)
    X = X.iloc[shift_steps:]
    y = y.loc[X.index]

    # last 24h as test
    split_index = int(len(X) - 24*(60/interval))
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    return X_train, X_test, y_train, y_test

def build_random_search(y_train, param_dist,interval,if_sliding=1):
    """
    Build random search framework. 

    Parameters:
    ----------
    y_train : pandas.DataFrame
        Targets of test data.
    param_dist : dict
        Parameters for random search.
    interval : int
        The time interval between data points.
    if_sliding : int
        Using whether sliding window or expanding window.

    Returns:
    -------
    ForecastingRandomizedSearchCV

    Examples:
    --------
    >>> random_search = build_random_search(y_train, param_dist,15,1)
    """
    n = int(24*(60/interval))
    
    # base model
    xgb_base = xgb.XGBRegressor(
        objective='reg:squarederror',
        tree_method='hist',
        n_jobs=-1
    )

    # forecaster
    forecaster = make_reduction(
        estimator=xgb_base,
        window_length=n,  # 24h lag
        strategy='recursive',
        scitype='infer'
    )

    # sliding window cv
    # train_size = len(y_train)
    window_length = int(2*n)  # Training window covers 80% of data
    step_length = n   # Step size to create ~5 folds
    slcv = SlidingWindowSplitter(
        window_length=window_length,
        step_length=step_length,
        fh=ForecastingHorizon([n], is_relative=True),
        start_with_window=True
    )

    # expanding window cv
    train_size = len(y_train)
    initial_window = int(0.8*train_size)
    step_length = int(0.2*train_size)   
    excv = ExpandingWindowSplitter(
        initial_window=initial_window,
        step_length=step_length,
        fh=ForecastingHorizon([n], is_relative=True)
    )

    # random search
    if if_sliding == 0:
        random_search = ForecastingRandomizedSearchCV(
            forecaster=forecaster,
            param_distributions=param_dist,
            n_iter=20,
            scoring=mean_absolute_error_sktime,
            cv=excv,
            verbose=1
        )
        return random_search
        
    random_search = ForecastingRandomizedSearchCV(
        forecaster=forecaster,
        param_distributions=param_dist,
        n_iter=20,
        scoring=mean_absolute_error_sktime,
        cv=slcv,
        verbose=1
    )

    return random_search

def create_lag_features_fast(df, target_col, lags):
    """
    Create lag features of targets. 

    Parameters:
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the whole dataset.
    target_col : str
        Name of target col.
    lags : array
        Array containing time intervals to be lagged.

    Returns:
    -------
    pandas.DataFrame
        The whole dataset with lag features.

    Examples:
    --------
    >>> df = create_lag_features_fast(df, 'recommended_fee_fastestFee', [1,2,3,4,5]])
    """
    lagged_dfs = [
        df[[target_col]].shift(lag).rename(columns={target_col: f'{target_col}_lag_{lag}'})
        for lag in lags
    ]
    return pd.concat([df] + lagged_dfs, axis=1)

def evaluate_model(df, param_dist, interval=15, weeks=10, fh=None, sliding=0):
    """
    Evaluate model performance using sliding window or expanding window over multiple weeks.

    Parameters:
    ----------
    df: pd.DataFrame
        Original dataset.
    param_dist: list
        Combinations of hyperparameters.
    interval: int
        Time interval between data points.
    weeks: int
        Number of weeks the dataset contains. 
    fh: ForecastingHorizon
        Forecasting time range.
    sliding: int
        Whether to use sliding window or not. 

    Returns:
    -------
    dic
        Dict of predict values. 
    list 
        LIST of dicts for each week's metrics.
    dict
        Dict of average metrics over all weeks.

    Examples:
    --------
    >>> metrics_per_week, avg_metrics,y_test,y_pred = evaluate_model(df,param_dist,15,10,fh,0)
    """

    metrics_per_week = []

    avg_mae = 0
    avg_rmse = 0
    avg_mape = 0
    avg_mae_std = 0

    base_avg_mae = 0
    base_avg_rmse = 0
    base_avg_mape = 0
    base_avg_mae_std = 0

    for i in range(weeks):
        # print(f"Week {i + 1}")
        df_new = pd.DataFrame()
        if(sliding == 1):
            df_new = df[0+i*7*96:7*96+i*7*96]
        else:
            df_new = df[:7*96+i*7*96]

        X_train, X_test, y_train, y_test = data_split(df_new, interval)

        random_search = build_random_search(y_train, param_dist, interval, sliding)
        random_search.fit(X=X_train, y=y_train, fh=fh)
        best_forecaster = random_search.best_forecaster_

        y_pred = best_forecaster.predict(fh=fh, X=X_test)
        y_pred.index = X_test.index

        y_baseline = [y_train.median()] * len(y_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)
        mae_std = mae_with_std_penalty_np(y_pred,y_test)

        base_mae = mean_absolute_error(y_test, y_baseline)
        base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
        base_mape = mean_absolute_percentage_error(y_test, y_baseline)
        base_mae_std = mae_with_std_penalty_np(y_baseline,y_test)

        avg_mae += mae
        avg_rmse += rmse
        avg_mape += mape
        avg_mae_std += mae_std

        base_avg_mae += base_mae
        base_avg_rmse += base_rmse
        base_avg_mape += base_mape
        base_avg_mae_std += base_mae_std

        metrics_per_week.append({
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "mae_std": mae_std,
            "base_mae": base_mae,
            "base_rmse": base_rmse,
            "base_mape": base_mape,
            "base_mae_std": base_mae_std
        })

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

    return metrics_per_week, avg_metrics,y_test,y_pred
    
def plot_result(y_test,y_pred):
    """
    Plot the comparison of predicted values vs. actual values.

    Parameters:
    ----------
    y_true
        ground truth values
    y_pred
        predicted values

    Returns:
        Null
    Examples:
    --------
    >>> y_test = [1,2,3]
    >>> y_pred = [1,3,4]
    >>> plot_result(y_test,y_pred)
    """
    fig,ax = plt.subplots(figsize=(14, 5))
    # Plot actual vs. predicted values
    ax.plot(y_test, label="Actual", color="black")
    ax.plot(y_pred, label="Forecast", color="blue")

    # Axis labels and title
    ax.set_title(f"Forecast vs Actual", fontsize=14)
    ax.set_xlabel("Timestamp", fontsize=12)
    ax.set_ylabel("Transaction Fee (sats/vB)", fontsize=12)
    # Improve appearance
    ax.grid(True)
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    
def calculate_metrics(best_model,df,interval,fh):
    """
    Evaluate the best model performance over the whole dataset.

    Parameters:
    ----------
    best_model : Forecast
        The best model
    df: pd.DataFrame
        Original dataset.
    interval: int
        Time interval between data points.
    fh: ForecastingHorizon
        Forecasting time range.

    Returns:
    -------
    dic
        Dict of predict values. 
    list 
        LIST of dicts for each week's metrics.
    dict
        Dict of average metrics over all weeks.

    Examples:
    --------
    >>> metrics_per_week, avg_metrics,y_test,y_pred = evaluate_model(df,param_dist,15,fh)
    """
    metrics = []

    X_train, X_test, y_train, y_test = data_split(df, interval)

    y_pred = best_model.predict(fh=fh, X=X_test)
    y_pred.index = X_test.index

    y_baseline = [y_train.median()] * len(y_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae_std = mae_with_std_penalty_np(y_pred,y_test)

    base_mae = mean_absolute_error(y_test, y_baseline)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
    base_mape = mean_absolute_percentage_error(y_test, y_baseline)
    base_mae_std = mae_with_std_penalty_np(y_baseline,y_test)

    metrics.append({
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "mae_std": mae_std,
        "base_mae": base_mae,
        "base_rmse": base_rmse,
        "base_mape": base_mape,
        "base_mae_std": base_mae_std
    })

    return metrics,y_test,y_pred




    