import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import seaborn as sns
from prophet.diagnostics import cross_validation, performance_metrics
import itertools
import json

def model_optimization(df,all_params):
    """
    Fine-tune prophet model.

    Parameters:
    ----------
    df : pandas.DataFrame
        Formatted dataset for prophet model training. 
    all_params : dict
        Parameters for fine-tuning.

    Returns:
    -------
    list
        A list containing all fine-tuning results.

    Examples:
    --------
    >>> results = model_optimization(df_prophet_new, {"a":[1,2,3]})
    """
    results = [] 

    for params in all_params:
        m = Prophet(
            daily_seasonality=False,
            weekly_seasonality=False,
            **params
        )
        m.add_seasonality(name='hourly', period=1/24, fourier_order=6)
        m.add_seasonality(name='daily', period=1, fourier_order=8)
        m.add_seasonality(name='weekly', period=7, fourier_order=4)
        
        m.fit(df)
        df_cv = cross_validation(m, initial='7 days', period='1 day', horizon='1 day', parallel="processes")
        df_p = performance_metrics(df_cv)
        
        results.append((params, df_p['rmse'].mean()))
    return results


def create_model_new():
    """
    Create a new prophet model using best hyperparameters. 

    Parameters:
    ----------
    Null

    Returns:
    -------
    Prophet
        Created Prophet model.

    Examples:
    --------
    >>> model = create_model_new()
    """
    
    with open("../analysis/saved_models/prophet.json", "r") as f:
        params = json.load(f)
        
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        changepoint_prior_scale = params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        changepoint_range=params["changepoint_range"],
        n_changepoints=params["n_changepoints"]
    )
    model.add_seasonality(name='hourly', period=1/24, fourier_order=5) 
    model.add_seasonality(name='daily', period=1, fourier_order=10)  
    model.add_seasonality(name='weekly', period=24, fourier_order=5) 

    return model


def create_model_new_holiday(y_train):
    """
    Create a new prophet model with holiday parameters using best hyperparameters. 

    Parameters:
    ----------
    y_train
        Targets of training data.

    Returns:
    -------
    Prophet
        Created Prophet model.

    Examples:
    --------
    >>> model = create_model_new_holiday(y_train)
    """
    with open("../analysis/saved_models/prophet.json", "r") as f:
        params = json.load(f)
        
    s = y_train.reset_index()
    threshold = s['recommended_fee_fastestFee'].quantile(0.9)
    spike_times = s[s['recommended_fee_fastestFee'] > threshold]['timestamp']

    # holidays dataframe
    holidays = pd.DataFrame({
        'ds': pd.to_datetime(spike_times),
        'holiday': 'congestion_spike'
    })

    # remove duplicate
    holidays = holidays.drop_duplicates(subset='ds')
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=False,
        changepoint_prior_scale = params["changepoint_prior_scale"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        changepoint_range=params["changepoint_range"],
        n_changepoints=params["n_changepoints"],
        holidays=holidays
    )

    model.add_seasonality(name='hourly', period=1/24, fourier_order=5) 
    model.add_seasonality(name='daily', period=1, fourier_order=10)  
    model.add_seasonality(name='weekly', period=24, fourier_order=5) 

    return model


def mae_with_std_and_shape_penalty(y_true, y_pred, std_weight=1.0, de_weight=1.0, clip_weight_std=None, clip_weight_dev=None):
    """
    Compute custom MAE loss with additional penalties on std deviation and shape deviation.

    Parameters:
        y_true: np.ndarray, ground truth values
        y_pred: np.ndarray, predicted values
        std_weight: float, weight for std penalty
        de_weight: float, weight for shape deviation penalty
        clip_weight_std: float or None, optional max clip value for std penalty weight
        clip_weight_dev: float or None, optional max clip value for shape deviation weight

    Returns:
        float: combined loss
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    base_loss = np.abs(y_pred - y_true)
    base_mae = np.mean(base_loss)

    # ---- STD penalty ----
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)
    std_penalty = np.abs(pred_std - true_std)
    w_std = base_mae / (std_penalty + 1e-8)
    if clip_weight_std is not None:
        w_std = min(w_std, clip_weight_std)

    # ---- Shape deviation penalty ----
    pred_dev = y_pred - np.mean(y_pred)
    true_dev = y_true - np.mean(y_true)
    dev_error = np.abs(pred_dev - true_dev)
    w_dev = base_loss / (dev_error + 1e-8)
    if clip_weight_dev is not None:
        w_dev = np.clip(w_dev, None, clip_weight_dev)

    # Final loss
    shape_penalty = np.mean(w_dev * dev_error)
    total_loss = base_mae + std_weight * w_std * std_penalty + de_weight * shape_penalty

    return total_loss

def get_result_new(df,y_test,y):
    df.index = y.index
    y_pred = df.iloc[-96:]
    y_pred = np.expm1(y_pred["yhat"])
    split_index = len(y) - 96
    y_train, a = y.iloc[:split_index], y.iloc[split_index:]
    y_baseline = [y_train.median()] * len(y_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    mae_std = mae_with_std_and_shape_penalty(y_test, y_pred)
    base_mae = mean_absolute_error(y_test, y_baseline)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
    base_mape = mean_absolute_percentage_error(y_test, y_baseline)   
    base_mae_std = mae_with_std_and_shape_penalty(y_test, y_baseline)
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.4f}")
    print(f"MAE with std and shape penalty: {mae_std:.4f}")
    print(f"Base MAE: {base_mae:.4f}")
    print(f"Base RMSE: {base_rmse:.4f}")
    print(f"Base MAPE: {base_mape:.4f}")
    print(f"Base MAE with std and shape penalty: {base_mae_std:.4f}")
    
    # result_df = pd.DataFrame({
    #     'Actual': y_test,
    #     'Predicted': y_pred
    # }, index=y_test.index)

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


def evaluate_model(df_new, weeks=10,holiday=0):
    """
    Evaluate model performance using sliding window over multiple weeks.

    Returns:
        y_pred_sliding: dict of forecasts
        metrics_per_week: list of dicts for each week's metrics
        avg_metrics: dict of average metrics over all weeks
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
        # print("Week",(i+1))
        df_sliding = df_new[0+i*7*96:7*96+i*7*96]
        y = df_sliding["recommended_fee_fastestFee"]
        split_index = len(y) - 96
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        df_prophet = y_train.reset_index()
        df_prophet = df_prophet.rename(columns={
            'timestamp': 'ds',
            'recommended_fee_fastestFee': 'y'
        })
        df_prophet['y'] = np.log1p(df_prophet['y'])

        # baseline
        y_baseline = [y_train.median()] * len(y_test)

        if holiday == 0:
            model = create_model_new()
        else:
            model = create_model_new_holiday(y_train)
        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=96, freq='15min')
        forecast = model.predict(future)

        y_pred_temp = forecast.iloc[-96:]
        y_pred_temp = np.expm1(y_pred_temp["yhat"])
        y_pred_temp.index =y_test.index

        mae = mean_absolute_error(y_test, y_pred_temp)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_temp))
        mape = mean_absolute_percentage_error(y_test, y_pred_temp)
        mae_std = mae_with_std_and_shape_penalty(y_test, y_pred_temp)

        base_mae = mean_absolute_error(y_test, y_baseline)
        base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
        base_mape = mean_absolute_percentage_error(y_test, y_baseline)   
        base_mae_std = mae_with_std_and_shape_penalty(y_test, y_baseline)

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

        # print(f"Baseline MAE: {base_mae:.4f}, RMSE: {base_rmse:.4f}, MAPE: {base_mape:.4f}, MAE+STD: {base_mae_std:.4f}")
        # print(f"Model    MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, MAE+STD: {mae_std:.4f}")
        # print("-" * 60)

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


def evaluate_model_external(df_new, weeks=10):
    """
    Evaluate model performance using sliding window over multiple weeks.

    Returns:
        y_pred_sliding: dict of forecasts
        metrics_per_week: list of dicts for each week's metrics
        avg_metrics: dict of average metrics over all weeks
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
        # print("Week",(i+1))
        df_sliding = df_new[0+i*7*96:7*96+i*7*96]

        X = df_sliding.drop(columns = "recommended_fee_fastestFee")
        y = df_sliding["recommended_fee_fastestFee"]

        # shift
        shift_steps = 96 # shift 24h
        X = X.shift(periods=shift_steps)
        X.dropna(inplace=True)
        y = y.loc[X.index]
        
        split_index = len(y) - 96
        X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
        y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
        df_prophet = y_train.reset_index()
        df_prophet = df_prophet.rename(columns={
            'timestamp': 'ds',
            'recommended_fee_fastestFee': 'y'
        })
        df_prophet['y'] = np.log1p(df_prophet['y'])

        # baseline
        y_baseline = [y_train.median()] * len(y_test)

        model = create_model_new()

        # add external features
        X_col = X_train.reset_index()
        X_col = X_col.drop(columns = "timestamp")
        for i in X_col.columns.values:
            df_prophet[i] = X_col[i].copy()
            model.add_regressor(i)

        model.fit(df_prophet)
        future = model.make_future_dataframe(periods=96, freq='15min')

        # add external features
        for i in X.columns.values:
            future[i] = list(df_prophet[i]) + list(X_test[i])

        forecast = model.predict(future)

        y_pred_temp = forecast.iloc[-96:]
        y_pred_temp = np.expm1(y_pred_temp["yhat"])
        y_pred_temp.index =y_test.index

        mae = mean_absolute_error(y_test, y_pred_temp)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_temp))
        mape = mean_absolute_percentage_error(y_test, y_pred_temp)
        mae_std = mae_with_std_and_shape_penalty(y_test, y_pred_temp)

        base_mae = mean_absolute_error(y_test, y_baseline)
        base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
        base_mape = mean_absolute_percentage_error(y_test, y_baseline)   
        base_mae_std = mae_with_std_and_shape_penalty(y_test, y_baseline)

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

        # print(f"Baseline MAE: {base_mae:.4f}, RMSE: {base_rmse:.4f}, MAPE: {base_mape:.4f}, MAE+STD: {base_mae_std:.4f}")
        # print(f"Model    MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.4f}, MAE+STD: {mae_std:.4f}")
        # print("-" * 60)

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

def evaluate_best_model(df_new):
    y = df_new["recommended_fee_fastestFee"]
    split_index = len(y) - 96
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]
    df_prophet = y_train.reset_index()
    df_prophet = df_prophet.rename(columns={
        'timestamp': 'ds',
        'recommended_fee_fastestFee': 'y'
    })
    df_prophet['y'] = np.log1p(df_prophet['y'])

    # baseline
    y_baseline = [y_train.median()] * len(y_test)

    model = create_model_new_holiday(y_train)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=96, freq='15min')
    forecast = model.predict(future)

    y_pred_temp = forecast.iloc[-96:]
    y_pred_temp = np.expm1(y_pred_temp["yhat"])
    y_pred_temp.index =y_test.index

    mae = mean_absolute_error(y_test, y_pred_temp)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_temp))
    mape = mean_absolute_percentage_error(y_test, y_pred_temp)
    mae_std = mae_with_std_and_shape_penalty(y_test, y_pred_temp)

    base_mae = mean_absolute_error(y_test, y_baseline)
    base_rmse = np.sqrt(mean_squared_error(y_test, y_baseline))
    base_mape = mean_absolute_percentage_error(y_test, y_baseline)   
    base_mae_std = mae_with_std_and_shape_penalty(y_test, y_baseline)


    # metrics_per_week = {
    #     "mae": mae,
    #     "rmse": rmse,
    #     "mape": mape,
    #     "mae_std": mae_std,
    #     "base_mae": base_mae,
    #     "base_rmse": base_rmse,
    #     "base_mape": base_mape,
    #     "base_mae_std": base_mae_std
    # }
    get_result_new(forecast,y_test,y)
    # return metrics_per_week