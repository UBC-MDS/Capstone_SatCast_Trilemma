import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer, mean_absolute_percentage_error
import matplotlib.pyplot as plt
import joblib  
from scipy.stats import uniform, randint
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import seaborn as sns
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.model_selection import temporal_train_test_split, ForecastingRandomizedSearchCV
from sktime.forecasting.base import ForecastingHorizon
from sktime.performance_metrics.forecasting import mean_absolute_error as mean_absolute_error_sktime
from sktime.split import SlidingWindowSplitter,ExpandingWindowSplitter

def data_split(df,interval):
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


    





    