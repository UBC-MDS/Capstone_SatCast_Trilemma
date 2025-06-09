import os
import sys  
import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import warnings
warnings.filterwarnings("ignore")


sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.preprocess_raw_parquet import preprocess_raw_parquet
from src.read_csv_data import read_csv_data
from src.save_csv_data import save_csv_data
from src.save_model import save_model
from src.custom_loss_eval import *
from scripts.hwes.extract_split import extract_split
from scripts.hwes.cv_optimization import cv_optimization

# Configuration
FORECAST = 192  # 48 hours * (60 / 15 mins)
DAILY = 96  # 24 hours * (60 / 15 mins)
WINDOWS = 672 * 5  # 5 weeks
STEPS = 672  # 1 week

# File Path and Directory
INPUT_DATA = "./data/raw/mar_5_may_12.parquet"
DATA_DIR = './data/processed/hwes'
RESULTS_DIR = './results'
MODEL_FROM = './results/models/hwes_best_train.pkl'

if __name__ == '__main__':

    ## ---------Step 1: Load, clean, and resample the raw data---------
    df = preprocess_raw_parquet(INPUT_DATA)

    ## ---------Step 2: Extract and split the target series------------
    fee_series, train, test = extract_split(df, forecast_horizon=FORECAST)

    # Save the processed and split data as CSV files
    save_csv_data(fee_series, os.path.join(DATA_DIR, 'fee_series.csv'), index=True)
    save_csv_data(train, os.path.join(DATA_DIR, 'train.csv'), index=True)
    save_csv_data(test, os.path.join(DATA_DIR, 'test.csv'), index=True)

    # ## -----Step 3: Hyperparameter optimization through Grid Search----
    # # Set the scaler to prevent numerical instability during grid search
    # scaler = MinMaxScaler(feature_range=(1, 2)) 

    # cv_results = cv_optimization(
    #     series=train,
    #     seasonal_periods=DAILY,
    #     horizon=DAILY,
    #     window_size=WINDOWS,  
    #     step=STEPS,
    #     scaler=scaler
    # )

    # # Save the results to a CSV file
    # os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)
    # cv_results.to_csv(os.path.join(RESULTS_DIR, 'tables', 'hwes_cv_results.csv'))

    ## -------------Step 4: Save the best parameters-------------------
    hyperparam_matrix = read_csv_data(os.path.join(RESULTS_DIR, 'tables', 'hwes_cv_results.csv'))
    best_trend, best_seasonal, best_damped = hyperparam_matrix.iloc[0][['trend', 'seasonal', 'damped']]
    print(f"Best HWES parameters: trend={best_trend}, seasonal={best_seasonal}, damped={best_damped}")

    ## -----Step 5: Train the final model with the best parameters-----
    final_model = ExponentialSmoothing(
        train,
        trend=best_trend,
        seasonal=best_seasonal,
        seasonal_periods=DAILY if best_seasonal else None,
        damped_trend=best_damped
    )

    final_fit = final_model.fit(optimized=True, use_brute=True)

    # Save the final training results using save_model
    os.makedirs(os.path.join(RESULTS_DIR, 'models'), exist_ok=True)
    save_model(final_fit, os.path.join(RESULTS_DIR, 'models', 'hwes_best_train.pkl'))
    # save_model(final_fit, os.path.join(RESULTS_DIR, 'models', 'hwes_best_sample.pkl'))

    ## ---------------Step 6: Make the forecast------------------------
    # read in training model (hwes_best_train model object)
    with open(MODEL_FROM, 'rb') as f:
        model_fit = pickle.load(f)

    forecast = model_fit.forecast(FORECAST)

    # Save the forecast results
    forecast_df = pd.DataFrame(forecast, columns=['forecast'])
    forecast_df.index = test.index
    
    save_csv_data(forecast_df, os.path.join(RESULTS_DIR, 'tables', 'hwes_forecast.csv'), index=True)

    ## -----------Step 7: Evaluate the model performance--------------
    # Evaluate for non-spike day 
    eval_results = eval_metrics(forecast[ : DAILY], test[ : DAILY])
    save_csv_data(eval_results, os.path.join(RESULTS_DIR, 'tables', 'hwes_eval_results.csv'), index=True)
    



