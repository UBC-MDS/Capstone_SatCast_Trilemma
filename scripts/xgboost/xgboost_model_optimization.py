# xgboost_model_optimization.py
# author: Tengwei Wang
# date: 2025-06-18

# Optimize xgboost model. 

import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "scripts" /"xgboost"
sys.path.insert(0, str(src_path))
from xgboost_data_preprocess import data_preprocess
from xgboost_utils import evaluate_best_model
import joblib
from sktime.forecasting.base import ForecastingHorizon
import numpy as np



def optimization(data_path,result):
    """
    Optimize to find the best hyperparameters. 

    Parameters:
    ----------
    data_path : str
        The path of training dataset.
    result: str
        The path to save the best model. 

    Returns:
    -------
    Null
    """
    param_dist = {
        'estimator__n_estimators': [50, 100, 150],
        'estimator__max_depth': [1, 2, 3],
        'estimator__learning_rate': [0.01, 0.05, 0.1],
        'estimator__subsample': [0.6, 0.8, 0.9],
        'estimator__colsample_bytree': [0.6, 0.8, 0.9],
        'estimator__gamma': [1, 3, 5],
        'estimator__reg_lambda': [5, 10, 20],
        'estimator__reg_alpha': [5, 10, 20]
    }
    fh = ForecastingHorizon(np.arange(1, 97), is_relative=True)
    df = data_preprocess(data_path)
    metrics,y_test,y_pred, best_forecaster = evaluate_best_model(df, param_dist, interval=15, fh=fh)
    file_path = result+"/xgboost.pkl"
    joblib.dump(best_forecaster, file_path)
    print("Best model saved.")
