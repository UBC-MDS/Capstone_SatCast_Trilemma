# xgboost_model_optimization.py
# author: Tengwei Wang
# date: 2025-06-18

"""
xgboost_model_optimization.py

Sets up and returns a randomized hyperparameter search for XGBoost time series forecasting.

Responsibilities:
-----------------
1. Splits preprocessed data into train/test sets.
2. Defines the hyperparameter grid for the XGBoost regressor.
3. Constructs a `RandomizedSearchCV` wrapper using sktime.

Key Features:
-------------
- Customizable forecasting interval for multi-step prediction.
- Returns an untrained search object, ready to be fit externally.
- Decouples search setup from model training.

Typical Usage:
--------------
Called during the optimization phase to generate the model search space and train/test inputs.
"""

import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "scripts" /"xgboost"
sys.path.insert(0, str(src_path))
from xgboost_data_preprocess import data_preprocess
from xgboost_utils import data_split, build_random_search
import joblib
import numpy as np



def optimization(df,result, interval=15):
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
    X_train, X_test, y_train, y_test = data_split(df, interval)
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
    random_search = build_random_search(y_train, param_dist, interval, 0) 
    return random_search, X_train, y_train
