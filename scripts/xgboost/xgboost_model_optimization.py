# xgboost_model_optimization.py
# author: Tengwei Wang
# date: 2025-06-18

"""
Script to define and return a randomized hyperparameter search space for 
XGBoost-based Bitcoin fee forecasting using time series data.

This script performs the following steps:
1. Splits the preprocessed dataset into training and testing sets using aligned lag logic.
2. Defines a grid of candidate hyperparameter values for randomized search.
3. Returns the parameter space and training data for use in model tuning scripts.

Intended as a utility during the model optimization phase—enabling consistent,
modular construction of hyperparameter spaces.

Usage:
    Called from training pipelines (e.g., baseline_xgboost.py) to retrieve:
        - `param_dist`: hyperparameter search space
        - `X_train`, `y_train`: training data inputs

Dependencies:
    - numpy, joblib
    - Custom modules: data_split (xgboost_utils)
"""

import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "scripts" /"xgboost"
sys.path.insert(0, str(src_path))
from xgboost_utils import data_split
import numpy as np



def optimization(df, interval=15):
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
        'estimator__n_estimators': [100, 150],
        'estimator__max_depth': [2, 3],
        'estimator__learning_rate': [0.01, 0.05],
        'estimator__subsample': [0.8],
        'estimator__colsample_bytree': [0.8],
        'estimator__gamma': [1, 3],
        'estimator__reg_lambda': [10],
        'estimator__reg_alpha': [10],
}
    return param_dist, X_train, y_train
