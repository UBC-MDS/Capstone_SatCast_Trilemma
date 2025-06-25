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
        'estimator__n_estimators': [50, 100, 150, 200],              # Number of boosting rounds
        'estimator__max_depth': [2, 3, 4, 5, 6],                     # Tree depth
        'estimator__learning_rate': [0.005, 0.01, 0.05, 0.1],        # Step size shrinkage
        'estimator__subsample': [0.6, 0.8, 1.0],                     # Row sampling
        'estimator__colsample_bytree': [0.6, 0.8, 1.0],              # Feature sampling
        'estimator__gamma': [0, 1, 3, 5],                            # Min loss reduction to split
        'estimator__reg_lambda': [1, 5, 10, 20],                     # L2 regularization
        'estimator__reg_alpha': [0, 1, 5, 10],                       # L1 regularization
        'estimator__min_child_weight': [1, 3, 5],                    # Min sum hessian in leaf
        'estimator__max_delta_step': [0, 1, 3],                      # Step constraint (often 0)
    }
    return param_dist, X_train, y_train
