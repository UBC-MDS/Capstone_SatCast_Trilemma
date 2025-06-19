# xgboost_model_training.py
# author: Tengwei Wang
# date: 2025-06-18

"""
xgboost_model_training.py

Fits and saves the best XGBoost model identified through randomized hyperparameter search.

Responsibilities:
-----------------
1. Performs model fitting on training data using `ForecastingHorizon`.
2. Extracts and serializes the best model from the random search.
3. Writes the best model to disk in `.pkl` format for future inference.

Key Features:
-------------
- Uses sktime-compatible `ForecastingHorizon` for multi-step predictions.
- Integrates seamlessly with a sktime `RandomizedSearchCV` object.
- Saves the trained model using `joblib`.

Typical Usage:
--------------
Used after hyperparameter search to finalize and persist the best-performing forecaster.
"""

import joblib
from pathlib import Path
import numpy as np
from sktime.forecasting.base import ForecastingHorizon

def train_and_save_best_model(random_search, X_train, y_train, result, interval=15):
    """
    Fit the random search model and save the best estimator.

    Parameters
    ----------
    random_search : RandomizedSearchCV
        Random search object returned from optimization.
    X_train : pd.DataFrame
        Training feature set.
    y_train : pd.Series
        Training target values.
    result : str
        Output directory to save the model.
    interval : int
        Forecast horizon size.
    
    Returns
    -------
    best_forecaster : sktime.forecasting model
        The best model selected from the search.
    """
    fh = ForecastingHorizon(np.arange(1, interval + 1), is_relative=True)
    random_search.fit(X=X_train, y=y_train, fh=fh)

    best_forecaster = random_search.best_forecaster_
    file_path = Path(result) / "xgboost.pkl"
    joblib.dump(best_forecaster, file_path)

    print("Best model saved to:", file_path)
    return best_forecaster
