# xgboost_random_search.py
# author: Ximin Xu
# date: 2025-06-18

"""
Script to perform random hyperparameter search for XGBoost-based time series forecasting
of Bitcoin transaction fees using full training data (no cross-validation).

This script performs the following steps:
1. Defines a 24-hour forecast horizon using 15-minute interval steps.
2. Samples hyperparameter configurations from a user-defined search space.
3. Trains a recursive forecaster using `make_reduction` on each configuration.
4. Evaluates each model’s MAE over the last 24 hours of the training series.
5. Identifies and returns the best-performing model and parameters.
6. Saves the best parameter set to `results/models/xgb_best_params.json`.

Usage:
    Called from model selection scripts to pre-tune XGBoost before formal training.

Dependencies:
    - sktime, xgboost, numpy, pandas, scikit-learn, json
"""

import xgboost as xgb
import numpy as np
from sklearn.model_selection import ParameterSampler
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import make_reduction
from sktime.performance_metrics.forecasting import mean_absolute_error

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
def build_random_search_full(X_train, y_train, param_dist, interval=15, n_iter=8, optimize=True):
    """
    Perform random search or load best params to train final XGBoost model.

    If optimize=True, performs random search and saves best result.
    If optimize=False or optimization fails, loads saved params and fits model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Exogenous features aligned with y_train.
    y_train : pd.Series
        Target time series to forecast.
    param_dist : dict
        Dictionary of hyperparameter distributions.
    interval : int, default=15
        Time interval in minutes between observations.
    n_iter : int, default=8
        Number of samples in random search.
    optimize : bool, default=True
        Whether to perform optimization or just load and fit.

    Returns
    -------
    best_forecaster : sktime forecaster
    best_params : dict
    best_score : float (MAE), or np.nan if using fallback
    """
    steps_per_day = int(24 * 60 / interval)
    fh = ForecastingHorizon(np.arange(1, steps_per_day + 1), is_relative=True)
    model_dir = PROJECT_ROOT / "results" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    best_forecaster = None
    best_params = None

    if optimize:
        print(f"Running random search over {n_iter} configurations...")
        for i, params in enumerate(ParameterSampler(param_dist, n_iter=n_iter, random_state=42)):
            print(f"\n[{i+1}/{n_iter}] Testing params: {params}")
            try:
                params_clean = {k.replace("estimator__", ""): v for k, v in params.items()}
                model = xgb.XGBRegressor(
                    objective="reg:squarederror",
                    tree_method="hist",
                    n_jobs=-1,
                    **params_clean
                )
                forecaster = make_reduction(
                    estimator=model,
                    window_length=steps_per_day * 7,
                    strategy="recursive"
                )
                forecaster.fit(y=y_train, X=X_train)
                y_pred = forecaster.predict(fh=fh, X=X_train)
                y_true = y_train.iloc[-steps_per_day:]
                score = mean_absolute_error(y_true, y_pred)
                print(f"MAE: {score:.4f}")
                if score < best_score:
                    print("New best model found!")
                    best_score = score
                    best_forecaster = forecaster
                    best_params = params
            except Exception as e:
                print(f"Skipping due to error: {e}")
                continue

        if best_params is not None:
            # Save best parameters
            with open(model_dir / "xgb_best_params.json", "w") as f:
                json.dump(best_params, f, indent=4)
            print(f"Best parameters saved to: {model_dir / 'xgb_best_params.json'}")
            return best_forecaster, best_params, best_score
        else:
            print("Optimization failed. Falling back to loading saved parameters...")

    # Fallback: Load and fit using saved parameters
    try:
        with open(model_dir / "xgb_best_params.json", "r") as f:
            best_params = json.load(f)
        best_params_clean = {k.replace("estimator__", ""): v for k, v in best_params.items()}
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            **best_params_clean
        )
        forecaster = make_reduction(
            estimator=model,
            window_length=steps_per_day * 7,
            strategy="recursive"
        )
        forecaster.fit(y=y_train, X=X_train)
        print("Loaded and trained model using saved best parameters.")
        return forecaster, best_params, np.nan
    except Exception as e:
        raise RuntimeError(f"Failed to load and fit model using saved parameters: {e}")
