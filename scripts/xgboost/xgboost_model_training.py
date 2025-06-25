# xgboost_random_search.py
# author: Ximin Xu
# date: 2025-06-18

"""
Script to perform random hyperparameter search for XGBoost-based time series forecasting
of Bitcoin transaction fees using full training data (no cross-validation).

This version performs CV using expanding windows defined by a specified number of folds.

Steps:
1. Divide the series into N expanding train/test folds.
2. For each hyperparameter sample, train and validate across all folds.
3. Use a custom loss function to evaluate performance.
4. Return the best-performing model, parameters, and score.

Dependencies:
    - sktime, xgboost, numpy, pandas, scikit-learn, json
"""

import xgboost as xgb
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import ParameterSampler
from sktime.forecasting.compose import make_reduction
import sys

# Set project root and import custom loss
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT / "src"))
from custom_loss_eval import custom_loss_eval

def build_random_search_cv(X, y, param_dist, n_iter=8, n_folds=5, horizon=96, optimize=True):
    """
    Perform random search with expanding-window cross-validation for XGBoost.

    Parameters
    ----------
    X : pd.DataFrame
        Exogenous features aligned with y.
    y : pd.Series
        Target time series.
    param_dist : dict
        Dictionary of hyperparameter distributions.
    n_iter : int
        Number of parameter configurations to sample.
    n_folds : int
        Number of CV folds (expanding window).
    horizon : int
        Forecast horizon in number of time steps.
    optimize : bool
        Whether to perform random search or load saved best parameters.

    Returns
    -------
    best_forecaster : sktime forecaster
        The best-performing model.
    best_params : dict
        Best hyperparameter configuration.
    best_score : float
        Mean custom loss across folds, or NaN if loaded.
    """

    # Directory to save best model parameters
    model_dir = PROJECT_ROOT / "results" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_score = float("inf")     # Lowest custom loss observed
    best_forecaster = None        # Best sktime forecaster object
    best_params = None            # Best hyperparameter config

    # === Fold Generator ===
    def get_folds_from_count(y, n_folds, horizon):
        """
        Create expanding-window train/test splits for CV.

        Parameters
        ----------
        y : pd.Series
            Target series.
        n_folds : int
            Number of folds to generate.
        horizon : int
            Forecast length for each fold.

        Returns
        -------
        folds : list of (train_idx, test_idx)
        """
        n = len(y)
        spacing = (n - horizon) // (n_folds + 1)
        folds = []

        for i in range(1, n_folds + 1):
            split = spacing * i
            train_idx = list(range(split))                     # expanding train set
            test_idx = list(range(split, split + horizon))     # next horizon steps
            folds.append((train_idx, test_idx))

        return folds

    # Create folds internally
    folds = get_folds_from_count(y, n_folds, horizon)

    if optimize:
        print(f"Running CV-based random search over {n_iter} configurations with {n_folds} folds...")

        # Sample parameter sets from search space
        param_samples = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))

        # Evaluate each sampled parameter set
        for i, params in enumerate(param_samples):
            print(f"\n[{i+1}/{n_iter}] Params: {params}")
            fold_scores = []

            try:
                # Loop through all folds for current parameter set
                for fold_idx, (train_idx, test_idx) in enumerate(folds):
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]

                    # Create XGBoost model with current hyperparameters
                    model = xgb.XGBRegressor(
                        objective="reg:squarederror",
                        tree_method="hist",
                        n_jobs=-1,  # use all cores
                        **{k.replace("estimator__", ""): v for k, v in params.items()}
                    )

                    # Wrap in sktime recursive forecaster
                    forecaster = make_reduction(
                        estimator=model,
                        window_length=96 * 7,  # use last 7 days to forecast next day
                        strategy="recursive"
                    )

                    # Train and forecast
                    forecaster.fit(y=y_train, X=X_train)
                    fh = np.arange(1, len(y_test) + 1)  # Forecast steps ahead
                    y_pred = forecaster.predict(fh=fh, X=X_test)

                    # Score with custom loss
                    score = custom_loss_eval(y_pred.values, y_test.values)
                    fold_scores.append(score)

                # Average across all folds
                avg_score = np.mean(fold_scores)
                print(f"Avg Custom Loss: {avg_score:.4f}")

                # Update best tracker if current score is lower
                if avg_score < best_score:
                    best_score = avg_score
                    best_params = params
                    best_forecaster = forecaster

            except Exception as e:
                print(f"Skipping due to error: {e}")
                continue

        # Save best config
        if best_params:
            with open(model_dir / "xgb_best_params_cv.json", "w") as f:
                json.dump(best_params, f, indent=4)
            print(f"Best params saved to: {model_dir / 'xgb_best_params_cv.json'}")
            return best_forecaster, best_params, best_score
        else:
            print("No valid model found. Falling back to saved parameters...")

    # === Fallback path: Load best params from file and fit to full data ===
    try:
        with open(model_dir / "xgb_best_params_cv.json", "r") as f:
            best_params = json.load(f)
        print("Loaded best params from file.")

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            tree_method="hist",
            n_jobs=-1,
            **{k.replace("estimator__", ""): v for k, v in best_params.items()}
        )

        forecaster = make_reduction(
            estimator=model,
            window_length=96 * 7,
            strategy="recursive"
        )
        forecaster.fit(y=y, X=X)
        print("Fitted model using loaded parameters.")
        return forecaster, best_params, np.nan

    except Exception as e:
        raise RuntimeError(f"Failed to load and fit model using saved parameters: {e}")
