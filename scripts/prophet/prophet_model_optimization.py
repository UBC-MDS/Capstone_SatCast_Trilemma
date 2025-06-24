# prophet_model_optimization.py
# author: Tengwei Wang
# date: 2025-06-18

"""
prophet_model_optimization.py

Performs hyperparameter tuning for the Prophet model using grid search.

Responsibilities:
-----------------
1. Defines a grid of Prophet hyperparameters.
2. Evaluates each parameter set using cross-validation and RMSE.
3. Selects the best set and writes it to a JSON config file.

Key Features:
-------------
- Explores seasonality, changepoint, and trend configurations.
- Automates scoring and sorting based on forecast accuracy.
- Stores best hyperparameters for reproducible modeling.

Typical Usage:
--------------
Called within automated training workflows to search optimal Prophet configurations.
"""

import sys
from pathlib import Path
import itertools
import json

# Set up project paths for importing local modules
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "scripts" / "prophet"
sys.path.insert(0, str(src_path))

# Import the custom optimization function
from prophet_utils import model_optimization as optimization

def model_optimization(df, y_train, result):
    """
    Run grid search to optimize Prophet hyperparameters and save the best config.

    Evaluates candidate parameter combinations using RMSE and selects the best.
    Writes the best parameter set to a JSON file at the specified output path.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed training data with columns 'ds' and 'y'.
    y_train : pd.Series
        Original target series (used to construct model with holidays).
    result : str
        Output directory where best model config will be saved.

    Returns
    -------
    None

    Example
    -------
    >>> df, y_train = data_preprocess("data/processed/fee_data.parquet")
    >>> best_params, best_rmse = model_optimization(df, y_train, "results/models/prophet")
    >>> print(best_params)
    >>> print(f"Best RMSE: {best_rmse:.2f}")
    """

    # Define the grid of hyperparameters to search over
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.1, 0.3, 0.5],
        'seasonality_prior_scale': [5.0, 10.0, 20.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_range': [0.8, 0.9, 0.95],
        'n_changepoints': [25, 50, 100]
    }

    # Create all combinations of parameters from the grid
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]

    # Run model evaluation on each parameter combination and return results
    results = optimization(df,y_train, all_params)

    # Select the best parameter set (with lowest RMSE)
    best_params = sorted(results, key=lambda x: x[1])[0]

    # Define output path and save best hyperparameters to a JSON file
    file_path = result + "/prophet_best_params.json"
    with open(file_path, "w") as f:
        json.dump(best_params[0], f, indent=4)
