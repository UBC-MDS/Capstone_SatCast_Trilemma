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
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "scripts" / "prophet"
sys.path.insert(0, str(src_path))
from prophet_utils import model_optimization as optimization,create_model_new_holiday
import itertools
import json
from prophet.serialize import model_to_json

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
    param_grid = {
        'changepoint_prior_scale': [0.01, 0.1, 0.3, 0.5],
        'seasonality_prior_scale': [5.0, 10.0, 20.0],
        'seasonality_mode': ['additive', 'multiplicative'],
        'changepoint_range': [0.8, 0.9, 0.95],
        'n_changepoints': [25, 50, 100]
    }
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    results = optimization(df, all_params)
    best_params = sorted(results, key=lambda x: x[1])[0]
    file_path = result+"/prophet_hyperparameter.json"
    with open(file_path, "w") as f:
        json.dump(best_params[0], f, indent=4)
    best_model = create_model_new_holiday(y_train)
    best_model.fit(df)
    model_path = result+"/prophet_model.json"
    with open(model_path, 'w') as fout:
        fout.write(model_to_json(best_model))  # Save model
    print(f"Best params：{best_params[0]}\nRMSE：{best_params[1]:.4f}")