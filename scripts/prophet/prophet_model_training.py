# save_best_prophet_model.py
# author: Tengwei Wang
# date: 2025-06-18

"""
Initializes and fits a Prophet model using the best hyperparameter configuration.

Responsibilities:
-----------------
1. Loads best hyperparameters from a saved JSON configuration.
2. Builds a Prophet model with holiday logic and applies it to training data.
3. Returns the fitted Prophet model object for further use.

Key Features:
-------------
- Supports modular workflows by separating training from serialization.
- Automatically applies best-tuned parameters stored from grid search.
- Integrates holiday effects into the Prophet forecaster.

Typical Usage:
--------------
Used after hyperparameter tuning to generate a trained model ready for saving, plotting, or forecasting.

"""


import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "scripts" / "prophet"
sys.path.insert(0, str(src_path))
from prophet_utils import create_model_new_holiday

def prophet_model_training(df, y_train, result):
    """
    Fit a Prophet model using best hyperparameters and return the trained model.

    Loads the best parameter configuration from a JSON file in the `result` directory,
    constructs a Prophet model with holiday effects, fits it on the provided data,
    and returns the trained model instance.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed training data with 'ds' and 'y' columns.
    y_train : pd.Series
        Original target values (used for building holiday-aware models).
    result : str
        Directory containing the best hyperparameter JSON file.

    Returns
    -------
    model : Prophet
        The trained Prophet model instance, ready for forecasting or saving.

    Example
    -------
    >>> model = prophet_model_training(df, y_train, "results/models/prophet")
    >>> forecast = model.predict(future_df)
    """
    model = create_model_new_holiday(y_train)
    model.fit(df)
    return model 

