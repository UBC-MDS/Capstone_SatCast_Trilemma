# save_best_prophet_model.py
# author: Tengwei Wang
# date: 2025-06-18

"""
prophet_model_training.py

Fits a Prophet model using the best hyperparameters and saves the trained model.

Responsibilities:
-----------------
1. Loads best hyperparameter configuration from JSON.
2. Initializes Prophet with holiday logic and fits it on training data.
3. Serializes the model to disk using Prophet's built-in format.

Key Features:
-------------
- Fully decouples tuning from model training.
- Integrates holiday effects using external utility.
- Ensures consistent model saving for inference and deployment.

Typical Usage:
--------------
Run after hyperparameter optimization to finalize and persist the best model.
"""


import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "scripts" / "prophet"
sys.path.insert(0, str(src_path))
from prophet.serialize import model_to_json
from prophet_utils import create_model_new_holiday

def prophet_model_training(df, y_train, result):
    """
    Fit a Prophet model using the best hyperparameters and save it to disk.

    Loads the best parameter configuration from a JSON file in `save_dir`,
    builds a Prophet model with holiday features, fits it on the given data,
    and writes the model as a JSON file.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed DataFrame with columns 'ds' and 'y'.
    y_train : pd.Series
        Original untransformed target values.
    save_dir : str
        Directory where the model will be saved.

    Returns
    -------
    None

    Example
    -------
    >>> df, y_train = data_preprocess("data/processed/fee_data.parquet")
    >>> save_best_model(df, y_train, "results/models/prophet")
    >>> # Output: results/models/prophet/prophet_model.json
    """
    model = create_model_new_holiday(y_train)
    model.fit(df)

    model_path = result + "/prophet_model.json"
    with open(model_path, 'w') as fout:
        fout.write(model_to_json(model))
