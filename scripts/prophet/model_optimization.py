import sys
from prophet import Prophet
import pickle

from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "src" 
sys.path.insert(0, str(src_path))
from prophet_utils import model_optimization as optimization
import itertools
import json

def model_optimization(df,result):
    """
    Optimize to find the best hyperparameters. 

    Parameters:
    ----------
    df : str
        The path of training dataset.
    result: str
        The path to save the best model. 

    Returns:
    -------
    Null
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
    print(f"Best params：{best_params[0]}\nRMSE：{best_params[1]:.4f}")
    final_model = Prophet(**best_params[0])
    final_model.fit(df)

    # Save the fitted model
    with open(result + "/prophet_model.pkl", "wb") as f:
        pickle.dump(final_model, f)


