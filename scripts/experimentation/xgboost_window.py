
import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[2]
src_path = project_root / "scripts" /"xgboost"
sys.path.insert(0, str(src_path))
from xgboost_utils import evaluate_model
from xgboost_data_preprocess import data_preprocess
from sktime.forecasting.base import ForecastingHorizon
import json
import click
import numpy as np

@click.command()
@click.option('--data-path', type=str, help="Path to training data")
@click.option('--result', type=str, help="Path to save results")
def main(data_path,result):
    fh = ForecastingHorizon(np.arange(1, 97), is_relative=True)
    param_dist = {
        'estimator__n_estimators': [50, 100, 150],
        'estimator__max_depth': [1, 2, 3],
        'estimator__learning_rate': [0.01, 0.05, 0.1],
        'estimator__subsample': [0.6, 0.8, 0.9],
        'estimator__colsample_bytree': [0.6, 0.8, 0.9],
        'estimator__gamma': [1, 3, 5],
        'estimator__reg_lambda': [5, 10, 20],
        'estimator__reg_alpha': [5, 10, 20]
    }

    # sliding window
    df = data_preprocess(data_path)
    metrics_per_week, avg_metrics,y_test,y_pred = evaluate_model(df,param_dist,interval=15,weeks=10,fh=fh,sliding=1)
    # save the result
    with open(result+"/sliding_metrics_list.json", "w") as f:
        json.dump(metrics_per_week, f, indent=4)
    with open(result+"/sliding_avg_metrics_list.json", "w") as f:
        json.dump(avg_metrics, f, indent=4)

    # expanding window
    metrics_per_week, avg_metrics,y_test,y_pred = evaluate_model(df,param_dist,interval=15,weeks=10,fh=fh,sliding=0)
    # save the result
    with open(result+"/expanding_metrics_list.json", "w") as f:
        json.dump(metrics_per_week, f, indent=4)
    with open(result+"/expanding_avg_metrics_list.json", "w") as f:
        json.dump(avg_metrics, f, indent=4)
if __name__ == '__main__':
    main()
