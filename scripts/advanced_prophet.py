# advanced_prophet.py
# author: Tengwei Wang
# date: 2025-06-18

# Create optimized prophet model and save it in desired path. 

import sys
from pathlib import Path
current_file = Path(__file__).resolve()
project_root = current_file.parents[1]
src_path = project_root / "scripts" / "prophet"
sys.path.insert(0, str(src_path))
from prophet_model_optimization import model_optimization
from prophet_data_preprocess import data_preprocess
from prophet_model_training import prophet_model_training
import click

@click.command()
@click.option('--df', type=str, help="Path to training data")
@click.option('--result', type=str, help="Path to save model")
def main(df,result):
    # Step 1: Preprocess raw data 
    df_processed, y_train = data_preprocess(df)

    # Step 2: Run optimization and write best hyperparameter JSON
    # model_optimization(df_processed, y_train, result)

    # Step 3: Fit and save the final model using the best params
    prophet_model_training(df_processed, y_train, result)

if __name__ == '__main__':
    main()