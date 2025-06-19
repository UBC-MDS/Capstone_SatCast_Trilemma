"""
final_train_save_model.py

Trains a final SARIMA model on the full historical dataset (excluding the last spike day),
and saves the trained model for downstream inference or notebook analysis.

Workflow:
---------
1. Loads preprocessed 15-minute interval time series data.
2. Drops the final day to avoid spike contamination.
3. Applies log1p transformation to stabilize variance.
4. Fits a SARIMA model with defined seasonal order.
5. Saves the trained model object to a pickle file.

Key Features:
-------------
- No train-test splitting or cross-validation.
- No evaluation or plots — designed purely for final model preservation.
- Saved model can be used in downstream notebooks or deployment.

Typical Usage:
--------------
$ python scripts/sarima/final_train_save_model.py \
    --data="./data/processed/sarima/preprocessed_sarima_15min.parquet" \
    --model="./results/models/sarima_final_model.pkl"
"""


import click
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import ARIMA

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.save_model import save_model

@click.command()
@click.option('--data', type=str, default="./data/processed/sarima/preprocessed_sarima_15min.parquet", help="Path to preprocessed Parquet file")
@click.option('--model', type=str, default="./results/models/sarima_final_model.pkl", help="Path to save trained SARIMA model (Pickle)")
def main(data, model):
    """
    Train and save the final SARIMA model.

    Parameters
    ----------
    data : str
        Path to the preprocessed 15-minute interval Parquet file.
    model : str
        Destination path to save the trained SARIMA model as a pickle.

    Returns
    -------
    None
        The function trains the model and saves it to disk; it does not return anything.
    """

    # Load data
    df_resampled = pd.read_parquet(data)
    y = df_resampled['recommended_fee_fastestFee'].iloc[:-96]
    y = y.astype(float)

    # Split into train/test (last 1 day)
    y_train, y_test = temporal_train_test_split(y, test_size=96)

    # SARIMA model
    y_train_log = np.log1p(y_train)
    forecaster = ARIMA(order=(1, 0, 1), seasonal_order=(1, 0, 1, 96))
    forecaster.fit(y_train_log)

    # Save model
    save_model(forecaster, model)
    print(f"\n✅ SARIMA model saved to {model}")
    
if __name__ == '__main__':
    main()
