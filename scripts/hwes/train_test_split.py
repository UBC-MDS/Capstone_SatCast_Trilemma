# train_test_split.py
# author: Jenny Zhang
# date: 2024-06-08

# Usage: 
# python scripts/split_dataset.py \
#     --train-file ./data/processed/diabetes_train.csv \
#     --test-file ./data/processed/diabetes_test.csv \
#     --output-dir ./data/processed

import os
import click
import sys
import pandas as pd


sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.read_csv_data import read_csv_data
from src.save_csv_data import save_csv_data


@click.command()
@click.option('--preprocessed-data', type=str, default="./data/processed/hwes/preprocessed.csv", help="Path to the hwes processed data file")
@click.option('--data-to', type=str, default="./data/processed/hwes", help="Path to the directory where split data will be saved")
def main(preprocessed_data, data_to):
    """
    Processes separate train and test datasets, separates features and target variable, 
    and saves them as separate CSV files.

    Parameters:
    -----------
    train_file : str
        Path to the input CSV file containing the processed training dataset.
    test_file : str
        Path to the input CSV file containing the processed testing dataset.
    output_dir : str
        Directory where the resulting split datasets (X_train, y_train, X_test, y_test) will be saved.
    """
    # Load the processed datasets
    fee_series = read_csv_data(preprocessed_data)

    # Set forecast horizon
    # The last 2 days (48 hours) of data as test set 
    # The final two days one day is a normal day and the other is a spike day
    # At 15-minute intervals time series data, the last 192 rows for testing will be used
    forecast_horizon = 192  # 48 hours * (60 / 15 mins)
    
    # Perform temporal split
    train = fee_series[:-forecast_horizon]
    test = fee_series[-forecast_horizon:]

    # Save the split data as CSV files
    save_csv_data(train, os.path.join(data_to, 'train.csv'))
    save_csv_data(test, os.path.join(data_to, 'test.csv'))

if __name__ == '__main__':
    main()