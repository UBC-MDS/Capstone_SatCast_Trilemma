# preprocess_data.py
# author: Yajing Liu
# date: 2025-06-18
"""
preprocess_data.py

Script to preprocess raw Bitcoin mempool Parquet data for SARIMA modeling.

This script performs the following steps:
1. Loads raw fee data from a Parquet file.
2. Resamples the time series to 15-minute intervals.
3. Fills missing values using time-based interpolation.
4. Saves the cleaned dataset to a target directory for downstream modeling.
"""


import click
import os
import sys
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.preprocess_raw_parquet import preprocess_raw_parquet

@click.command()
@click.option('--data', type=str, default="./data/raw/mar_5_may_12.parquet", help="Path to raw Parquet file")
@click.option('--data-to', type=str, default="./data/processed", help="Directory to save processed Parquet file")
def main(data, data_to):
    """
    Preprocesses raw Bitcoin mempool Parquet data for SARIMA modeling.

    This function loads raw transaction fee data from a Parquet file,
    resamples it to 15-minute intervals, interpolates missing values,
    and saves the cleaned time series to a specified output directory.

    Parameters
    ----------
    data : str
        Path to the raw Parquet file containing unprocessed time series data.
    data_to : str
        Destination directory where the preprocessed Parquet file will be saved.

    Output
    ------
    Saves a file named 'preprocessed_sarima_15min.parquet' in the specified output directory.
    """

    # Preprocess raw data
    df_resampled = preprocess_raw_parquet(data)

    # Save preprocessed Parquet
    output_path = os.path.join(data_to, "preprocessed_sarima_15min.parquet")
    os.makedirs(data_to, exist_ok=True)
    df_resampled.to_parquet(output_path)
    
    print(f"✅ Preprocessed data saved to {output_path}")

if __name__ == '__main__':
    main()
