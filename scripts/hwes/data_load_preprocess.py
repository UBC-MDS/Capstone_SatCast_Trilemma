# data_load_preprocess.py
# author: Jenny Zhang
# date: 2025-06-08

# Usage:
# python scripts/hwes/data_load_preprocess.py \
#     --data="./data/raw/mar_5_may_12.parquet" \
#     --data-to=./data/processed/hwes

import click
import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
from src.preprocess_raw_parquet import preprocess_raw_parquet
from src.save_csv_data import save_csv_data

@click.command()
@click.option('--data', type=str, default="./data/raw/mar_5_may_12.parquet", help="Path to file where raw data is saved")
@click.option('--data-to', type=str, default="./data/processed/hwes", help="Path to directory where processed data will be written to")
def main(data, data_to):
    """
    Cleans and preprocesses raw Parquet data, extracts the target time series,
    and saves the result for modeling.
    """
    
    # Raw data cleaning and resample
    df = preprocess_raw_parquet(data)
    
    # Validate no zero values in target
    zero_count = (df['recommended_fee_fastestFee'] == 0).sum()
    print(f"Number of zero values in 'recommended_fee_fastestFee': {zero_count}")

    # Extract the target series for modeling
    fee_series = df['recommended_fee_fastestFee'].astype(float).sort_index()
    fee_series.index.freq = pd.infer_freq(fee_series.index)

    # Save the preprocessed data
    save_csv_data(fee_series, os.path.join(data_to, 'preprocessed.csv'), index=True)

if __name__ == '__main__':
    main()