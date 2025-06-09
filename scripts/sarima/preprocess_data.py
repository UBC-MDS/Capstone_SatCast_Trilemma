# preprocess_data.py
# author: Yajing Liu
# date: 2025-06-08

# Usage:
# python scripts/sarima/preprocess_data.py \
#     --data="./data/raw/mar_5_may_12.parquet" \
#     --data-to="./data/processed/sarima"

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
    Cleans and preprocesses raw Parquet data for SARIMA modeling,
    resamples to 15min intervals, interpolates, and saves as Parquet.
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
