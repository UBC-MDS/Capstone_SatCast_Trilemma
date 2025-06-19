"""
preprocess_data.py

Preprocesses raw Bitcoin mempool Parquet data for SARIMA modeling.

Workflow:
---------
1. Loads raw data from a Parquet file.
2. Applies time-based resampling to 15-minute intervals.
3. Interpolates missing values and ensures datetime index.
4. Saves the cleaned data to disk as a new Parquet file.

Key Features:
-------------
- Designed for preparing data specifically for SARIMA model training.
- Modular usage with Click CLI: accepts input/output paths.
- Uses `preprocess_raw_parquet()` from the project's `src/` module.
- Saves output to `data/processed/preprocessed_sarima_15min.parquet` by default.

Typical Usage:
--------------
Run from the command line:

python scripts/sarima/preprocess_data.py \
    --data="./data/raw/mar_5_may_12.parquet" \
    --data-to="./data/processed"
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
