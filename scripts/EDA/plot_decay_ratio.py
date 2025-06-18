"""
Script: plot_decay_ratio.py
Location: scripts/EDA/
Description: Computes PACF decay ratios for various resampling intervals
             and generates a bar plot to visualize short-term signal strength.

Output: img/optimal_interval.png (relative to project root)
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
from pathlib import Path

# Set paths relative to script location
project_root = Path(__file__).resolve().parent.parent.parent
data_path = project_root / "data" / "raw" / "mar_5_may_12.parquet"
output_path = project_root / "img" / "optimal_interval.png"

# Load data
df = pd.read_parquet(Path(project_root, 'data/raw/mar_5_may_12.parquet'))

# Convert 'timestamp' to datetime if it's not already
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.sort_values('timestamp', inplace=True)
df.set_index('timestamp', inplace=True)
df = df.iloc[1:]


# Define resampling intervals to evaluate
freqs = ['5min', '10min', '15min', '20min', '30min', '1h']
results = []

# Compute decay ratio from PACF
for freq in freqs:
    resampled = df['recommended_fee_fastestFee'].resample(freq).mean().dropna()
    pacf_vals = pacf(resampled, nlags=5, method='yw')

    if len(pacf_vals) >= 3:
        lag1 = pacf_vals[1]
        lag2 = pacf_vals[2]
        decay_ratio = abs(lag1) / abs(lag2) if lag2 != 0 else float('inf')
        results.append((freq, decay_ratio))

# Create dataframe for plotting
plot_df = pd.DataFrame(results, columns=["frequency", "decay_ratio"])

# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(plot_df["frequency"], plot_df["decay_ratio"], color="#4B8BBE")

# Add annotations
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5,
             f"{height:.2f}", ha='center', va='bottom', fontsize=10)

plt.title("Decay Ratio by Sampling Interval")
plt.xlabel("Sampling Interval")
plt.ylabel("Decay Ratio")
plt.tight_layout()

# Save to img folder
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()
