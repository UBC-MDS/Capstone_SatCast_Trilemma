"""
Script: EDA.py
Location: scripts/
Description: Generates exploratory data analysis (EDA) plots.
             All visual outputs are saved to the img/ directory for downstream reporting.

Output: img/*.png (relative to project root)
"""

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import spearmanr
import seaborn as sns

# Set paths relative to script location
project_root = Path(__file__).resolve().parent.parent
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

# Multi-decomposition
output_path = project_root / "img" / "decomposition_multiplicative.png"
result = seasonal_decompose(df["recommended_fee_fastestFee"], model="multiplicative", period=288)
fig = result.plot()
fig.set_size_inches((10, 6))

# Format x-axis (weekly ticks, 'Jun 01' format)
for ax in fig.axes:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

# Label first and last subplot
fig.axes[0].set_ylabel("Distribution")
fig.axes[-1].set_xlabel("Date")
fig.suptitle("Fastest Fee (sats/vByte) — Multiplicative Decomposition (5-min)", y=1.02)

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")

# acf and pacf
output_path = project_root / "img" / "acf_pacf_plot.png"
# resample to 15min
df_15min = df['recommended_fee_fastestFee'].resample('15min').mean().dropna()
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot ACF
plot_acf(
    df_15min,  
    lags=96, 
    ax=axes[0],
    vlines_kwargs={'colors': 'C0', 'linestyles': '-', 'linewidth': 2},
    alpha=0.05
)
axes[0].set_ylim(-1, 1)
axes[0].set_title("Autocorrelation (ACF) — Lags up to 1 Day (96)")
axes[0].set_xlabel("Lag")
axes[0].set_ylabel("Correlation Coefficient")

# Plot PACF
plot_pacf(
    df_15min,  
    lags=96, 
    ax=axes[1],
    vlines_kwargs={'colors': 'C0', 'linestyles': '-', 'linewidth': 2},
    alpha=0.05
)
axes[1].set_ylim(-1, 1)
axes[1].set_title("Partial Autocorrelation (PACF) — Lags up to 1 Day (96)")
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("Correlation Coefficient")

plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")

# Spearman correlation heatmap
corr_features = [
    "recommended_fee_fastestFee",
    "mempool_total_fee",
    "mempool_count",
    "mempool_blocks_nTx",
    "mempool_blocks_blockVSize",
    "difficulty_adjustment_difficultyChange",
    "mempool_blocks_totalFees"
]

label_map = {
    "recommended_fee_fastestFee": "Fastest Fee",
    "mempool_total_fee": "Mempool Total Fee",
    "mempool_count": "Mempool Count",
    "mempool_blocks_nTx": "Next Block # of Transaction",
    "mempool_blocks_blockVSize": "Block vSize",
    "difficulty_adjustment_difficultyChange": "Difficulty Change",
    "mempool_blocks_totalFees": "Next Block Total Fee"
}

# spearman correlation
output_path = project_root / "img" / "spearman_correlation.png"
# Prepare correlation and p-value matrices
spearman_corr = pd.DataFrame(index=corr_features, columns=corr_features)
spearman_pval = pd.DataFrame(index=corr_features, columns=corr_features)

for i in corr_features:
    for j in corr_features:
        x = df[i].dropna()
        y = df[j].dropna()
        common_idx = x.index.intersection(y.index)
        if len(common_idx) > 2:
            # Spearman
            r_s, p_s = spearmanr(x[common_idx], y[common_idx])
            spearman_corr.loc[i, j] = r_s
            spearman_pval.loc[i, j] = p_s
            
# Convert to float and rename axes
spearman_corr = spearman_corr.astype(float).rename(index=label_map, columns=label_map)
spearman_pval = spearman_pval.astype(float).rename(index=label_map, columns=label_map)


plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr.astype(float), annot=True, cmap='coolwarm', center=0, square=True)
plt.xticks(rotation=45, ha='right')
plt.title("Spearman Correlation")
plt.tight_layout()
plt.savefig(output_path, dpi=300)
