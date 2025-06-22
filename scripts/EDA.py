# EDA.py
# author: Yajing Liu
# date: 2025-06-18

"""
Script for exploratory data analysis (EDA) of Bitcoin
transaction-fee time-series.

This script performs the following steps:
1. Loads a raw or pre-processed Parquet/CSV file of 15-minute fees.
2. Generates time-series plots, histograms, and PACF/ACF diagnostics
   to uncover seasonal structure and volatility clusters.
3. Computes summary statistics (hourly means, intraday variance,
   spike counts) and anomaly flags useful for model design.
4. Explores resampling frequencies to quantify decay ratios in
   autocorrelation.
5. Saves all figures and CSV summaries to ``results/eda``.

Usage: 
    python scripts/EDA.py
"""

# === Imports ===
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import pacf
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.dates as mdates
from pathlib import Path
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.stats import spearmanr
import seaborn as sns

# === Setup Paths ===
# Define project-level root, input data path, and plot output path
project_root = Path(__file__).resolve().parent.parent
data_path = project_root / "data" / "raw" / "mar_5_may_12.parquet"
output_path = project_root / "results"/ "plots" / "optimal_interval.png"

# === Load and Preprocess Data ===
df = pd.read_parquet(data_path)

# Ensure timestamp is in datetime format and sort
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df.sort_values('timestamp', inplace=True)
df.set_index('timestamp', inplace=True)
df = df.iloc[1:]  # Drop first row if needed (e.g., corrupted or placeholder)

# === Decay Ratio by Sampling Interval ===
print("Generating decay ratio by interval plot...")

# Test several resampling intervals for PACF decay behavior
freqs = ['5min', '10min', '15min', '20min', '30min', '1h']
results = []

for freq in freqs:
    # Resample the target variable
    resampled = df['recommended_fee_fastestFee'].resample(freq).mean().dropna()
    pacf_vals = pacf(resampled, nlags=5, method='yw')  # Use Yule-Walker estimation

    # Compute decay ratio as |lag1| / |lag2| if valid
    if len(pacf_vals) >= 3:
        lag1 = pacf_vals[1]
        lag2 = pacf_vals[2]
        decay_ratio = abs(lag1) / abs(lag2) if lag2 != 0 else float('inf')
        results.append((freq, decay_ratio))

# Convert to DataFrame and plot
plot_df = pd.DataFrame(results, columns=["frequency", "decay_ratio"])

plt.figure(figsize=(8, 5))
bars = plt.bar(plot_df["frequency"], plot_df["decay_ratio"], color="#4B8BBE")

# Annotate bar heights
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{height:.2f}", ha='center', va='bottom', fontsize=10)

plt.title("Decay Ratio by Sampling Interval")
plt.xlabel("Sampling Interval")
plt.ylabel("Decay Ratio")
plt.tight_layout()

# Save the plot
print(f"Saving to {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

# === Distribution Plot of Fastest Fee ===
print("Plotting distribution of recommended_fee_fastestFee...")
output_path = project_root / "results"/ "plots"/ "fee_distribution.png"

# Drop zero or missing values
fee_series = df["recommended_fee_fastestFee"]
fee_series = fee_series[fee_series > 0].dropna()

plt.figure(figsize=(8, 5))
sns.histplot(fee_series, bins=50, kde=True, color='steelblue', edgecolor='black', linewidth=0.5)
plt.title("Distribution of Recommended Fee (Fastest)")
plt.xlabel("Fastest Fee (satoshis)")
plt.ylabel("Frequency")
plt.tight_layout()

print(f"Saving to {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

# === Seasonal Decomposition Plot ===
print("Performing multiplicative decomposition...")
output_path = project_root / "results"/ "plots"/ "decomposition_multiplicative.png"

# Decompose using a daily cycle (96 15-min intervals per day)
result = seasonal_decompose(df["recommended_fee_fastestFee"], model="multiplicative", period=288)
fig = result.plot()
fig.set_size_inches((10, 6))

# Format x-axis with weekly ticks
for ax in fig.axes:
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))

fig.axes[0].set_ylabel("Distribution")
fig.axes[-1].set_xlabel("Date")
fig.suptitle("Fastest Fee (sats/vByte) — Multiplicative Decomposition (5-min)", y=1.02)

plt.tight_layout()
print(f"Saving to {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches="tight")

# === Peak Detection with Summary Lines ===
print("Plotting time series with peak stats and thresholds...")
output_path = project_root / "results"/ "plots"/ "fee_peaks_summary.png"

s = df['recommended_fee_fastestFee'].dropna()
mean_val = s.mean()
median_val = s.median()
upper99 = s.quantile(0.99)

plt.figure(figsize=(12, 6))
plt.plot(s.index, s, color='dimgray', linewidth=1, label='fastestFee')

# Add summary lines
plt.axhline(mean_val, color='blue', linestyle='--', alpha=0.6, label=f'Mean ({mean_val:.1f})')
plt.axhline(median_val, color='green', linestyle='--', alpha=0.6, label=f'Median ({median_val:.1f})')
plt.axhline(upper99, color='red', linestyle='--', alpha=0.6, label=f'99th Percentile ({upper99:.1f})')

# Final formatting
plt.title('FastestFee Time Series with Mean, Median, and 99th Percentile')
plt.xlabel('Date')
plt.ylabel('Fee (sats/vByte)')
plt.legend(loc='upper left')
plt.tight_layout()

print(f"Saving to {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

# === ACF and PACF Plot ===
print("Plotting ACF and PACF...")
output_path = project_root / "results"/ "plots"/ "acf_pacf_plot.png"

# Re-aggregate to 15-min intervals
df_15min = df['recommended_fee_fastestFee'].resample('15min').mean().dropna()

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# ACF plot up to 96 lags (1 day)
plot_acf(df_15min, lags=96, ax=axes[0],
         vlines_kwargs={'colors': 'C0', 'linestyles': '-', 'linewidth': 2}, alpha=0.05)
axes[0].set_ylim(-1, 1)
axes[0].set_title("Autocorrelation (ACF) — Lags up to 1 Day (96)")
axes[0].set_xlabel("Lag")
axes[0].set_ylabel("Correlation Coefficient")

# PACF plot
plot_pacf(df_15min, lags=96, ax=axes[1],
          vlines_kwargs={'colors': 'C0', 'linestyles': '-', 'linewidth': 2}, alpha=0.05)
axes[1].set_ylim(-1, 1)
axes[1].set_title("Partial Autocorrelation (PACF) — Lags up to 1 Day (96)")
axes[1].set_xlabel("Lag")
axes[1].set_ylabel("Correlation Coefficient")

plt.tight_layout()
print(f"Saving to {output_path}")
plt.savefig(output_path, dpi=300, bbox_inches="tight")

# === Spearman Correlation Heatmap ===
print("Computing Spearman correlation heatmap...")
output_path = project_root / "results"/ "plots" / "spearman_correlation.png"

# Selected features for correlation analysis
corr_features = [
    "recommended_fee_fastestFee",
    "mempool_total_fee",
    "mempool_count",
    "mempool_blocks_nTx",
    "mempool_blocks_blockVSize",
    "difficulty_adjustment_difficultyChange",
    "mempool_blocks_totalFees"
]

# Map for nicer axis labels
label_map = {
    "recommended_fee_fastestFee": "Fastest Fee",
    "mempool_total_fee": "Mempool Total Fee",
    "mempool_count": "Mempool Count",
    "mempool_blocks_nTx": "Next Block # of Transaction",
    "mempool_blocks_blockVSize": "Block vSize",
    "difficulty_adjustment_difficultyChange": "Difficulty Change",
    "mempool_blocks_totalFees": "Next Block Total Fee"
}

# Initialize empty DataFrames for correlation and p-values
spearman_corr = pd.DataFrame(index=corr_features, columns=corr_features)
spearman_pval = pd.DataFrame(index=corr_features, columns=corr_features)

# Compute Spearman correlation for each pair
for i in corr_features:
    for j in corr_features:
        x = df[i].dropna()
        y = df[j].dropna()
        common_idx = x.index.intersection(y.index)
        if len(common_idx) > 2:
            r_s, p_s = spearmanr(x[common_idx], y[common_idx])
            spearman_corr.loc[i, j] = r_s
            spearman_pval.loc[i, j] = p_s

# Rename labels for plot
spearman_corr = spearman_corr.astype(float).rename(index=label_map, columns=label_map)
spearman_pval = spearman_pval.astype(float).rename(index=label_map, columns=label_map)

# Plot heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(spearman_corr, annot=True, cmap='coolwarm', center=0, square=True)
plt.xticks(rotation=45, ha='right')
plt.title("Spearman Correlation")
plt.tight_layout()

print(f"Saving to {output_path}")
plt.savefig(output_path, dpi=300)
