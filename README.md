# Capstone_SatCast_Trilemma

Forecasting Bitcoin Transaction Fees using Time Series and Machine Learning  
📍UBC Master of Data Science – Capstone Project | In collaboration with Trilemma Capital

## Welcome!

Welcome to our capstone project! In the world of Bitcoin, transaction fees can spike dramatically due to network congestion. These spikes make it hard for users and services to estimate optimal fees.

**SatCast** aims to tackle this problem by predicting the "fastestFee" (sat/vByte) using real-time mempool data and advanced time series forecasting models.

## Motivation and Purpose

The Bitcoin network doesn't enforce fixed fees. Instead, users compete in an open auction for limited block space. As a result, fees can jump sharply during network congestion.

Our goals:

- Help users forecast transaction costs for the next 24 hours  
- Analyze congestion and fee dynamics using public mempool data  
- Evaluate both traditional and deep learning models for short-term forecasting  

## What Can You Learn from This Project?

This project helps you learn how to:

- Perform **time series EDA** on Bitcoin fees, mempool congestion, and block activity  
- Build **short-term forecasting models (15-min resolution)** using SARIMA, HWES, XGBoost, Prophet, and DeepAR  
- Evaluate model performance under both stable and volatile conditions  
- Create a **reproducible forecasting pipeline** using real-time mempool data

---
## Installation & Setup

1. Clone the repository

``` bash
git clone git@github.com:UBC-MDS/Capstone_SatCast_Trilemma.git
```

2. Create and activate the virtual environment

``` bash
conda env create -f environment.yml
conda activate satcast
```

3. (Optional) Open the terminal and run the following commands to render the report

``` bash
quarto render reports/proposal/proposal_report.qmd
quarto render reports/final/final_report.qmd
```

---
## Guidance to Run This Project

To reproduce the analysis and modeling results in this repository, follow the steps below. For each model, there is a modular notebook saved under `analysis` folder accompanies with logical pipelines in `.py` format under `scripts` folder.

### Step 1: Exploratory Data Analysis (EDA)

Run `data_spec.ipynb` to explore the dataset and gain insights:
   - Summarizes key feature groups, data types, and their roles  
   - Performs time series analysis to examine trends, seasonality, stationarity, and other time-based patterns

   ````bash
   jupyter lab analysis/data_spec.ipynb   
   ````

---
### Step 2: Baseline Forecasting Models

Baseline models include Holt-Winters, SARIMA, and XGBoost.


#### Model I: Holt-Winters Exponential Smoothing (HWES) 
   - `baseline_hwes.ipynb`
   - Captures trend and seasonality

Run the following command to evaluate the HWES model:

   ```bash
   jupyter lab analysis/baseline_hwes.ipynb
   python scripts/hwes_main.py
   ```


#### Model II: SARIMA (Seasonal Autoregressive Integrated Moving Average)
   - `baseline_sarima.ipynb`  
   - Seasonal order selected via ACF/PACF analysis

Run the following command to evaluate the HWES SARIMA model:

   ```bash
   jupyter lab analysis/baseline_sarima.ipynb
   python scripts/sarima_main.py
   ```
   **⚠️ Note:** The trained SARIMA model (`sarima_final_model.pkl`) is too large to be included in the repository.  
   If you wish to inspect the model or rerun the evaluation, please run `scripts/sarima_main.py` locally.


#### Model III: XGBoost
   - `baseline_xgboost.ipynb`  
   - Gradient boosting tree model with lagged features and rolling statistics

Run the following command to evaluate the XGBoost model:

   ```bash
   jupyter lab analysis/baseline_xgboost.ipynb
   python scripts/baseline_xgboost.py --data-path data/raw/mar_5_may_12.parquet --result results/models
   ```
(Optional) Run the following command to compare performance of sliding window and expanding window:

   ```bash
   python scripts/experimentation/xgboost_window.py --data-path data/raw/mar_5_may_12.parquet --result results/tables/xgboost
   ```


To implement the expanding/sliding window experiments for baseline models, please go to respective model subfolders under `scripts` and run respectively. 

**⚠️ Note:** These window experiments are not included in the main scripts. Please run the corresponding scripts individually if needed.

**❗IMPORTANT:** The running of these scripts will take a long time, as they are designed to evaluate the model performance over multiple time windows.


---
### Step 3: Advanced Models

Advanced models include Prophet, DeepAR, and Temporal Fusion Transformers (TFT).


#### Model IV: Prophet
   - `advanced_prophet.ipynb`  
   - Developed by Facebook, Prophet model improves HWES and SARIMA by handling holidays and special events
   - Multiplicative seasonal model

Run the following command to evaluate the XGBoost model:

   ```bash
   jupyter lab analysis/advanced_prophet.ipynb
   python scripts/advanced_prophet.py --df data/raw/mar_5_may_12.parquet --result results/models
   ```
**⚠️ Note:** Replace the path with your own `.parquet` file if needed.

#### Model V: DeepAR (Deep Autoregressive Model)
   - `advanced_deepar.ipynb`  
   - Developed by Amazon, probabilistic forecasting model using recurrent neural networks (RNNs)
   - Captures complex temporal patterns and uncertainty

Run the following command to evaluate the DeepAR model:

```bash
jupyter lab analysis/advanced_deepar.ipynb
python scripts/deepar.py --parquet_path ./data/raw/sample_8_days.parquet
```
**⚠️ Note:** Replace the path with your own `.parquet` file if needed.


#### Model VI. Temporal Fusion Transformer (TFT)
   - `advanced_tft.ipynb`
   - A state-of-the-art deep learning model for time series forecasting
   - Combines attention mechanisms with recurrent neural networks (RNNs)

Run the following command to evaluate the TFT model:

```bash
jupyter lab analysis/advanced_tft.ipynb
python scripts/advanced_tft.py --parquet_path ./data/raw/sample_8_days.parquet
```
**⚠️ Note:** Replace the path with your own `.parquet` file if needed.

---
## Contributing

We’d love for you to contribute to this project! Whether it’s adding new forecasting models, improving data pipelines, or fixing bugs, your input is valuable.  
Check out our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to get started.

If you are new to open source, look for issues labeled **"good first issue"** — these are great entry points to begin contributing!

## Support

Encountered a problem or have a question?  
Please open an [issue](https://github.com/UBC-MDS/Capstone_SatCast_Trilemma/issues) on this repository, and we’ll get back to you as soon as possible.

## Contributors

Jenny Zhang, Ximin Xu, Yajing Liu, Tengwei Wang

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE.md](LICENSE.md) file for full details.
