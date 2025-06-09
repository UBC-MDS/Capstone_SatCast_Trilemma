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
quarto render reports/proposal_report.qmd
```

## Guidance to Run This Project

To reproduce the analysis and modeling results in this repository, follow the steps below. Each notebook is modular but follows a logical pipeline.

### Step 1: Exploratory Data Analysis (EDA)

Run this to explore the dataset and gain insights:

3. `data_spec.ipynb`  
   - Summarizes key feature groups, data types, and their roles  
   - Performs time series analysis to examine trends, seasonality, stationarity, and other time-based patterns

### Step 2: Baseline Forecasting Models

Run and evaluate the following baseline models:

4. `baseline_hwes.ipynb`  
   - Holt-Winters Exponential Smoothing model  
   - Captures trend and seasonality

5. `baseline_sarima.ipynb`  
   - Seasonal ARIMA model  
   - Seasonal order selected via ACF/PACF analysis

#### 6. SARIMA

##### View Test Results

To evaluate the performance of the trained SARIMA model:

1. Open your terminal.
2. Activate the project environment.
3. Run the following command:

    ```bash
    python scripts/sarima_main.py
    ```

4. You will see:
    - Test predictions of the final SARIMA model on the second-to-last day (May 11)
    - Evaluation metrics (e.g., MAE, RMSE, MAPE, custom loss)
    - Forecast saved to `./results/tables/sarima/sarima_forecast.csv` for visualization

**⚠️ Note:** The trained SARIMA model (`sarima_final_model.pkl`) is too large to be included in the repository.  
If you wish to inspect the model or rerun the evaluation, please run `scripts/sarima_main.py` locally.

##### Train the Model (Window Experiments)

To run SARIMA expanding/sliding window experiments:

1. Open your terminal.
2. Activate the project environment.
3. Run the desired training script:

    ```bash
    python scripts/sarima/expanding_window_weekly_train.py
    ```

4. Similarly, you can run:

    ```bash
    python scripts/sarima/sliding_window_weekly_train.py
    python scripts/sarima/expanding_window_reverse_weekly_train.py
    ```

- Results will be saved to `./results/tables/sarima/`.

**⚠️ Note:** These window experiments are not included in `sarima_main.py`.  
Please run the corresponding scripts individually if needed.

7. `baseline_with_newdata.ipynb`  
   - Reruns baseline models on an extended date range (full dataset)

### Step 3: Advanced Models

8. `advanced_prophet.ipynb`  
    - Forecasting with Facebook Prophet  
    - Multiplicative seasonal model

#### 9. DeepAR

##### View Test Results

To evaluate the performance of the trained TFT model:

1. Open the Jupyter notebook:
   ```bash
   jupyter lab advanced_deepar.ipynb
   ```
2. Run all cells.
3. You will see:
   - Test predictions of the best saved model
   - Evaluation metrics (e.g., MAE, RMSE, MAPE, custom loss)
   - Forecast plots for visual inspection

##### Train the Model

To train the deepAR model using a sample dataset (or your own):

1. Open your terminal
2. Activate the project environment
3. Run the training script:
   ```bash
   python scripts/deepar.py --parquet_path ./data/raw/sample_8_days.parquet
   ```
   - Replace the path with your own `.parquet` file if needed.
  
#### 10. Temporal Fusion Transformer (TFT)

##### View Test Results

To evaluate the performance of the trained TFT model:

1. Open the Jupyter notebook:
   ```bash
   jupyter lab advanced_tft.ipynb
   ```
2. Run all cells.
3. You will see:
   - Test predictions of the best saved model
   - Evaluation metrics (e.g., MAE, RMSE, MAPE, custom loss)
   - Forecast plots for visual inspection

##### Train the Model

To train the TFT model using a sample dataset (or your own):

1. Open your terminal
2. Activate the project environment
3. Run the training script:
   ```bash
   python scripts/advanced_tft.py --parquet_path ./data/raw/sample_8_days.parquet
   ```
   - Replace the path with your own `.parquet` file if needed.
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
