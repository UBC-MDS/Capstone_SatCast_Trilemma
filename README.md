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

This repository provides forecasting pipelines to predict Bitcoin transaction fees using six different models, including classical time series, gradient boosting, and deep learning methods.  
Users can either **view the results** directly in our summary notebook or **run the scripts** to reproduce and extend the work.

### How to Use This Repository

You have three options to explore this project:

### Option 1: Quick Overview

To get a summary of all models, results, and visualizations:

**Run:**

```bash
jupyter lab analysis/model_overview.ipynb
```

This notebook compares all models in a unified view and links to individual model notebooks.  
If you wish to re-run this notebook **end-to-end**, you must first generate the SARIMA model file.

> ⚠️ **Note**: The SARIMA model (`sarima_final_model.pkl`) is too large for GitHub.  
> To regenerate it, run:

```bash
python scripts/baseline_sarima.py
```

### Option 2: Re-run All Models via One Script

To reproduce results fully with one script, run the full forecasting pipeline with one command:

```bash
python scripts/analysis.py
```

This will:

- Load and preprocess the dataset
- Run all six models: HWES, SARIMA, XGBoost, Prophet, DeepAR, TFT
- Save forecasts and metrics to `results/models/`, `results/tables/`, and `results/plots/`

### Option 3: Run Individual Models

Each model has its own Jupyter notebook and corresponding main script.

| Model   | Notebook                          | Script File                   |
| ------- | --------------------------------- | ----------------------------- |
| HWES    | `analysis/baseline_hwes.ipynb`    | `scripts/baseline_hwes.py`    |
| SARIMA  | `analysis/baseline_sarima.ipynb`  | `scripts/baseline_sarima.py`  |
| XGBoost | `analysis/baseline_xgboost.ipynb` | `scripts/baseline_xgboost.py` |
| Prophet | `analysis/advanced_prophet.ipynb` | `scripts/advanced_prophet.py` |
| DeepAR  | `analysis/advanced_deepar.ipynb`  | `scripts/advanced_deepar.py`  |
| TFT     | `analysis/advanced_tft.ipynb`     | `scripts/advanced_tft.py`     |

Example:

- Run each script like this:

```bash
python python scripts/<model>.py --parquet-path ./data/raw/sample_8_days.parquet
```

> Open the script to view its command-line arguments and customize inputs or outputs.
>
> `scripts/<model_name>/`: Contains each model’s helper scripts and training/inference logic.

- Run each notebook like this:

```bash
jupyter lab analysis/<model>.ipynb
```

### Window-Based Evaluation (Optional)

To run time-based window evaluations (e.g., expanding, sliding windows), check scripts under:

```bash
scripts/experimentation/
```

These contain additional experiments for HWES, SARIMA and XGBoost window performance.  
They are **not required** to reproduce the main results but provide deeper insights.
Usage are shown in each script.

Example:

```bash
python scripts/experimentation/hwes_window.py
```

> ⚠️ These scripts may take longer to run as they iterate across multiple time slices.

### Tests (Optional)

To run the function tests, enter the following in the root of the repository: 

``` bash
pytest
```

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
