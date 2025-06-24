# Capstone_SatCast_Trilemma

Forecasting Bitcoin Transaction Fees using Time Series and Machine Learning  
📍[UBC Master of Data Science](https://masterdatascience.ubc.ca/) – Capstone Project | In collaboration with [Trilemma Foundation](https://www.trilemma.foundation/)

Special thanks to our mentor [Hedayat Zarkoob](https://www.linkedin.com/in/hedayat-zarkoob-6b218b106/) from the UBC MDS program for his invaluable guidance and support throughout this project.

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

3. (Optional) If Jupyter can't find your environment kernel, you may need to manually add it:

```bash
python -m ipykernel install --user --name=satcast
```

## Repository Structure

| Folder / File          | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `analysis/`            | Jupyter notebooks for EDA, modeling, and final results                  |
| `scripts/`             | High-level entry-point scripts to run full forecasting pipelines        |
| ├─`<model_name>/`      | Each subfolder contains helper functions to load, preprocess, and train |
| `src/`                 | Modularized utility functions used across notebooks and scripts         |
| `data/`                | Raw and processed data files; includes live API fetch script            |
| `results/`             | Saved plots, model outputs, and evaluation metrics                      |
| `reports/`             | Project proposal and final report (Quarto format)                       |
| `tests/`               | Unit tests for utility functions in `src/`                              |
| `environment.yml`      | Conda environment specification                                         |

## Start Here

If you're new to this project, start with one of the following:

1. **[Final Report (PDF)](reports/final/final_report.pdf)** — A complete summary of our goals, methodology, EDA, model results, and insights.

**Run:**

```bash
quarto render reports/final/final_report.qmd
```

2. **[Overview Notebook](analysis/comprehensive_overview.ipynb)** — A single Jupyter notebook that visualizes and compares all six forecasting models. 
 
**Run:**

```bash
jupyter lab analysis/comprehensive_overview.ipynb
```

   This is the most convenient way to understand model performance and evaluation results.

> These two files offer the quickest way to explore our project.

## How to Navigate This Project

We designed this section to support **two types of users**:

1. **Exploratory Users**: Those who want to **explore the data, visualizations, and model results** without rerunning code or configuring environments.
2. **Technical Developers**: Those who want to **reproduce results, run forecasting models**, or customize the pipeline using scripts and configuration options.

Feel free to start at the level that suits your needs — whether it's viewing a single notebook or re-running the full pipeline from scratch.

### For Exploratory Users

If you're mainly here to **review results**, **explore insights**, or **see the models in action**, you’re in the right place.

After reviewing the **Start Here** section (see above), we recommend exploring the following notebooks for more detail:

#### Notebooks You Can Explore

- [analysis/data_spec.ipynb](analysis/data_spec.ipynb): In-depth exploratory data analysis (e.g., trends, seasonality, stationarity).
- [analysis/baseline_hwes.ipynb](analysis/baseline_hwes.ipynb): Exponential smoothing baseline forecasting.
- [analysis/baseline_sarima.ipynb](analysis/baseline_sarima.ipynb): SARIMA baseline model and diagnostics.
- [analysis/baseline_xgboost.ipynb](analysis/baseline_xgboost.ipynb): Gradient boosting on tabular time features.
- [analysis/advanced_prophet.ipynb](analysis/advanced_prophet.ipynb): Facebook Prophet forecasting model.
- [analysis/advanced_deepar.ipynb](analysis/advanced_deepar.ipynb): DeepAR probabilistic forecasting using GluonTS.
- [analysis/advanced_tft.ipynb](analysis/advanced_tft.ipynb): Temporal Fusion Transformer for multivariate time series.

> All plots and metrics have been pre-saved in the `results/` folder. You can explore the notebooks directly without re-running them.  

⚠️ If you want to re-run `baseline_sarima.ipynb`, please first generate the SARIMA model file by running:

```bash
python scripts/baseline_sarima.py --parquet-path data/raw/mar_5_may_12.parquet
```

> The `sarima_final_model.pkl` file is too large to include in the repo.

### For Technical Developers

If you're looking to **reproduce results**, **train models**, or **extend the pipeline**, this section is for you.

We offer a modular setup that supports three levels of interaction:

#### 1. Run Pretrained Models for Fast Forecasting

You can skip training and use pretrained models to generate predictions([scripts/analysis.py](scripts/analysis.py)):

```bash
python scripts/analysis.py
```

This script:

- Load and preprocess the dataset
- Use stored models (HWES, SARIMA, XGBoost, Prophet, DeepAR, TFT) to generate forecasts
- Save forecasts and metrics to `results/models/`, `results/tables/`, and `results/plots/` and no model training is triggered.

⚠️ **Before running this script, make sure you generate the SARIMA model file manually:**

```bash
python scripts/baseline_sarima.py --parquet-path data/raw/mar_5_may_12.parquet
```

> The `sarima_final_model.pkl` file is too large to include in the repo and must be created locally.

#### 2. Train Individual Models

If you want to customize hyperparameters or train from scratch, you can run each model's main script:

| Model   | Script File                        |
|---------|------------------------------------|
| HWES    | [scripts/baseline_hwes.py](scripts/baseline_hwes.py)          |
| SARIMA  | [scripts/baseline_sarima.py](scripts/baseline_sarima.py)      |
| XGBoost | [scripts/baseline_xgboost.py](scripts/baseline_xgboost.py)    |
| Prophet | [scripts/advanced_prophet.py](scripts/advanced_prophet.py)    |
| DeepAR  | [scripts/advanced_deepar.py](scripts/advanced_deepar.py)      |
| TFT     | [scripts/advanced_tft.py](scripts/advanced_tft.py)            |

Each script has a `usage` section at the top that shows all command-line arguments.  
Common option include:

- `--skip-optimization`: Skip grid search (only applies to XGBoost and Prophet). HWES does not support this option, since it's hyperparameter tuning time is short.

Basic usage example:

```bash
python scripts/<model>.py --parquet-path data/raw/mar_5_may_12.parquet
```

> A sample dataset (`data/raw/sample_8_days.parquet`) is provided for quick testing.  
> But this is **not suitable** for HWES, XGBoost, or Prophet — these models require sufficient data to tune hyperparameters effectively.

Approximate training time on a high-end laptop (i9-13980HX CPU, RTX 4090 GPU):

| Model    | Full Data Runtime |
|----------|-------------------|
| HWES     | ~5 minutes        |
| SARIMA   | ~5 minutes        |
| XGBoost  | ~2 hours          |
| Prophet  | ~3–4 hours        |
| DeepAR   | ~6 hours          |
| TFT      | ~8–9 hours        |

### Window-Based Evaluation (Optional)

To evaluate model performance across time-based windows (e.g., expanding, reverse expanding, sliding), we provide additional scripts under:

```bash
scripts/experimentation/
```

These experiments are designed for deeper insight and are **not required** to reproduce the main results.

#### Available Modes

Each script supports the following time-based windowing strategies:

1. **Weekly Expanding Window**  
2. **Reverse Weekly Expanding Window**  
3. **Weekly Sliding Window**

Specify the mode using the `--mode` argument.

#### Example: HWES Window Evaluation

```bash
# Reverse weekly expanding
python scripts/experimentation/hwes_window.py \
  --parquet-path ./data/raw/mar_5_may_12.parquet \
  --mode reverse

# Weekly expanding
python scripts/experimentation/hwes_window.py \
  --parquet-path ./data/raw/mar_5_may_12.parquet \
  --mode expanding

# Weekly sliding
python scripts/experimentation/hwes_window.py \
  --parquet-path ./data/raw/mar_5_may_12.parquet \
  --mode sliding
```

> You can find similar scripts for SARIMA (`sarima_window.py`) and XGBoost (`xgboost_window.py`).

#### Runtime Estimates (Full Data)

| Model    | Script                                                | Est. Runtime  |
|----------|-------------------------------------------------------|---------------|
| HWES     | [hwes_window.py](scripts/experimentation/hwes_window.py)           | ~15 minutes   |
| XGBoost  | [xgboost_window.py](scripts/experimentation/xgboost_window.py)      | ~1 hour       |
| SARIMA   | [sarima_window.py](scripts/experimentation/sarima_window.py)       | ~1.5 hours    |

### Tests (Optional)

To run the function tests, enter the following in the root of the repository: 

``` bash
pytest
```

## Contributing

We’d love for you to contribute to this project! Whether it’s adding new forecasting models, improving data pipelines, or fixing bugs, your input is valuable.  
Check out our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to get started.

If you are new to open source, look for issues labeled **"good first issue"** — these are great entry points to begin contributing!

## Support

Encountered a problem or have a question?  
Please open an [issue](https://github.com/UBC-MDS/Capstone_SatCast_Trilemma/issues) on this repository, and we’ll get back to you as soon as possible.

## Contributors

[Jenny Zhang](https://github.com/JennyonOort), [Ximin Xu](https://github.com/davyxuximin), [Yajing Liu](https://github.com/yajing03), [Tengwei Wang](https://github.com/interestingtj)

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE.md](LICENSE.md) file for full details.
