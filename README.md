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

4. (Optional) Open the terminal and run the following commands to render the [final report](reports/final/final_report.pdf)

``` bash
quarto render reports/final/final_report.qmd
```

## Repository Structure

Here is how this repository is organized:

| Folder / File          | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `analysis/`            | Jupyter notebooks for EDA, baseline models, and advanced models.  |
| `scripts/`             | Python scripts for running each model, data prep, and experimentation.|
| `data/`                | Contains raw and preprocessed data files and how to fetch data from API. |
| `results/`             | Stores generated plots, forecasts, and summary tables.                 |
| `reports/`             | Proposal and final report in Quarto format (rendered as PDF).     |
| `src/`                 | Modularized utility functions used across notebooks and scripts         |
| `tests/`               | Unit tests for utility functions in `src/`                              |
| `environment.yml`      | Conda environment configuration file.                                  |

#### Notes on `scripts/`

Each model has:

- A main script (e.g., `baseline_sarima.py`) to run from command line.

- A subfolder (e.g., `scripts/baseline_sarima/`) with helper functions for loading, preprocessing, and training.

Users typically run the main script; helper files are imported and not meant to be executed directly.

## Explore the Project: From Overview to Full Reproduction

This repository offers a layered approach to forecasting Bitcoin transaction fees using classical time series models, gradient boosting, and deep learning techniques.

Whether you're here for a quick summary or looking to reproduce everything from scratch, follow the three levels below — from light exploration to full model training.

### Quick Overview (Start Here)

If you want to get a summary of all models, results, and visualizations, start by opening the [comprehensive_overview.ipynb](analysis/comprehensive_overview.ipynb):

**Run:**

```bash
jupyterlab
```

After launching JupyterLab, navigate to `analysis/comprehensive_overview.ipynb` to explore the full project summary. (Note: This ensures all images render correctly, since they use relative paths.)

This notebook:

- Summarizes all models, forecasts, and key results

- Includes links to each model’s detailed notebook for deeper analysis

> ⚠️ **Note**: To run this notebook end-to-end, you must first generate the SARIMA model file (`sarima_final_model.pkl`), which is too large to store in the repo.
> To regenerate it, run:

```bash
python scripts/baseline_sarima.py --parquet-path data/raw/mar_5_may_12.parquet
```

### One-Command Forecast Generation

If you'd like to quickly reproduce all final forecasts using saved models:

```bash
python scripts/analysis.py
```

This will:

- Load and preprocess the dataset
- Use stored models (HWES, SARIMA, XGBoost, Prophet, DeepAR, TFT) to generate forecasts
- Save forecasts and metrics to `results/models/`, `results/tables/`, and `results/plots/` and no model training is triggered.

> ⚠️ **Note on Model Consistency**

To ensure `analysis.py` generates the intended final forecasts, make sure the models in `results/models/` match the officially trained versions stored in `results/saved_models/`.

If you've run other scripts (e.g., for tuning or ablation), the files in `results/models/` may have been overwritten.  
In that case, **please copy the official model files from `saved_models/` back into `results/models/`** before running:

```bash
python scripts/analysis.py
```

### Full Model Execution (Custom Training)

If you'd like to understand and customize how each model is trained:

- Use the corresponding Jupyter notebook or script

- Check the top of each script for command-line usage

**Exploratory Data Analysis (EDA)**  

- [analysis/data_spec.ipynb](analysis/data_spec.ipynb): Provides detailed EDA, including time coverage, seasonality, stationarity checks, and correlation analysis.

**Models**

| Model   | Notebook                          | Script File                   |
| ------- | --------------------------------- | ----------------------------- |
| HWES    | [analysis/baseline_hwes.ipynb](analysis/baseline_hwes.ipynb)    | [scripts/baseline_hwes.py](scripts/baseline_hwes.py)    |
| SARIMA  | [analysis/baseline_sarima.ipynb](analysis/baseline_sarima.ipynb)  | [scripts/baseline_sarima.py](scripts/baseline_sarima.py)  |
| XGBoost | [analysis/baseline_xgboost.ipynb](analysis/baseline_xgboost.ipynb) | [scripts/baseline_xgboost.py](scripts/baseline_xgboost.py) |
| Prophet | [analysis/advanced_prophet.ipynb](analysis/advanced_prophet.ipynb) | [scripts/advanced_prophet.py](scripts/advanced_prophet.py) |
| DeepAR  | [analysis/advanced_deepar.ipynb](analysis/advanced_deepar.ipynb)  | [scripts/advanced_deepar.py](scripts/advanced_deepar.py)  |
| TFT     | [analysis/advanced_tft.ipynb](analysis/advanced_tft.ipynb)     | [scripts/advanced_tft.py](scripts/advanced_tft.py)     |

### Notes on Running Individual Scripts and Notebooks

#### Scripts (`scripts/*.py`)

Each script is runnable from the command line and supports customizable arguments.  
You can find **usage instructions** at the top of every script file.

**Example (Run XGBoost with full data):**

```bash
python scripts/baseline_xgboost.py \
    --parquet-path data/raw/mar_5_may_12.parquet \
    --skip-optimization
```

- `--parquet-path` can be changed to point to your own data file.

- We provide a sample dataset (`data/raw/sample_8_days.parquet`) for quick testing, but:
  - It **should not** be used for models that require hyperparameter tuning (HWES, XGBoost, Prophet).

- For XGBoost and Prophet models that support skipping optimization, use `--skip-optimization` to load pre-tuned parameters. (Note: HWES script have optimization process but does not have the option `--skip-optimization` as it does not take very long to run.)

> The trained models will be saved at `results/models/temp_models`

**Reference Compute Setup and Runtime:**

| Model    | Full Run Time (est.) | Optimization Required | Notes                        |
|----------|----------------------|------------------------|------------------------------|
| HWES     | ~5 min               | ✅ Yes                 | Fastest                      |
| SARIMA   | ~5 min               | ❌ No                  | Fastest                      |
| XGBoost  | ~2 hrs               | ✅ Yes                 | Parallelizable               |
| Prophet  | ~3–4 hrs             | ✅ Yes                 | Sensitive to daily pattern   |
| DeepAR   | ~6 hrs               | ❌ No                  | GPU supported for CUDA     |
| TFT      | ~8–9 hrs             | ❌ No                  | Best run on GPU              |

> All benchmarks run on: `Intel i9-13980HX`, `RTX 4090 labtop GPU`, `Windows 11 Pro`.
>
> All scripts save outputs to `results/models/`, `results/plots/`, and `results/tables/`.

#### Notebooks (`analysis/*.ipynb`)

Each model also includes a notebook version for interactive exploration.

**Run in Jupyter Lab:**

```bash
jupyter lab analysis/<model>.ipynb
```

**Use notebooks if you want to:**

- Understand model logic step by step
- View intermediate outputs and plots
- Modify hyperparameters manually

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

| Model    | Script                      | Est. Runtime      |
|----------|-----------------------------|-------------------|
| SARIMA   | [sarima_window.py](scripts/experimentation/sarima_window.py)          | ~1.5 hours        |
| XGBoost  | [xgboost_window.py](scripts/experimentation/xgboost_window.py)         | ~1 hour           |
| HWES     | [hwes_window.py](scripts/experimentation/hwes_window.py)            | ~15 minutes       |

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
