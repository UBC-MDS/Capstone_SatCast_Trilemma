# 🪙 SatCast: Forecasting Bitcoin Transaction Fees

📍[UBC Master of Data Science](https://masterdatascience.ubc.ca/) – Capstone Project | In partnership with [Trilemma Foundation](https://www.trilemma.foundation/)

Special thanks to our mentor [Hedayat Zarkoob](https://www.linkedin.com/in/hedayat-zarkoob-6b218b106/) from the UBC MDS program for his invaluable guidance and support throughout this project.

## 📚 Overview

Bitcoin transaction fees are unpredictable, often spiking without warning. Most existing tools offer only short-term estimates with limited foresight (<1 hour). 

This project tackles that problem by building a system to forecast Bitcoin transaction fees 24 hours ahead.

Our data product includes:

- Custom volatility-aware loss functions for improved evaluation of fee spikes
- Modular, end-to-end pipelines for six forecasting models: HWES, SARIMA, Prophet, XGBoost, DeepAR, and TFT
- Exploratory notebooks detailing data preparation, EDA, model training, and evaluation for each method
- A final report summarizing key findings and benchmarking model performance across metrics

Whether you're exploring volatility, comparing time series models, or forecasting blockchain costs, this repo offers a practical and extensible foundation.

## 📁 Repository Structure

| Folder / File          | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| `analysis/`            | Jupyter notebooks for overview, EDA, and walkthroughs of each model.   |
| `scripts/`             | Main training/evaluation scripts for each model, with helper functionsorganized in model-specific subfolders. |
| `data/`                | Contains raw data, processed data, and script for data extraction from API. |
| `results/`             | Stores generated plots, tables, and figures generated from scrpts.    |
| `reports/`             | Project proposal and final report in Quarto format (rendered as PDF).     |
| `src/`                 | Utility functions used across notebooks and scripts.         |
| `tests/`               | Unit tests for utility functions in `src/`.                              |
| `environment.yml`      | Conda environment configuration file.                                  |


## 💻 Installation & Setup

1. Clone the repository

  ``` bash
  git clone git@github.com:UBC-MDS/Capstone_SatCast_Trilemma.git
  ```

2. Create and activate the virtual environment

  ``` bash
  conda env create -f environment.yml
  conda activate satcast
  ```

3. (Optional) If Jupyter can't find your environment kernel, you may need to manually add it by running the following command in the terminal:

  ```bash
  python -m ipykernel install --user --name=satcast
  ```

## 🔰 Getting Started

If you're new to this project, we recommend starting with one of the following:

1. **[Final Report (PDF)](reports/final/final_report.pdf)** — A complete summary of our goals, methodology, EDA, model results, and insights.  
  
  To regenerate the report, you need to have [Quarto](https://quarto.org/docs/get-started/) installed. Then run the following command in terminal:  
  
  ```bash
  quarto render reports/final/final_report.qmd
  ```

2. **[Overview Notebook](analysis/comprehensive_overview.ipynb)** — An all-in-one Jupyter notebook that showcase the most important findings of the project.

  To open the notebook, you can use Jupyter Lab or Jupyter Notebook. Type the following command in terminal and then select the notebook from the Jupyter interface to ensure all images and links are rendered properly.

  ```bash
  jupyterlab
  ```

## 🛠️ Further Navigation and Exploration

This project is designed to be modular and user-friendly, allowing you to explore, run, and reproduce results at different levels of detail.

### First Things First

⚠️ Due to Github's file size limits, we **cannot** include the pretrained SARIMA model file (`sarima_final_model.pkl`) in the repository. Instead, you can generate it locally by training the model with the following command:

```bash
python scripts/baseline_sarima.py --parquet-path data/raw/mar_5_may_12.parquet
```

> **Note:** You must first generate `sarima_final_model.pkl` for any SARIMA-related scripts, notebooks, or `scripts/analysis.py` to run without errors.

### Exploratory Users

For those who prefer to engage with the project using minimal code while still gaining a comprehensive understanding of the data, models, and results, we recommend reviewing the notebooks in the `analysis/` folder. 

These are designed to emphasize reasoning, interpretation, and model logic over implementation details.

Please use the same command (`jupyterlab`) in the terminal to open the Jupyter Lab interface and the following table to navigate through the notebooks:

| Item   | Notebook                                   | Reading Time                  |
| ----------| -------------------------------------------------| -------------------------------|
| EDA    | [analysis/data_spec.ipynb](analysis/data_spec.ipynb)    | ~10-15 minutes    |
| HWES    | [analysis/baseline_hwes.ipynb](analysis/baseline_hwes.ipynb)    | ~5 minutes    |
| SARIMA  | [analysis/baseline_sarima.ipynb](analysis/baseline_sarima.ipynb)  | ~5 minutes    |
| XGBoost | [analysis/baseline_xgboost.ipynb](analysis/baseline_xgboost.ipynb) | ~5 minutes    |
| Prophet | [analysis/advanced_prophet.ipynb](analysis/advanced_prophet.ipynb) | ~5 minutes    |
| DeepAR  | [analysis/advanced_deepar.ipynb](analysis/advanced_deepar.ipynb)  | ~5 minutes    |
| TFT     | [analysis/advanced_tft.ipynb](analysis/advanced_tft.ipynb)     | ~5-10 minutes    |

> **Note:** You can also navigate to these notebooks directly from the comprehensive overview notebook (`analysis/comprehensive_overview.ipynb`), which includes inline links embedded throughout the summary.

### Technical Developers

If you're looking to reproduce results, train models, or extend the pipeline, this section is for you.

We offer a modular setup that supports three levels of interaction:

#### 1. Run Pretrained Models for Fast Forecasting

You can skip training and use pretrained models to generate predictions from [scripts/analysis.py](scripts/analysis.py) by running the following command in the terminal:

```bash
python scripts/analysis.py
```

#### 2. Train Individual Models

If you want to customize hyperparameters or train from scratch, you can run each model's main script:

| Model   | Script File                        | Training Time (est.) | Optimization Required | Time Saving Mechanism |
|---------|------------------------------------| --------------------------------|-----------------------|-----------------------|
| HWES    | [scripts/baseline_hwes.py](scripts/baseline_hwes.py) | ~5 minutes | ✅ Yes | Fast, not applicable |
| SARIMA  | [scripts/baseline_sarima.py](scripts/baseline_sarima.py) | ~5 minutes | ❌ No | Fast, not applicable |
| XGBoost | [scripts/baseline_xgboost.py](scripts/baseline_xgboost.py) | ~2 hours | ✅ Yes | Skip optimization |  
| Prophet | [scripts/advanced_prophet.py](scripts/advanced_prophet.py) | ~3–4 hours | ✅ Yes | Skip optimization |
| DeepAR  | [scripts/advanced_deepar.py](scripts/advanced_deepar.py) | ~6 hours | ❌ No | Sample Data Trial |
| TFT     | [scripts/advanced_tft.py](scripts/advanced_tft.py) | ~8–9 hours | ❌ No | Sample Data Trial |


⚠️ **Important Notes**:

- Training time is estimated based on a compute setup of `Intel i9-13980HX`, `RTX 4090 labtop GPU`, `Windows 11 Pro`. Actual time may vary depending on your hardware and configuration.

- Given the different configurations and arguments of each model, command-line options may differ depending on the model you are running. Therefore, **please refer to the top of each script file for detailed usage instructions**.

- For models that require hyperparameter tuning (HWES, XGBoost, Prophet), sample data `data/raw/sample_8_days.parquet` **cannot** be used as it is too small to capture the necessary patterns.

- For models that require long time to train, we have built in mechanisms to save time:
  - For **XGBoost** and **Prophet**, you can use the **`--skip-optimization`** flag to load pre-tuned hyperparameters. This will save time by skipping the optimization step and directly using the best hyperparameters saved during our best model training.
  - For **DeepAR** and **TFT**, we provided a sample dataset (`data/raw/sample_8_days.parquet`) that allows you to quickly test the model without waiting for long training times. However, for full training, you should use the larger dataset (`data/raw/mar_5_may_12.parquet`). This is controlled by the **`--parquet-path`** argument in the command.
  - Pay attention to the specific requirements and dependencies for each model, as outlined in their respective script files.
- If you have played around with the scripts, which may have modified files in `results/models` folder, and want to reset to default:
  - **please copy the official model files from `saved_models/` back into `results/models/`**
  - Re-run the analysis script (`scripts/analysis.py`) to ensure that all results are up-to-date
  - (Optional) Re-render the final report to reflect actual project findings


#### 3. Extensions to Pipelines

If you want to customize the training process or experiment with different configurations, you can modify the respective script files in the `scripts/` directory. Each model has its own script, and you can adjust preprocessing, training, and prediction as needed. For each model, you can find:

- A main script (e.g., `baseline_sarima.py`) in `scripts` folder that can be run from the command line.
- A subfolder (e.g., `scripts/sarima/`) with modularized helper functions for loading, preprocessing, training, forecasting, and evaluation.
- Users typically run the main script; helper files are imported and not meant to be executed directly.

### (Optional) Other Scripts

#### 1. Unit Tests for Utility Functions

We have included unit tests for the utility functions used by multiple models in the project. These tests can be found in the `tests/` directory and are designed to ensure the correctness of the utility functions.

To run the function tests, enter the following in the root of the repository: 

``` bash
pytest
```

#### 2. Window-Based Evaluation

We have provided additional scripts for evaluating model performance across different time-based windows (e.g., expanding, reverse expanding, sliding) for HWES, SARIMA, and XGBoost under the `scripts/experimentation/` directory.

These scripts are designed for deeper insight and are **not required** to reproduce the main results. They can be run independently to explore how models perform under different time-based conditions.

| Model    | Script                      | Est. Runtime      |
|----------|-----------------------------|-------------------|
| SARIMA   | [sarima_window.py](scripts/experimentation/sarima_window.py)          | ~1.5 hours        |
| XGBoost  | [xgboost_window.py](scripts/experimentation/xgboost_window.py)         | ~1 hour           |
| HWES     | [hwes_window.py](scripts/experimentation/hwes_window.py)            | ~15 minutes       |

⚠️ **Notes**:

- Runtime estimates are based on a standard compute setup and may vary based on your hardware.
- Please refer to the top of the specific script files for detailed usage instructions and available modes.
- The available arguments for each script include **`--parquet-path`** to specify the data file, and **`--mode`** to select the time-based windowing strategy (e.g., expanding, reverse expanding, sliding).
- These experiments can only be excuted on the full dataset (`data/raw/mar_5_may_12.parquet`) and are **not** compatible with the sample dataset (`data/raw/sample_8_days.parquet`).

## 🖇 Contributing

We’d love for you to contribute to this project! Whether it’s adding new forecasting models, improving data pipelines, or fixing bugs, your input is valuable.  
Check out our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to get started.

If you are new to open source, look for issues labeled **"good first issue"** — these are great entry points to begin contributing!

## 🤜 Support

Encountered a problem or have a question?  
Please open an [issue](https://github.com/UBC-MDS/Capstone_SatCast_Trilemma/issues) on this repository, and we’ll get back to you as soon as possible.

## 👥 Contributors

[Jenny Zhang](https://github.com/JennyonOort), [Ximin Xu](https://github.com/davyxuximin), [Yajing Liu](https://github.com/yajing03), [Tengwei Wang](https://github.com/interestingtj)

## 📜 License

This project is licensed under the **MIT License**.  
See the [LICENSE.md](LICENSE.md) file for full details.
