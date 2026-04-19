# Home Credit Default Risk

**Kaggle Competition** · Binary Classification · AUC-ROC

> Predicting whether a loan applicant will default, using behavioral and financial data from multiple relational tables.

---

## Overview

This project is a full end-to-end machine learning pipeline built for the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition. The goal is to predict the probability of loan default (`TARGET = 1`) for clients who may have little or no credit history, using alternative data sources.

The pipeline covers:
- Exploratory Data Analysis (EDA) with actionable insights
- Feature engineering across 7 relational tables
- LightGBM model with Optuna hyperparameter tuning
- 5-Fold Stratified Cross-Validation
- Experiment tracking with MLflow

A detailed technical report (LaTeX/PDF) is available in the [`report/`](report/) folder.

---

## Results

| Metric | Value |
|---|---|
| OOF AUC (5-fold) | *see training output* |
| Validation strategy | StratifiedKFold (n=5) |
| Model | LightGBM (GBDT) |
| Tuning | Optuna (20 trials × 3-fold) |

---

## Project Structure

```
├── notebooks/
│   ├── 01_EDA.ipynb                # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb # Feature engineering pipeline
│   └── 03_train_lightgbm.ipynb     # Model training, tuning, submission
├── src/
│   └── features.py                 # Reusable aggregation functions
├── scripts/
│   └── 01_download_data.py         # Kaggle API data download
├── report/
│   ├── report.tex                  # Full technical report (LaTeX)
│   └── report.pdf                  # Rendered PDF
├── data/                           # Raw CSVs (not versioned)
│   └── processed/                  # Processed parquets (not versioned)
├── requirements.txt
└── .env.example                    # API token template
```

---

## Methodology

### 1. Data & Problem

The dataset contains 7 tables with information about loan applications, previous credit history (bureau data), installment payments, credit card balances, and POS cash records. The main challenge is heavy class imbalance (~8% default rate) and a large proportion of missing values in key features.

### 2. Key EDA Insights

| Finding | Action |
|---|---|
| `DAYS_EMPLOYED == 365243` is an encoded NA | Replaced with NaN; added binary flag |
| `EXT_SOURCE_1/2/3` are the most predictive features | Combined via mean, std, product, pairwise interactions |
| Real estate block has >60% missing values | Dropped columns above 60% null threshold |
| 8% default rate | Used `scale_pos_weight` in LightGBM |

### 3. Feature Engineering

All auxiliary tables were aggregated to the `SK_ID_CURR` level and merged with the main application table:

- **Bureau + Bureau Balance** — credit history from other institutions; status flags (good/late payments)
- **Previous Applications** — approval ratios, days since last application
- **Installments Payments** — payment differences, days past due, on-time payment rate
- **Credit Card Balance** — credit utilization ratio
- **POS Cash Balance** — numeric and categorical aggregations

Manual features created:
- Financial ratios (`CREDIT_INCOME_RATIO`, `ANNUITY_CREDIT_RATIO`, etc.)
- Age and employment duration in years
- `EXT_SOURCE` combinations (mean, std, product, pairwise products)
- Document count (`DOCS_PROVIDED`)

### 4. Model

**LightGBM** (Gradient Boosted Decision Trees) with:
- Hyperparameter tuning via **Optuna** (TPE sampler, 20 trials, 3-fold CV)
- Final training with **5-Fold StratifiedKFold** to preserve class ratio
- Out-of-Fold (OOF) predictions to estimate generalization AUC
- `scale_pos_weight` to compensate for class imbalance
- Early stopping (100 rounds) on each fold

Tuned parameters: `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `bagging_freq`, `reg_alpha`, `reg_lambda`.

---

## Setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Valvitor/kaggle-HomeCreditDefaultRisk.git
cd home-credit-default-risk
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Kaggle API

```bash
cp .env.example .env
# Edit .env and add your KAGGLE_API_TOKEN
```

### 3. Download data

```bash
python scripts/01_download_data.py
```

### 4. Run notebooks in order

```
01_EDA.ipynb → 02_feature_engineering.ipynb → 03_train_lightgbm.ipynb
```

---

## Technologies

| Tool | Purpose |
|---|---|
| Python 3.10 | Core language |
| pandas / numpy | Data manipulation |
| LightGBM | Gradient boosting model |
| Optuna | Hyperparameter optimization |
| scikit-learn | Cross-validation, metrics |
| MLflow | Experiment tracking |
| pyarrow | Parquet I/O |
| matplotlib / seaborn | Visualization |

---

## Author

**Valvitor Santos**
Portfolio project — [GitHub](https://github.com/Valvitor)
