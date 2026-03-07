# Predicting Household Appliance Energy Consumption

**MSIN0097 — Predictive Analytics Individual Coursework 2025–26**

## Project Overview

An end-to-end machine learning pipeline for predicting household appliance energy consumption (Wh) at 10-minute intervals. The project uses the UCI Appliances Energy Prediction dataset and follows a structured workflow: problem framing → EDA → data preparation → model selection → fine-tuning → final evaluation.

## Dataset

- **Source:** [UCI Machine Learning Repository — Appliances Energy Prediction](https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction)
- **File:** `energydata_complete.csv` (19,735 rows, 29 columns)
- **Period:** 11 January – 27 May 2016
- **Location:** Single low-energy house, Stambruges, Belgium
- **Sampling:** 10-minute intervals

Place `energydata_complete.csv` in the same directory as the notebook before running.

## Repository Structure

```
├── energy_prediction_steps1_3.ipynb   # Main notebook (Steps 1–6)
├── energydata_complete.csv            # Dataset (not included — see above)
├── requirements.txt                   # Python dependencies with pinned versions
├── README.md                          # This file
├── Energy_Prediction_Report_Final.docx # Report with appendix
```

## Environment Setup

**Python version:** 3.13.5

### Install dependencies

```bash
pip install -r requirements.txt
```

### Or using conda

```bash
conda create -n energy-pred python=3.13
conda activate energy-pred
pip install -r requirements.txt
```

## How to Run

1. Clone or download this repository
2. Place `energydata_complete.csv` in the project root directory
3. Install dependencies: `pip install -r requirements.txt`
4. Open the notebook:
   ```bash
   jupyter notebook energy_prediction_steps1_3.ipynb
   ```
5. Run all cells sequentially (Kernel → Restart & Run All)

**Important:** Cells must be run in order. Step 5 rebuilds the dataset from `energydata_complete.csv` with additional lag features, so the raw CSV must be accessible throughout.

## Pipeline Summary

| Step | Description | Key Output |
|------|-------------|------------|
| **Step 1** | Problem framing — define target, metrics, constraints | Regression on `Appliances` (Wh); R², MAE, RMSE |
| **Step 2** | Exploratory data analysis — distributions, temporal patterns, correlations | Right-skewed target; strong diurnal cycle; weak feature correlations |
| **Step 3** | Data preparation — feature engineering, chronological split, scaling | 32 features; 70/15/15 train/val/test; StandardScaler on train only |
| **Step 4** | Model selection — LR, RF, Gradient Boosting, MLP compared | All models weak without lag features (R² ≈ 0); diagnostic analysis |
| **Step 5** | Fine-tuning — lag features, grid search, ablation study | GB tuned R² = 0.59 (val), 0.57 (test); lag1 dominates |
| **Step 6** | Final solution — test evaluation, model card, error analysis | Production-ready GB model with documented caveats |

## Models Used

All models use **scikit-learn** only — no external deep learning frameworks required.

- **Linear Regression** — baseline
- **Random Forest** (`RandomForestRegressor`)
- **Gradient Boosting** (`GradientBoostingRegressor`) — final selected model
- **MLP Neural Network** (`MLPRegressor`)

## Final Results (Test Set)

| Model | R² | MAE (Wh) | RMSE (Wh) |
|-------|-----|----------|-----------|
| Linear Regression | 0.5972 | 26.84 | 57.74 |
| **Gradient Boosting (tuned)** | **0.5724** | **28.71** | **59.49** |
| MLP (tuned) | 0.4928 | 41.76 | 64.79 |

## Key Features

- **37 features** including 5 engineered lag/rolling features
- **Chronological split** — no temporal data leakage
- **Ablation study** — isolates contribution of individual lag features
- **Standardised evaluation** — same `evaluate_model()` function across all models

## Dependencies

| Package | Version |
|---------|---------|
| numpy | 2.4.2 |
| pandas | 3.0.1 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |
