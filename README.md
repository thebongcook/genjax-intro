# GenJAX House Price Prediction

Bayesian linear regression for house price prediction using [GenJAX](https://github.com/ChiSym/genjax), a probabilistic programming library built on JAX.

## Overview

This project demonstrates uncertainty-aware house price prediction. Instead of point estimates, the model provides credible intervals for predictions, enabling better decision-making.

## Setup

1. Install dependencies with [uv](https://docs.astral.sh/uv/):
```bash
uv sync
```

2. Download the [Kaggle House Prices dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data) and place `train.csv` in the `data/` directory.

## Usage

```bash
uv run python house_price_genjax.py
```

## Features

- Bayesian linear regression with priors on coefficients
- Importance sampling inference with 1000 particles
- 70/30 train/test split for holdout evaluation
- Posterior uncertainty estimates with 95% credible intervals
- Test set metrics: MAE and 90% CI coverage

## Model

The model uses four features:
- `GrLivArea`: Above grade living area (sq ft)
- `OverallQual`: Overall material and finish quality
- `YearBuilt`: Original construction date
- `TotalBsmtSF`: Total basement square feet
