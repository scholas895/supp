
# Malaria Forecasting with ST‑AWXE Ensemble

This repository contains code for predicting malaria case counts using a novel **Spatio‑Temporal Adaptive Weighted Ensemble (ST‑AWXE)**. The model combines four base learners (XGBoost, LightGBM, CatBoost, Random Forest) with adaptive instance‑wise weighting and a Ridge meta‑learner that incorporates country and year features.

## Dataset

The data (`malaria_fully_combined_filled.csv`) covers 48 African countries from 2000 to 2021. It includes:
- Cases (target) and deaths
- Intervention coverage (ITN)
- Incidence rate, infection prevalence, mortality rate
- Lag features (1‑ and 2‑year lags) and a 3‑year moving average

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
