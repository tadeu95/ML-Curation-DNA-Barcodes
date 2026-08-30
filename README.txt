# Machine learning for DNA barcode reference library curation

This repository contains the scripts documenting the analytical workflow of a pilot study evaluating rule-based and machine-learning approaches for supporting the curation of DNA barcode reference libraries.

The workflow includes data preparation, feature extraction, hyperparameter tuning, and final model evaluation for classifying records within discordant Barcode Index Numbers (BINs).

## Contents

- `01_data_preparation.R` — Data retrieval, preprocessing, feature extraction, and preparation of the final dataset.
- `02_ann_grid_search.py` — Hyperparameter search for ANN models.
- `03_xgboost_grid_search.py` — Hyperparameter search for XGBoost models.
- `04_logistic_grid_search.py` — Hyperparameter search for logistic regression models.
- `05_ann_final.py` — Final ANN evaluation.
- `06_logistic_final.py` — Final logistic regression evaluation.
- `07_xgboost_final.py` — Final XGBoost evaluation and permutation importance.
- `08_simple_rule.py` — Evaluation of the simple rule-based baseline.


