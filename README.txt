This repository contains the scripts used in a pilot study evaluating artificial neural networks (ANNs) and XGBoost models for supporting the curation of DNA barcode reference libraries. The workflow includes data preparation, feature extraction, hyperparameter tuning, and final cross-validated model evaluation for classifying records within discordant Barcode Index Numbers (BINs).

Contents

01_feature_extraction.R — Mines records from BOLD and extracts data features.

02_ann_grid_search.py — Hyperparameter search for ANN models.

03_xgboost_grid_search.py — Hyperparameter search for XGBoost models.

04_ann_final_cv.py — Final 10-fold cross-validation evaluation of the selected ANN configuration.

05_xgboost_final_cv.py — Final 10-fold cross-validation evaluation of the selected XGBoost model.

Purpose

These scripts document the analytical workflow used to test whether machine-learning models can assist in identifying putatively supported vs. inconclusive records in discordant BINs.