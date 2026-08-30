# -*- coding: utf-8 -*-

"""
XGBoost grid search with fold-specific hyperparameter selection.
"""

import os
import json
import random
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
N_SPLITS = 10
VALIDATION_FRACTION = 0.20
CLASSIFICATION_THRESHOLD = 0.50


# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.abspath(
    os.path.join(
        SCRIPT_DIR,
        ".."
    )
)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "labelled_for_nn_ready.tsv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "xgboost_grid_search"
)

CURVES_DIR = os.path.join(
    OUTPUT_DIR,
    "learning_curves"
)

ANN_TEST_BINS_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "ann_grid_search",
    "test_bins_by_fold.tsv"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    CURVES_DIR,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    sep="\t"
)

group_column = "BIN_nn"
id_column = "processid"
target_column = "ground_truth_label"

X_full = df.drop(
    columns=[
        target_column,
        id_column,
        group_column
    ]
)

y_full = df[target_column].values
groups = df[group_column].values


# ============================================================
# FEATURE GROUPS
# ============================================================

percent_cols = [
    "percent_of_bin_records_belonging_to_species_nn",
    "genus_prop_in_bin_nn"
]

count_cols = [
    "frequency_species_nn",
    "total_records_in_bin_nn",
    "species_records_in_bin_nn",
    "unique_identifiers_nn",
    "unique_institutions_nn",
    "species_per_bin_nn",
    "bin_per_species_nn"
]

entropy_cols = [
    "shannon_entropy_nn"
]


# ============================================================
# HYPERPARAMETER GRID
# ============================================================

param_grid = {

    "n_estimators": [
        75,
        100,
        125,
        150
    ],

    "max_depth": [
        3,
        4,
        5
    ],

    "learning_rate": [
        0.01,
        0.05,
        0.1
    ],

    "subsample": [
        0.8,
        1.0
    ],

    "colsample_bytree": [
        0.8,
        1.0
    ],

    "min_child_weight": [
        1,
        5
    ]
}


all_combinations = [
    dict(
        zip(
            param_grid.keys(),
            values
        )
    )
    for values in itertools.product(
        *param_grid.values()
    )
]

N_CONFIGS = len(
    all_combinations
)

print(
    f"\nTotal XGBoost configurations: "
    f"{N_CONFIGS}"
)

assert N_CONFIGS == 288, (
    f"Expected 288 configurations, "
    f"found {N_CONFIGS}."
)


# ============================================================
# OUTER GROUPED CROSS-VALIDATION
# ============================================================

outer_cv = GroupKFold(
    n_splits=N_SPLITS
)

outer_splits = list(
    outer_cv.split(
        X_full,
        y_full,
        groups
    )
)


# ============================================================
# SAVE TEST BINS
# ============================================================

test_bins_all_folds = []

for fold_index, (
    _,
    test_idx
) in enumerate(
    outer_splits,
    start=1
):

    test_bins = np.unique(
        groups[test_idx]
    )

    fold_bins = pd.DataFrame({
        "fold": fold_index,
        "BIN": test_bins
    })

    test_bins_all_folds.append(
        fold_bins
    )


test_bins_df = pd.concat(
    test_bins_all_folds,
    ignore_index=True
)

test_bins_output_path = os.path.join(
    OUTPUT_DIR,
    "test_bins_by_fold.tsv"
)

test_bins_df.to_csv(
    test_bins_output_path,
    sep="\t",
    index=False
)


# ============================================================
# VERIFY SAME OUTER FOLDS AS ANN
# ============================================================

if os.path.exists(
    ANN_TEST_BINS_PATH
):

    ann_test_bins_df = pd.read_csv(
        ANN_TEST_BINS_PATH,
        sep="\t"
    )

    ann_test_bins_df = (
        ann_test_bins_df
        .sort_values(
            by=[
                "fold",
                "BIN"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    xgb_test_bins_df = (
        test_bins_df
        .sort_values(
            by=[
                "fold",
                "BIN"
            ]
        )
        .reset_index(
            drop=True
        )
    )

    pd.testing.assert_frame_equal(
        ann_test_bins_df,
        xgb_test_bins_df,
        check_dtype=False
    )

    print(
        "\nXGBoost outer test BINs exactly "
        "match the ANN outer test BINs."
    )


# ============================================================
# GRID SEARCH
# ============================================================

fold_results = []


for config_number, params in enumerate(
    all_combinations,
    start=1
):

    print(
        f"\nConfiguration "
        f"{config_number}/{N_CONFIGS}"
    )

    print(params)

    fold_train_losses = []
    fold_val_losses = []


    for fold_index, (
        train_val_idx,
        test_idx
    ) in enumerate(
        outer_splits,
        start=1
    ):

        print(
            f"  Fold "
            f"{fold_index}/{N_SPLITS}"
        )


        # ----------------------------------------------------
        # OUTER TRAINING POOL AND TEST BINS
        # ----------------------------------------------------

        X_pool = X_full.iloc[
            train_val_idx
        ].copy()

        y_pool = y_full[
            train_val_idx
        ]

        pool_groups = groups[
            train_val_idx
        ]

        test_groups = groups[
            test_idx
        ]


        # ----------------------------------------------------
        # INTERNAL GROUPED VALIDATION SPLIT
        # ----------------------------------------------------

        rng = np.random.RandomState(
            SEED + (fold_index - 1)
        )

        unique_bins = np.unique(
            pool_groups
        )

        rng.shuffle(
            unique_bins
        )

        n_validation_bins = int(
            VALIDATION_FRACTION *
            len(unique_bins)
        )

        validation_bins = unique_bins[
            :n_validation_bins
        ]

        validation_mask = np.isin(
            pool_groups,
            validation_bins
        )

        training_mask = ~validation_mask


        X_train = X_pool.loc[
            training_mask
        ].copy()

        y_train = y_pool[
            training_mask
        ]

        X_val = X_pool.loc[
            validation_mask
        ].copy()

        y_val = y_pool[
            validation_mask
        ]


        # ----------------------------------------------------
        # VERIFY NO BIN OVERLAP
        # ----------------------------------------------------

        train_bins = set(
            pool_groups[
                training_mask
            ]
        )

        val_bins = set(
            pool_groups[
                validation_mask
            ]
        )

        outer_test_bins = set(
            test_groups
        )

        assert not (
            train_bins & val_bins
        ), (
            f"Train/validation BIN overlap "
            f"in fold {fold_index}."
        )

        assert not (
            train_bins & outer_test_bins
        ), (
            f"Train/test BIN overlap "
            f"in fold {fold_index}."
        )

        assert not (
            val_bins & outer_test_bins
        ), (
            f"Validation/test BIN overlap "
            f"in fold {fold_index}."
        )


        # ====================================================
        # PREPROCESSING
        # ====================================================

        scaler_percent = StandardScaler()
        scaler_entropy = StandardScaler()


        # Percentage-based predictors
        X_train[percent_cols] = (
            scaler_percent.fit_transform(
                X_train[
                    percent_cols
                ]
            )
        )

        X_val[percent_cols] = (
            scaler_percent.transform(
                X_val[
                    percent_cols
                ]
            )
        )


        # Shannon entropy
        X_train[entropy_cols] = (
            scaler_entropy.fit_transform(
                X_train[
                    entropy_cols
                ]
            )
        )

        X_val[entropy_cols] = (
            scaler_entropy.transform(
                X_val[
                    entropy_cols
                ]
            )
        )


        # Count-based predictors
        X_train[count_cols] = np.log1p(
            X_train[
                count_cols
            ]
        )

        X_val[count_cols] = np.log1p(
            X_val[
                count_cols
            ]
        )


        # ====================================================
        # XGBOOST MODEL
        # ====================================================

        model = xgb.XGBClassifier(

            objective="binary:logistic",

            use_label_encoder=False,

            seed=SEED,

            eval_metric="logloss",

            **params
        )


        # ====================================================
        # TRAIN MODEL
        # ====================================================

        model.fit(

            X_train,
            y_train,

            eval_set=[
                (
                    X_train,
                    y_train
                ),
                (
                    X_val,
                    y_val
                )
            ],

            verbose=False
        )


        # ====================================================
        # LEARNING CURVES
        # ====================================================

        evals_result = (
            model.evals_result()
        )

        train_loss = (
            evals_result[
                "validation_0"
            ][
                "logloss"
            ]
        )

        val_loss = (
            evals_result[
                "validation_1"
            ][
                "logloss"
            ]
        )

        fold_train_losses.append(
            train_loss
        )

        fold_val_losses.append(
            val_loss
        )


        fold_curve_data = {

            "train_logloss":
                train_loss,

            "val_logloss":
                val_loss
        }


        fold_curve_path = os.path.join(

            CURVES_DIR,

            (
                f"config{config_number}_"
                f"fold{fold_index}_"
                f"learning_curve.json"
            )
        )


        with open(
            fold_curve_path,
            "w"
        ) as file:

            json.dump(
                fold_curve_data,
                file
            )


        # ====================================================
        # VALIDATION PREDICTIONS
        # ====================================================

        y_val_probability = (
            model.predict_proba(
                X_val
            )[:, 1]
        )

        y_val_pred = (
            y_val_probability >
            CLASSIFICATION_THRESHOLD
        ).astype(int)


        # ====================================================
        # VALIDATION METRICS
        # ====================================================

        f1_per_class = f1_score(
            y_val,
            y_val_pred,
            average=None,
            zero_division=0
        )

        precision_per_class = precision_score(
            y_val,
            y_val_pred,
            average=None,
            zero_division=0
        )

        recall_per_class = recall_score(
            y_val,
            y_val_pred,
            average=None,
            zero_division=0
        )


        fold_results.append({

            "fold":
                fold_index,

            "config_number":
                config_number,

            "n_estimators":
                params[
                    "n_estimators"
                ],

            "max_depth":
                params[
                    "max_depth"
                ],

            "learning_rate":
                params[
                    "learning_rate"
                ],

            "subsample":
                params[
                    "subsample"
                ],

            "colsample_bytree":
                params[
                    "colsample_bytree"
                ],

            "min_child_weight":
                params[
                    "min_child_weight"
                ],

            "f1_macro":
                f1_score(
                    y_val,
                    y_val_pred,
                    average="macro",
                    zero_division=0
                ),

            "f1_weighted":
                f1_score(
                    y_val,
                    y_val_pred,
                    average="weighted",
                    zero_division=0
                ),

            "accuracy":
                accuracy_score(
                    y_val,
                    y_val_pred
                ),

            "precision_macro":
                precision_score(
                    y_val,
                    y_val_pred,
                    average="macro",
                    zero_division=0
                ),

            "recall_macro":
                recall_score(
                    y_val,
                    y_val_pred,
                    average="macro",
                    zero_division=0
                ),

            "precision_weighted":
                precision_score(
                    y_val,
                    y_val_pred,
                    average="weighted",
                    zero_division=0
                ),

            "recall_weighted":
                recall_score(
                    y_val,
                    y_val_pred,
                    average="weighted",
                    zero_division=0
                ),

            "f1_class_0":
                f1_per_class[0],

            "f1_class_1":
                f1_per_class[1],

            "precision_class_0":
                precision_per_class[0],

            "precision_class_1":
                precision_per_class[1],

            "recall_class_0":
                recall_per_class[0],

            "recall_class_1":
                recall_per_class[1]
        })


    # ========================================================
    # AVERAGE LEARNING CURVE ACROSS OUTER FOLDS
    # ========================================================

    min_len = min(
        map(
            len,
            fold_val_losses
        )
    )


    avg_train_loss = np.mean(

        [
            loss[:min_len]
            for loss in
            fold_train_losses
        ],

        axis=0
    )


    avg_val_loss = np.mean(

        [
            loss[:min_len]
            for loss in
            fold_val_losses
        ],

        axis=0
    )


    # --------------------------------------------------------
    # SAVE AVERAGE CURVE DATA
    # --------------------------------------------------------

    average_curve_data = {

        "train_logloss":
            avg_train_loss.tolist(),

        "val_logloss":
            avg_val_loss.tolist()
    }


    average_curve_path = os.path.join(

        CURVES_DIR,

        (
            f"config{config_number}_"
            f"avg_learning_curve.json"
        )
    )


    with open(
        average_curve_path,
        "w"
    ) as file:

        json.dump(
            average_curve_data,
            file
        )


    # --------------------------------------------------------
    # SAVE AVERAGE CURVE FIGURE
    # --------------------------------------------------------

    plt.figure()

    plt.plot(
        avg_train_loss,
        label="Training logloss"
    )

    plt.plot(
        avg_val_loss,
        label="Validation logloss"
    )

    plt.title(
        f"Average logloss - "
        f"configuration {config_number}"
    )

    plt.xlabel(
        "Boosting round"
    )

    plt.ylabel(
        "Logloss"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CURVES_DIR,
            (
                f"config{config_number}_"
                f"avg_learning_curve.png"
            )
        ),
        dpi=300
    )

    plt.close()


# ============================================================
# SAVE COMPLETE GRID SEARCH
# ============================================================

fold_results_df = pd.DataFrame(
    fold_results
)


expected_rows = (
    N_CONFIGS *
    N_SPLITS
)

assert len(
    fold_results_df
) == expected_rows, (
    f"Expected {expected_rows} grid-search rows, "
    f"found {len(fold_results_df)}."
)


duplicate_count = (
    fold_results_df
    .duplicated(
        subset=[
            "config_number",
            "fold"
        ]
    )
    .sum()
)

assert duplicate_count == 0, (
    "Duplicate configuration/fold "
    "combinations detected."
)


configs_per_fold = (
    fold_results_df
    .groupby(
        "fold"
    )[
        "config_number"
    ]
    .nunique()
)

assert (
    configs_per_fold ==
    N_CONFIGS
).all(), (
    "Not all 288 configurations were "
    "evaluated in every outer fold."
)


grid_output_path = os.path.join(
    OUTPUT_DIR,
    "xgb_grid_search_by_fold_1_to_288.tsv"
)

fold_results_df.to_csv(
    grid_output_path,
    sep="\t",
    index=False
)


# ============================================================
# SELECT BEST CONFIGURATION WITHIN EACH OUTER FOLD
# ============================================================

# Highest validation macro-F1 is selected.
# Exact ties are resolved by retaining the configuration
# appearing first in the predefined grid.

ranked_results = (
    fold_results_df
    .sort_values(
        by=[
            "fold",
            "f1_macro",
            "config_number"
        ],
        ascending=[
            True,
            False,
            True
        ]
    )
)


selected_hyperparameters = (
    ranked_results
    .groupby(
        "fold",
        as_index=False
    )
    .first()
)


selected_columns = [
    "fold",
    "config_number",
    "n_estimators",
    "max_depth",
    "learning_rate",
    "subsample",
    "colsample_bytree",
    "min_child_weight",
    "f1_macro",
    "accuracy",
    "precision_macro",
    "recall_macro"
]


selected_hyperparameters = (
    selected_hyperparameters[
        selected_columns
    ]
)


selected_output_path = os.path.join(
    OUTPUT_DIR,
    "selected_hyperparameters_by_fold_XGB.tsv"
)

selected_hyperparameters.to_csv(
    selected_output_path,
    sep="\t",
    index=False
)


# ============================================================
# FINAL CHECK
# ============================================================

assert len(
    selected_hyperparameters
) == N_SPLITS, (
    "Expected one selected configuration "
    "for each outer fold."
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "XGBoost grid search completed successfully."
)

print(
    "========================================"
)

print(
    f"\nConfigurations evaluated: "
    f"{N_CONFIGS}"
)

print(
    f"Outer folds: "
    f"{N_SPLITS}"
)

print(
    "\nSelected configuration "
    "for each outer fold:\n"
)


for _, row in (
    selected_hyperparameters.iterrows()
):

    print(
        f"Fold {int(row['fold'])}: "
        f"Config "
        f"{int(row['config_number'])} | "
        f"Validation macro-F1 = "
        f"{row['f1_macro']:.6f}"
    )


print(
    "\nComplete grid-search results:"
)

print(
    grid_output_path
)

print(
    "\nSelected hyperparameters:"
)

print(
    selected_output_path
)