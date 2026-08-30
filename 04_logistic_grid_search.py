# -*- coding: utf-8 -*-

"""
Logistic regression grid search with fold-specific hyperparameter selection.
"""

import os
import itertools
import random

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.linear_model import LogisticRegression


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
N_SPLITS = 10
VALIDATION_FRACTION = 0.20


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
    "logistic_grid_search"
)

ANN_TEST_BINS_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "ann_grid_search",
    "test_bins_by_fold.tsv"
)

XGB_TEST_BINS_PATH = os.path.join(
    PROJECT_DIR,
    "results",
    "xgboost_grid_search",
    "test_bins_by_fold.tsv"
)

os.makedirs(
    OUTPUT_DIR,
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

    "C": [
        0.001,
        0.01,
        0.1,
        1,
        10,
        100
    ],

    "penalty": [
        "l2"
    ],

    "solver": [
        "lbfgs",
        "liblinear"
    ],

    "class_weight": [
        None,
        "balanced"
    ],

    "max_iter": [
        1000
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
    f"\nTotal Logistic Regression configurations: "
    f"{N_CONFIGS}"
)

assert N_CONFIGS == 24, (
    f"Expected 24 configurations, "
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

    logistic_test_bins_df = (
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
        logistic_test_bins_df,
        check_dtype=False
    )

    print(
        "\nLogistic Regression outer test BINs "
        "exactly match the ANN outer test BINs."
    )


# ============================================================
# VERIFY SAME OUTER FOLDS AS XGBOOST
# ============================================================

if os.path.exists(
    XGB_TEST_BINS_PATH
):

    xgb_test_bins_df = pd.read_csv(
        XGB_TEST_BINS_PATH,
        sep="\t"
    )

    xgb_test_bins_df = (
        xgb_test_bins_df
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

    logistic_test_bins_df = (
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
        xgb_test_bins_df,
        logistic_test_bins_df,
        check_dtype=False
    )

    print(
        "\nLogistic Regression outer test BINs "
        "exactly match the XGBoost outer test BINs."
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
        scaler_counts = StandardScaler()


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


        # Count-based predictors:
        # log(1 + x) followed by standardization
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

        X_train[count_cols] = (
            scaler_counts.fit_transform(
                X_train[
                    count_cols
                ]
            )
        )

        X_val[count_cols] = (
            scaler_counts.transform(
                X_val[
                    count_cols
                ]
            )
        )


        # ====================================================
        # LOGISTIC REGRESSION MODEL
        # ====================================================

        model = LogisticRegression(
            random_state=SEED,
            **params
        )


        # ====================================================
        # TRAIN MODEL
        # ====================================================

        model.fit(
            X_train,
            y_train
        )


        # ====================================================
        # VALIDATION PREDICTIONS
        # ====================================================

        y_val_pred = model.predict(
            X_val
        )


        # ====================================================
        # VALIDATION METRICS
        # ====================================================

        f1_per_class = f1_score(
            y_val,
            y_val_pred,
            labels=[
                0,
                1
            ],
            average=None,
            zero_division=0
        )

        precision_per_class = precision_score(
            y_val,
            y_val_pred,
            labels=[
                0,
                1
            ],
            average=None,
            zero_division=0
        )

        recall_per_class = recall_score(
            y_val,
            y_val_pred,
            labels=[
                0,
                1
            ],
            average=None,
            zero_division=0
        )


        fold_results.append({

            "fold":
                fold_index,

            "config_number":
                config_number,

            "C":
                params[
                    "C"
                ],

            "penalty":
                params[
                    "penalty"
                ],

            "solver":
                params[
                    "solver"
                ],

            "class_weight":
                params[
                    "class_weight"
                ],

            "max_iter":
                params[
                    "max_iter"
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


folds_per_config = (
    fold_results_df
    .groupby(
        "config_number"
    )[
        "fold"
    ]
    .nunique()
)

assert (
    folds_per_config ==
    N_SPLITS
).all(), (
    "At least one configuration does not "
    "contain all 10 folds."
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
    "Not all 24 configurations were "
    "evaluated in every outer fold."
)


grid_output_path = os.path.join(
    OUTPUT_DIR,
    "logistic_grid_search_by_fold_1_to_24.tsv"
)

fold_results_df.to_csv(
    grid_output_path,
    sep="\t",
    index=False
)


# ============================================================
# SELECT BEST CONFIGURATION WITHIN EACH OUTER FOLD
# ============================================================

selected_rows = []


for fold_index in range(
    1,
    N_SPLITS + 1
):

    fold_subset = (
        fold_results_df[
            fold_results_df[
                "fold"
            ] == fold_index
        ]
        .copy()
    )


    best_f1 = fold_subset[
        "f1_macro"
    ].max()


    # Exact ties are resolved by selecting the lowest
    # configuration number in the predefined grid.
    tied_best = fold_subset[
        np.isclose(
            fold_subset[
                "f1_macro"
            ],
            best_f1,
            rtol=0,
            atol=1e-15
        )
    ].sort_values(
        "config_number"
    )


    best_row = tied_best.iloc[
        0
    ]


    selected_rows.append(
        best_row
    )


selected_hyperparameters = pd.DataFrame(
    selected_rows
).reset_index(
    drop=True
)


selected_columns = [
    "fold",
    "config_number",
    "C",
    "penalty",
    "solver",
    "class_weight",
    "max_iter",
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


# ============================================================
# VERIFY SELECTED HYPERPARAMETERS
# ============================================================

assert len(
    selected_hyperparameters
) == N_SPLITS, (
    "Expected one selected configuration "
    "for each outer fold."
)

assert selected_hyperparameters[
    "fold"
].nunique() == N_SPLITS, (
    "Duplicate or missing outer folds in "
    "selected hyperparameters."
)

assert set(
    selected_hyperparameters[
        "fold"
    ]
) == set(
    range(
        1,
        N_SPLITS + 1
    )
), (
    "Selected hyperparameter file does not "
    "contain folds 1 to 10."
)


# ============================================================
# SAVE SELECTED HYPERPARAMETERS
# ============================================================

selected_output_path = os.path.join(
    OUTPUT_DIR,
    "selected_hyperparameters_by_fold_Logistic.tsv"
)

selected_hyperparameters.to_csv(
    selected_output_path,
    sep="\t",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "Logistic Regression grid search "
    "completed successfully."
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