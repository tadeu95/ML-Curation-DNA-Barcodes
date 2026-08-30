# -*- coding: utf-8 -*-
"""
ANN grid search with fold-specific hyperparameter selection.
"""

import os
import json
import random
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
N_SPLITS = 10
VALIDATION_FRACTION = 0.20
CLASSIFICATION_THRESHOLD = 0.50


# ============================================================
# REPOSITORY PATHS
# ============================================================

# Location of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Assumes that the script is stored inside project/scripts/
PROJECT_DIR = os.path.abspath(
    os.path.join(SCRIPT_DIR, "..")
)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "labelled_for_nn_ready.tsv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "ann_grid_search"
)

HISTORIES_DIR = os.path.join(
    OUTPUT_DIR,
    "histories"
)

CURVES_DIR = os.path.join(
    OUTPUT_DIR,
    "learning_curves"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HISTORIES_DIR, exist_ok=True)
os.makedirs(CURVES_DIR, exist_ok=True)


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


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
# ANN HYPERPARAMETER GRID
# ============================================================

param_grid = {
    "hidden_layers": [
        [16],
        [32],
        [64],
        [32, 16],
        [64, 32]
    ],
    "learning_rate": [
        1e-4,
        1e-3,
        3e-3
    ],
    "batch_size": [
        16,
        32,
        64
    ],
    "activation": [
        "relu",
        "elu"
    ],
    "epochs": [
        10,
        20,
        30
    ]
}

all_combinations = [
    dict(zip(param_grid.keys(), values))
    for values in itertools.product(
        *param_grid.values()
    )
]

N_CONFIGS = len(all_combinations)

print(
    f"\nTotal ANN configurations: {N_CONFIGS}"
)

assert N_CONFIGS == 270, (
    f"Expected 270 configurations, found {N_CONFIGS}."
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
# SAVE TEST BINS FOR EACH OUTER FOLD
# ============================================================

test_bins_all_folds = []

for fold_index, (_, test_idx) in enumerate(
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

test_bins_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_bins_by_fold.tsv"
    ),
    sep="\t",
    index=False
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

    history_acc = []
    history_val_acc = []
    history_loss = []
    history_val_loss = []

    for fold_index, (
        train_val_idx,
        test_idx
    ) in enumerate(
        outer_splits,
        start=1
    ):

        print(
            f"  Fold {fold_index}/{N_SPLITS}"
        )

        # ----------------------------------------------------
        # Outer training pool and held-out outer test set
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
        # Internal grouped validation split
        # ----------------------------------------------------

        # Same grouped validation subset for all configurations
        # within a given outer fold.
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
        # Verify that BINs do not overlap
        # ----------------------------------------------------

        train_bins = set(
            pool_groups[training_mask]
        )

        val_bins = set(
            pool_groups[validation_mask]
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


        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        scaler_percent = StandardScaler()
        scaler_entropy = StandardScaler()
        scaler_counts = StandardScaler()


        # Percentage-based predictors
        X_train[percent_cols] = (
            scaler_percent.fit_transform(
                X_train[percent_cols]
            )
        )

        X_val[percent_cols] = (
            scaler_percent.transform(
                X_val[percent_cols]
            )
        )


        # Shannon entropy
        X_train[entropy_cols] = (
            scaler_entropy.fit_transform(
                X_train[entropy_cols]
            )
        )

        X_val[entropy_cols] = (
            scaler_entropy.transform(
                X_val[entropy_cols]
            )
        )


        # Count-based predictors:
        # log(1 + x) transformation followed by standardization
        X_train[count_cols] = np.log1p(
            X_train[count_cols]
        )

        X_val[count_cols] = np.log1p(
            X_val[count_cols]
        )

        X_train[count_cols] = (
            scaler_counts.fit_transform(
                X_train[count_cols]
            )
        )

        X_val[count_cols] = (
            scaler_counts.transform(
                X_val[count_cols]
            )
        )


        # ----------------------------------------------------
        # BUILD ANN
        # ----------------------------------------------------

        model = Sequential()

        model.add(
            Dense(
                params["hidden_layers"][0],
                input_dim=X_train.shape[1],
                activation=params["activation"]
            )
        )

        for units in params[
            "hidden_layers"
        ][1:]:

            model.add(
                Dense(
                    units,
                    activation=params[
                        "activation"
                    ]
                )
            )

        model.add(
            Dense(
                1,
                activation="sigmoid"
            )
        )

        model.compile(
            optimizer=Adam(
                learning_rate=params[
                    "learning_rate"
                ]
            ),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )


        # ----------------------------------------------------
        # TRAIN ANN
        # ----------------------------------------------------

        history = model.fit(
            X_train,
            y_train,
            validation_data=(
                X_val,
                y_val
            ),
            epochs=params["epochs"],
            batch_size=params[
                "batch_size"
            ],
            verbose=0
        )


        # ----------------------------------------------------
        # SAVE TRAINING HISTORY
        # ----------------------------------------------------

        history_path = os.path.join(
            HISTORIES_DIR,
            (
                f"config{config_number}_"
                f"fold{fold_index}_history.json"
            )
        )

        with open(
            history_path,
            "w"
        ) as file:

            json.dump(
                history.history,
                file
            )


        history_acc.append(
            history.history[
                "accuracy"
            ]
        )

        history_val_acc.append(
            history.history[
                "val_accuracy"
            ]
        )

        history_loss.append(
            history.history[
                "loss"
            ]
        )

        history_val_loss.append(
            history.history[
                "val_loss"
            ]
        )


        # ----------------------------------------------------
        # VALIDATION PREDICTIONS
        # ----------------------------------------------------

        y_val_probability = (
            model.predict(
                X_val,
                verbose=0
            ).flatten()
        )

        y_val_pred = (
            y_val_probability >
            CLASSIFICATION_THRESHOLD
        ).astype(int)


        # ----------------------------------------------------
        # VALIDATION METRICS
        # ----------------------------------------------------

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

            "fold": fold_index,

            "config_number":
                config_number,

            "hidden_layers":
                str(
                    params[
                        "hidden_layers"
                    ]
                ),

            "learning_rate":
                params[
                    "learning_rate"
                ],

            "batch_size":
                params[
                    "batch_size"
                ],

            "activation":
                params[
                    "activation"
                ],

            "epochs":
                params[
                    "epochs"
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
    # AVERAGE LEARNING CURVES ACROSS OUTER FOLDS
    # ========================================================

    avg_acc = np.mean(
        np.array(history_acc),
        axis=0
    )

    avg_val_acc = np.mean(
        np.array(history_val_acc),
        axis=0
    )

    avg_loss = np.mean(
        np.array(history_loss),
        axis=0
    )

    avg_val_loss = np.mean(
        np.array(history_val_loss),
        axis=0
    )


    # Accuracy curve
    plt.figure()

    plt.plot(
        avg_acc,
        label="Training accuracy"
    )

    plt.plot(
        avg_val_acc,
        label="Validation accuracy"
    )

    plt.title(
        f"Average accuracy - "
        f"configuration {config_number}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CURVES_DIR,
            (
                f"config{config_number}_"
                f"avg_accuracy.png"
            )
        ),
        dpi=300
    )

    plt.close()


    # Loss curve
    plt.figure()

    plt.plot(
        avg_loss,
        label="Training loss"
    )

    plt.plot(
        avg_val_loss,
        label="Validation loss"
    )

    plt.title(
        f"Average loss - "
        f"configuration {config_number}"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CURVES_DIR,
            (
                f"config{config_number}_"
                f"avg_loss.png"
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


configs_per_fold = (
    fold_results_df
    .groupby("fold")[
        "config_number"
    ]
    .nunique()
)

assert (
    configs_per_fold ==
    N_CONFIGS
).all(), (
    "Not all 270 configurations were evaluated "
    "in every outer fold."
)


grid_output_path = os.path.join(
    OUTPUT_DIR,
    "grid_search_by_fold_1_to_270.tsv"
)

fold_results_df.to_csv(
    grid_output_path,
    sep="\t",
    index=False
)


# ============================================================
# SELECT BEST CONFIGURATION WITHIN EACH OUTER FOLD
# ============================================================

# Sort first by:
#   1. fold
#   2. validation macro-F1, descending
#   3. configuration number, ascending
#
# Therefore, an exact macro-F1 tie is resolved by retaining
# the configuration appearing first in the predefined grid.

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
    "hidden_layers",
    "learning_rate",
    "batch_size",
    "activation",
    "epochs",
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
    "selected_hyperparameters_by_fold.tsv"
)

selected_hyperparameters.to_csv(
    selected_output_path,
    sep="\t",
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "ANN grid search completed successfully."
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