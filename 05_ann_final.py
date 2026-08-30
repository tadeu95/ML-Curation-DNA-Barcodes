# -*- coding: utf-8 -*-

"""
Final ANN training and evaluation using fold-specific hyperparameters.
"""

import os
import ast
import json
import time
import random
import platform

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report
)

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
N_SPLITS = 10
CLASSIFICATION_THRESHOLD = 0.50

start_time = time.time()


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

GRID_SEARCH_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "ann_grid_search"
)

GRID_RESULTS_PATH = os.path.join(
    GRID_SEARCH_DIR,
    "grid_search_by_fold_1_to_270.tsv"
)

SELECTED_PARAMS_PATH = os.path.join(
    GRID_SEARCH_DIR,
    "selected_hyperparameters_by_fold.tsv"
)

GRID_TEST_BINS_PATH = os.path.join(
    GRID_SEARCH_DIR,
    "test_bins_by_fold.tsv"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "ann_final"
)

MODELS_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

HISTORIES_DIR = os.path.join(
    OUTPUT_DIR,
    "histories"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)

os.makedirs(
    HISTORIES_DIR,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# ENVIRONMENT INFORMATION
# ============================================================

env_info = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "processor": platform.processor(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "tensorflow": tf.__version__
}

with open(
    os.path.join(
        OUTPUT_DIR,
        "environment_info.json"
    ),
    "w"
) as file:

    json.dump(
        env_info,
        file,
        indent=4
    )


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
ids = df[id_column].values


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
# LOAD GRID-SEARCH RESULTS
# ============================================================

grid_results_df = pd.read_csv(
    GRID_RESULTS_PATH,
    sep="\t"
)

selected_df = pd.read_csv(
    SELECTED_PARAMS_PATH,
    sep="\t"
)


# ============================================================
# CHECK COMPLETE GRID SEARCH
# ============================================================

expected_grid_rows = (
    270 *
    N_SPLITS
)

assert len(
    grid_results_df
) == expected_grid_rows, (
    f"Expected {expected_grid_rows} ANN grid-search rows, "
    f"found {len(grid_results_df)}."
)


configs_per_fold = (
    grid_results_df
    .groupby(
        "fold"
    )[
        "config_number"
    ]
    .nunique()
)

assert (
    configs_per_fold == 270
).all(), (
    "Not all 270 ANN configurations were "
    "evaluated in every outer fold."
)


# ============================================================
# CHECK SELECTED HYPERPARAMETERS
# ============================================================

assert len(
    selected_df
) == N_SPLITS, (
    f"Expected {N_SPLITS} selected configurations, "
    f"found {len(selected_df)}."
)

assert selected_df[
    "fold"
].nunique() == N_SPLITS, (
    "Expected one selected ANN configuration "
    "for each outer fold."
)

assert set(
    selected_df[
        "fold"
    ]
) == set(
    range(
        1,
        N_SPLITS + 1
    )
), (
    "Selected ANN hyperparameter file does not "
    "contain folds 1 to 10 exactly once."
)

assert selected_df[
    "config_number"
].notna().all(), (
    "Missing ANN configuration number detected."
)

selected_df = (
    selected_df
    .sort_values(
        by="fold"
    )
    .reset_index(
        drop=True
    )
)


# ============================================================
# VERIFY HYPERPARAMETER SELECTION AGAINST COMPLETE GRID
# ============================================================

ranked_grid = (
    grid_results_df
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

expected_selected = (
    ranked_grid
    .groupby(
        "fold",
        as_index=False
    )
    .first()
    .sort_values(
        by="fold"
    )
    .reset_index(
        drop=True
    )
)


pd.testing.assert_series_equal(
    selected_df[
        "config_number"
    ].astype(int),
    expected_selected[
        "config_number"
    ].astype(int),
    check_names=False
)


parameter_columns = [
    "hidden_layers",
    "learning_rate",
    "batch_size",
    "activation",
    "epochs"
]


for column in parameter_columns:

    pd.testing.assert_series_equal(
        selected_df[
            column
        ].astype(str),
        expected_selected[
            column
        ].astype(str),
        check_names=False
    )


print(
    "\nSelected ANN hyperparameters were "
    "successfully verified against the complete grid search."
)


# ============================================================
# SAVE HYPERPARAMETERS USED IN FINAL TRAINING
# ============================================================

selected_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "selected_hyperparameters_by_fold.tsv"
    ),
    sep="\t",
    index=False
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
# SAVE FINAL TEST BINS
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

test_bins_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_bins_by_fold.tsv"
    ),
    sep="\t",
    index=False
)


# ============================================================
# VERIFY FINAL TEST BINS AGAINST GRID SEARCH
# ============================================================

grid_test_bins_df = pd.read_csv(
    GRID_TEST_BINS_PATH,
    sep="\t"
)

grid_test_bins_df = (
    grid_test_bins_df
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

final_test_bins_df = (
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
    grid_test_bins_df,
    final_test_bins_df,
    check_dtype=False
)

print(
    "Final ANN test BIN assignments exactly "
    "match the grid search."
)


# ============================================================
# STORE PREDICTIONS AND METRICS
# ============================================================

all_predictions = []
all_metrics = []


# ============================================================
# FINAL MODEL TRAINING AND TESTING
# ============================================================

for fold_index, (
    train_idx,
    test_idx
) in enumerate(
    outer_splits,
    start=1
):

    print(
        "\n" + "=" * 60
    )

    print(
        f"Running ANN fold {fold_index}"
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # SELECT HYPERPARAMETERS FOR THIS FOLD
    # --------------------------------------------------------

    fold_selection = selected_df[
        selected_df[
            "fold"
        ] == fold_index
    ].iloc[0]


    config_number = int(
        fold_selection[
            "config_number"
        ]
    )


    hidden_layers = ast.literal_eval(
        fold_selection[
            "hidden_layers"
        ]
    )


    best_params = {

        "hidden_layers":
            hidden_layers,

        "learning_rate":
            float(
                fold_selection[
                    "learning_rate"
                ]
            ),

        "batch_size":
            int(
                fold_selection[
                    "batch_size"
                ]
            ),

        "activation":
            str(
                fold_selection[
                    "activation"
                ]
            ),

        "epochs":
            int(
                fold_selection[
                    "epochs"
                ]
            )
    }


    print(
        f"Selected configuration: "
        f"{config_number}"
    )

    print(
        f"Parameters: "
        f"{best_params}"
    )


    # --------------------------------------------------------
    # REPRODUCIBILITY FOR THIS FOLD
    # --------------------------------------------------------

    fold_seed = (
        SEED +
        fold_index
    )

    random.seed(
        fold_seed
    )

    np.random.seed(
        fold_seed
    )

    tf.random.set_seed(
        fold_seed
    )

    tf.keras.backend.clear_session()


    # --------------------------------------------------------
    # VERIFY NO BIN OVERLAP
    # --------------------------------------------------------

    train_bins = set(
        groups[
            train_idx
        ]
    )

    test_bins = set(
        groups[
            test_idx
        ]
    )

    assert not (
        train_bins &
        test_bins
    ), (
        f"Train/test BIN overlap "
        f"in fold {fold_index}."
    )


    # --------------------------------------------------------
    # OUTER TRAIN AND TEST SETS
    # --------------------------------------------------------

    X_train = X_full.iloc[
        train_idx
    ].copy()

    y_train = y_full[
        train_idx
    ]

    X_test = X_full.iloc[
        test_idx
    ].copy()

    y_test = y_full[
        test_idx
    ]

    test_ids = ids[
        test_idx
    ]

    test_groups = groups[
        test_idx
    ]


    # ========================================================
    # PREPROCESSING
    # ========================================================

    scaler_percent = StandardScaler()
    scaler_entropy = StandardScaler()
    scaler_counts = StandardScaler()


    # Percentage-based predictors
    X_train[
        percent_cols
    ] = scaler_percent.fit_transform(
        X_train[
            percent_cols
        ]
    )

    X_test[
        percent_cols
    ] = scaler_percent.transform(
        X_test[
            percent_cols
        ]
    )


    # Shannon entropy
    X_train[
        entropy_cols
    ] = scaler_entropy.fit_transform(
        X_train[
            entropy_cols
        ]
    )

    X_test[
        entropy_cols
    ] = scaler_entropy.transform(
        X_test[
            entropy_cols
        ]
    )


    # Count-based predictors:
    # log(1 + x) followed by standardization
    X_train[
        count_cols
    ] = np.log1p(
        X_train[
            count_cols
        ]
    )

    X_test[
        count_cols
    ] = np.log1p(
        X_test[
            count_cols
        ]
    )

    X_train[
        count_cols
    ] = scaler_counts.fit_transform(
        X_train[
            count_cols
        ]
    )

    X_test[
        count_cols
    ] = scaler_counts.transform(
        X_test[
            count_cols
        ]
    )


    # ========================================================
    # BUILD ANN
    # ========================================================

    model = Sequential()

    model.add(
        Dense(
            best_params[
                "hidden_layers"
            ][0],
            input_dim=X_train.shape[1],
            activation=best_params[
                "activation"
            ]
        )
    )


    for units in best_params[
        "hidden_layers"
    ][1:]:

        model.add(
            Dense(
                units,
                activation=best_params[
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


    # ========================================================
    # COMPILE ANN
    # ========================================================

    model.compile(

        optimizer=Adam(
            learning_rate=best_params[
                "learning_rate"
            ]
        ),

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )


    # ========================================================
    # TRAIN ON COMPLETE OUTER TRAINING SET
    # ========================================================

    history = model.fit(

        X_train,
        y_train,

        epochs=best_params[
            "epochs"
        ],

        batch_size=best_params[
            "batch_size"
        ],

        verbose=0
    )


    # ========================================================
    # SAVE MODEL
    # ========================================================

    model.save(
        os.path.join(
            MODELS_DIR,
            f"fold{fold_index}_model.h5"
        )
    )


    # ========================================================
    # SAVE FINAL TRAINING HISTORY
    # ========================================================

    history_path = os.path.join(
        HISTORIES_DIR,
        f"fold{fold_index}_history.json"
    )


    with open(
        history_path,
        "w"
    ) as file:

        json.dump(
            history.history,
            file
        )


    # ========================================================
    # PREDICT ON OUTER TEST SET
    # ========================================================

    y_probabilities = (
        model.predict(
            X_test,
            verbose=0
        )
        .flatten()
    )


    y_predictions = (
        y_probabilities >
        CLASSIFICATION_THRESHOLD
    ).astype(int)


    # ========================================================
    # SAVE FOLD PREDICTIONS
    # ========================================================

    fold_predictions = pd.DataFrame({

        "processid":
            test_ids,

        "BIN":
            test_groups,

        "ground_truth":
            y_test,

        "predicted":
            y_predictions,

        "probability":
            y_probabilities,

        "fold":
            fold_index,

        "config_number":
            config_number
    })


    all_predictions.append(
        fold_predictions
    )


    # ========================================================
    # FOLD METRICS
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_predictions
    )


    f1_macro = f1_score(
        y_test,
        y_predictions,
        average="macro",
        zero_division=0
    )


    f1_weighted = f1_score(
        y_test,
        y_predictions,
        average="weighted",
        zero_division=0
    )


    precision_macro = precision_score(
        y_test,
        y_predictions,
        average="macro",
        zero_division=0
    )


    recall_macro = recall_score(
        y_test,
        y_predictions,
        average="macro",
        zero_division=0
    )


    confusion = confusion_matrix(
        y_test,
        y_predictions,
        labels=[
            0,
            1
        ]
    )


    tn, fp, fn, tp = (
        confusion.ravel()
    )


    report = classification_report(
        y_test,
        y_predictions,
        labels=[
            0,
            1
        ],
        output_dict=True,
        zero_division=0
    )


    all_metrics.append({

        "fold":
            fold_index,

        "config_number":
            config_number,

        "accuracy":
            accuracy,

        "f1_macro":
            f1_macro,

        "f1_weighted":
            f1_weighted,

        "precision_macro":
            precision_macro,

        "recall_macro":
            recall_macro,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp,

        "precision_class_0":
            report[
                "0"
            ][
                "precision"
            ],

        "recall_class_0":
            report[
                "0"
            ][
                "recall"
            ],

        "f1_class_0":
            report[
                "0"
            ][
                "f1-score"
            ],

        "support_class_0":
            report[
                "0"
            ][
                "support"
            ],

        "precision_class_1":
            report[
                "1"
            ][
                "precision"
            ],

        "recall_class_1":
            report[
                "1"
            ][
                "recall"
            ],

        "f1_class_1":
            report[
                "1"
            ][
                "f1-score"
            ],

        "support_class_1":
            report[
                "1"
            ][
                "support"
            ]
    })


    print(
        f"Fold {fold_index} completed | "
        f"Accuracy = {accuracy:.6f} | "
        f"Macro-F1 = {f1_macro:.6f}"
    )


# ============================================================
# COMBINE ALL OUT-OF-FOLD PREDICTIONS
# ============================================================

all_predictions_df = pd.concat(
    all_predictions,
    ignore_index=True
)


# ============================================================
# FINAL PREDICTION CHECKS
# ============================================================

assert len(
    all_predictions_df
) == len(
    df
), (
    f"Expected {len(df)} predictions, "
    f"found {len(all_predictions_df)}."
)


assert all_predictions_df[
    "processid"
].nunique() == len(
    df
), (
    "At least one processid is duplicated or "
    "missing from the final predictions."
)


assert all_predictions_df[
    "fold"
].nunique() == N_SPLITS, (
    "Final predictions do not contain "
    "exactly 10 folds."
)


assert all_predictions_df[
    "predicted"
].notna().all(), (
    "Missing ANN predictions detected."
)


assert all_predictions_df[
    "probability"
].notna().all(), (
    "Missing ANN prediction probabilities detected."
)


print(
    "\nFinal ANN prediction checks passed."
)


print(
    f"Total predictions: "
    f"{len(all_predictions_df)}"
)


print(
    f"Unique processids: "
    f"{all_predictions_df['processid'].nunique()}"
)


# ============================================================
# SAVE ALL OUT-OF-FOLD PREDICTIONS
# ============================================================

result_file_path = os.path.join(
    OUTPUT_DIR,
    "result_file.tsv"
)


all_predictions_df.to_csv(
    result_file_path,
    sep="\t",
    index=False
)


# ============================================================
# SAVE FOLD METRICS + MEAN / SD / SE
# ============================================================

metrics_df = pd.DataFrame(
    all_metrics
)


mean_metrics = metrics_df.mean(
    numeric_only=True
)

std_metrics = metrics_df.std(
    numeric_only=True
)

se_metrics = metrics_df.sem(
    numeric_only=True
)


summary_df = pd.concat([

    metrics_df,

    pd.DataFrame(
        [
            mean_metrics
        ],
        index=[
            "mean"
        ]
    ),

    pd.DataFrame(
        [
            std_metrics
        ],
        index=[
            "std"
        ]
    ),

    pd.DataFrame(
        [
            se_metrics
        ],
        index=[
            "se"
        ]
    )
])


summary_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "metrics_summary.tsv"
    ),
    sep="\t"
)


# ============================================================
# OVERALL METRICS FROM ALL OUT-OF-FOLD PREDICTIONS
# ============================================================

y_true_all = all_predictions_df[
    "ground_truth"
].values


y_pred_all = all_predictions_df[
    "predicted"
].values


overall_accuracy = accuracy_score(
    y_true_all,
    y_pred_all
)


overall_f1_macro = f1_score(
    y_true_all,
    y_pred_all,
    average="macro",
    zero_division=0
)


overall_f1_weighted = f1_score(
    y_true_all,
    y_pred_all,
    average="weighted",
    zero_division=0
)


overall_precision_macro = precision_score(
    y_true_all,
    y_pred_all,
    average="macro",
    zero_division=0
)


overall_recall_macro = recall_score(
    y_true_all,
    y_pred_all,
    average="macro",
    zero_division=0
)


overall_confusion = confusion_matrix(
    y_true_all,
    y_pred_all,
    labels=[
        0,
        1
    ]
)


overall_tn, overall_fp, overall_fn, overall_tp = (
    overall_confusion.ravel()
)


overall_report = classification_report(
    y_true_all,
    y_pred_all,
    labels=[
        0,
        1
    ],
    output_dict=True,
    zero_division=0
)


overall_metrics = pd.DataFrame([{

    "accuracy":
        overall_accuracy,

    "f1_macro":
        overall_f1_macro,

    "f1_weighted":
        overall_f1_weighted,

    "precision_macro":
        overall_precision_macro,

    "recall_macro":
        overall_recall_macro,

    "TN":
        overall_tn,

    "FP":
        overall_fp,

    "FN":
        overall_fn,

    "TP":
        overall_tp,

    "precision_class_0":
        overall_report[
            "0"
        ][
            "precision"
        ],

    "recall_class_0":
        overall_report[
            "0"
        ][
            "recall"
        ],

    "f1_class_0":
        overall_report[
            "0"
        ][
            "f1-score"
        ],

    "support_class_0":
        overall_report[
            "0"
        ][
            "support"
        ],

    "precision_class_1":
        overall_report[
            "1"
        ][
            "precision"
        ],

    "recall_class_1":
        overall_report[
            "1"
        ][
            "recall"
        ],

    "f1_class_1":
        overall_report[
            "1"
        ][
            "f1-score"
        ],

    "support_class_1":
        overall_report[
            "1"
        ][
            "support"
        ]
}])


overall_metrics.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "overall_metrics.tsv"
    ),
    sep="\t",
    index=False
)


# ============================================================
# RUNTIME
# ============================================================

total_seconds = (
    time.time() -
    start_time
)


minutes, seconds = divmod(
    int(
        total_seconds
    ),
    60
)


print(
    "\n" + "=" * 60
)

print(
    "FINAL ANN EVALUATION COMPLETED"
)

print(
    "=" * 60
)


print(
    f"Overall Accuracy: "
    f"{overall_accuracy:.6f}"
)

print(
    f"Overall Macro-F1: "
    f"{overall_f1_macro:.6f}"
)

print(
    f"Overall Precision (macro): "
    f"{overall_precision_macro:.6f}"
)

print(
    f"Overall Recall (macro): "
    f"{overall_recall_macro:.6f}"
)

print(
    f"Total runtime: "
    f"{minutes} min "
    f"{seconds} s "
    f"({total_seconds:.2f} s)"
)


with open(
    os.path.join(
        OUTPUT_DIR,
        "runtime.txt"
    ),
    "w"
) as file:

    file.write(
        f"Total runtime: "
        f"{minutes} min "
        f"{seconds} s "
        f"({total_seconds:.2f} s)\n"
    )