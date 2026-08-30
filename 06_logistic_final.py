# -*- coding: utf-8 -*-

"""
Final Logistic Regression training and evaluation using fold-specific hyperparameters.
"""

import os
import json
import time
import random
import platform

import numpy as np
import pandas as pd
import sklearn
import joblib

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
from sklearn.linear_model import LogisticRegression


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
    "logistic_grid_search"
)

GRID_RESULTS_PATH = os.path.join(
    GRID_SEARCH_DIR,
    "logistic_grid_search_by_fold_1_to_24.tsv"
)

SELECTED_PARAMS_PATH = os.path.join(
    GRID_SEARCH_DIR,
    "selected_hyperparameters_by_fold_Logistic.tsv"
)

GRID_TEST_BINS_PATH = os.path.join(
    GRID_SEARCH_DIR,
    "test_bins_by_fold.tsv"
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

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "results",
    "logistic_final"
)

MODELS_DIR = os.path.join(
    OUTPUT_DIR,
    "models"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    MODELS_DIR,
    exist_ok=True
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# ENVIRONMENT INFORMATION
# ============================================================

env_info = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "processor": platform.processor(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "scikit_learn": sklearn.__version__
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
# LOAD GRID SEARCH RESULTS
# ============================================================

grid_results_df = pd.read_csv(
    GRID_RESULTS_PATH,
    sep="\t"
)

selected_df = pd.read_csv(
    SELECTED_PARAMS_PATH,
    sep="\t"
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
# CHECK COMPLETE GRID SEARCH
# ============================================================

expected_grid_rows = (
    24 *
    N_SPLITS
)

assert len(
    grid_results_df
) == expected_grid_rows, (
    f"Expected {expected_grid_rows} Logistic Regression "
    f"grid-search rows, found {len(grid_results_df)}."
)


duplicate_count = (
    grid_results_df
    .duplicated(
        subset=[
            "config_number",
            "fold"
        ]
    )
    .sum()
)

assert duplicate_count == 0, (
    "Duplicate configuration/fold combinations detected."
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
    configs_per_fold == 24
).all(), (
    "Not all 24 Logistic Regression configurations "
    "were evaluated in every outer fold."
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
    "Expected one selected Logistic Regression "
    "configuration for each outer fold."
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
    "Selected hyperparameter file does not "
    "contain folds 1 to 10 exactly once."
)

assert selected_df[
    "config_number"
].notna().all(), (
    "Missing configuration number detected."
)


# ============================================================
# VERIFY HYPERPARAMETER SELECTION AGAINST COMPLETE GRID
# ============================================================

for fold_index in range(
    1,
    N_SPLITS + 1
):

    fold_grid = (
        grid_results_df[
            grid_results_df[
                "fold"
            ] == fold_index
        ]
        .copy()
    )

    best_f1 = fold_grid[
        "f1_macro"
    ].max()

    tied_best = fold_grid[
        np.isclose(
            fold_grid[
                "f1_macro"
            ],
            best_f1,
            rtol=0,
            atol=1e-15
        )
    ].sort_values(
        by="config_number"
    )

    expected_config = int(
        tied_best.iloc[0][
            "config_number"
        ]
    )

    selected_config = int(
        selected_df.loc[
            selected_df[
                "fold"
            ] == fold_index,
            "config_number"
        ].iloc[0]
    )

    assert selected_config == expected_config, (
        f"Fold {fold_index}: selected configuration "
        f"{selected_config}, but expected "
        f"{expected_config}."
    )


print(
    "\nSelected Logistic Regression hyperparameters "
    "were successfully verified against the complete grid search."
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


final_test_bins_df = pd.concat(
    test_bins_all_folds,
    ignore_index=True
)

final_test_bins_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "test_bins_by_fold.tsv"
    ),
    sep="\t",
    index=False
)


# ============================================================
# VERIFY FINAL TEST BINS AGAINST LOGISTIC GRID SEARCH
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

final_test_bins_sorted = (
    final_test_bins_df
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
    final_test_bins_sorted,
    check_dtype=False
)

print(
    "Final Logistic Regression outer test BINs "
    "exactly match the grid search."
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

    pd.testing.assert_frame_equal(
        ann_test_bins_df,
        final_test_bins_sorted,
        check_dtype=False
    )

    print(
        "Final Logistic Regression outer test BINs "
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

    pd.testing.assert_frame_equal(
        xgb_test_bins_df,
        final_test_bins_sorted,
        check_dtype=False
    )

    print(
        "Final Logistic Regression outer test BINs "
        "exactly match the XGBoost outer test BINs."
    )


# ============================================================
# STORAGE
# ============================================================

all_predictions = []
all_metrics = []


# ============================================================
# FINAL TRAINING AND TESTING
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
        f"Running Logistic Regression fold {fold_index}"
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


    # Empty cells representing None are read by pandas as NaN.
    class_weight = (
        None
        if pd.isna(
            fold_selection[
                "class_weight"
            ]
        )
        else str(
            fold_selection[
                "class_weight"
            ]
        )
    )


    best_params = {

        "C":
            float(
                fold_selection[
                    "C"
                ]
            ),

        "penalty":
            str(
                fold_selection[
                    "penalty"
                ]
            ),

        "solver":
            str(
                fold_selection[
                    "solver"
                ]
            ),

        "class_weight":
            class_weight,

        "max_iter":
            int(
                fold_selection[
                    "max_iter"
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
    # OUTER TRAIN AND TEST DATA
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
    # LOGISTIC REGRESSION MODEL
    # ========================================================

    model = LogisticRegression(
        random_state=SEED,
        **best_params
    )


    # ========================================================
    # TRAIN ON COMPLETE OUTER TRAINING SET
    # ========================================================

    model.fit(
        X_train,
        y_train
    )


    # ========================================================
    # SAVE MODEL AND PREPROCESSING
    # ========================================================

    model_bundle = {

        "model":
            model,

        "config_number":
            config_number,

        "params":
            best_params,

        "feature_names":
            X_train.columns.tolist(),

        "percent_cols":
            percent_cols,

        "entropy_cols":
            entropy_cols,

        "count_cols":
            count_cols,

        "scaler_percent":
            scaler_percent,

        "scaler_entropy":
            scaler_entropy,

        "scaler_counts":
            scaler_counts,

        "count_transformation":
            "np.log1p"
    }


    joblib.dump(
        model_bundle,
        os.path.join(
            MODELS_DIR,
            f"fold{fold_index}_model.joblib"
        )
    )


    # ========================================================
    # OUTER TEST PREDICTIONS
    # ========================================================

    y_probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
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
].duplicated().sum() == 0, (
    "Duplicate processid predictions detected."
)


assert all_predictions_df[
    "processid"
].nunique() == df[
    "processid"
].nunique(), (
    "Mismatch in number of unique processids."
)


assert set(
    all_predictions_df[
        "processid"
    ]
) == set(
    df[
        "processid"
    ]
), (
    "Final prediction file does not contain "
    "exactly the original processids."
)


assert all_predictions_df[
    "fold"
].nunique() == N_SPLITS, (
    "Expected predictions from exactly 10 folds."
)


assert all_predictions_df[
    "predicted"
].notna().all(), (
    "Missing predictions detected."
)


assert all_predictions_df[
    "probability"
].notna().all(), (
    "Missing prediction probabilities detected."
)


print(
    "\nFinal Logistic Regression prediction checks passed."
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


overall_metrics_df = pd.DataFrame([{

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


overall_metrics_df.to_csv(
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
    "FINAL LOGISTIC REGRESSION EVALUATION COMPLETED"
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