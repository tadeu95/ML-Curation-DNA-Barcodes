# -*- coding: utf-8 -*-

"""
Simple rule-based classification across majority thresholds.
"""

import os

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


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
    "simple_rule"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    DATA_PATH,
    sep="\t"
)


# ============================================================
# KEY COLUMNS
# ============================================================

group_column = "BIN_nn"
id_column = "processid"
target_column = "ground_truth_label"

percent_col = (
    "percent_of_bin_records_belonging_to_species_nn"
)

synonym_col = "ingroup_synonym_nn"
ambiguous_col = "ambiguous_name_nn"


# ============================================================
# DATA TYPES
# ============================================================

df[percent_col] = pd.to_numeric(
    df[percent_col]
)

df[synonym_col] = (
    pd.to_numeric(
        df[synonym_col]
    )
    .astype(int)
)

df[ambiguous_col] = (
    pd.to_numeric(
        df[ambiguous_col]
    )
    .astype(int)
)

df[target_column] = (
    pd.to_numeric(
        df[target_column]
    )
    .astype(int)
)


y_true = df[
    target_column
].to_numpy()


# ============================================================
# IDENTIFY SYNONYM-ONLY BINS
# ============================================================

synonym_only_bin = (
    df
    .groupby(
        group_column
    )[
        synonym_col
    ]
    .transform(
        lambda x: (
            x == 1
        ).all()
    )
)


# ============================================================
# THRESHOLDS
# ============================================================

thresholds = list(
    range(
        55,
        100,
        5
    )
)

assert thresholds == [
    55,
    60,
    65,
    70,
    75,
    80,
    85,
    90,
    95
]


# Detect whether percentage values are stored as 0–1 or 0–100
is_proportion = (
    df[
        percent_col
    ].max() <= 1.0
)


# ============================================================
# STORAGE
# ============================================================

results = []

predictions_df = df[
    [
        id_column,
        group_column,
        target_column
    ]
].copy()


# ============================================================
# APPLY RULE ACROSS ALL THRESHOLDS
# ============================================================

for threshold_percent in thresholds:

    threshold = (
        threshold_percent / 100
        if is_proportion
        else threshold_percent
    )


    # --------------------------------------------------------
    # DEFAULT CLASSIFICATION: INCONCLUSIVE
    # --------------------------------------------------------

    predictions = np.zeros(
        len(df),
        dtype=int
    )


    # --------------------------------------------------------
    # RULE 1:
    # Ambiguous species names remain inconclusive.
    #
    # This is implemented by excluding ambiguous records
    # from the supported-class rules below.
    # --------------------------------------------------------


    # --------------------------------------------------------
    # RULE 2:
    # Synonym-only BINs -> supported
    # --------------------------------------------------------

    synonym_mask = (
        (df[ambiguous_col] == 0) &
        synonym_only_bin
    )

    predictions[
        synonym_mask
    ] = 1


    # --------------------------------------------------------
    # RULE 3:
    # Species proportion >= majority threshold -> supported
    # for records not belonging to synonym-only BINs
    # --------------------------------------------------------

    threshold_mask = (
        (df[ambiguous_col] == 0) &
        (~synonym_only_bin) &
        (
            df[percent_col] >=
            threshold
        )
    )

    predictions[
        threshold_mask
    ] = 1


    # ========================================================
    # METRICS
    # ========================================================

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[
            0,
            1
        ]
    ).ravel()


    results.append({

        "threshold_percent":
            threshold_percent,

        "accuracy":
            accuracy_score(
                y_true,
                predictions
            ),

        "f1_macro":
            f1_score(
                y_true,
                predictions,
                average="macro",
                zero_division=0
            ),

        "precision_inconclusive":
            precision_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0
            ),

        "recall_inconclusive":
            recall_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0
            ),

        "f1_inconclusive":
            f1_score(
                y_true,
                predictions,
                pos_label=0,
                zero_division=0
            ),

        "precision_supported":
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0
            ),

        "recall_supported":
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0
            ),

        "f1_supported":
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0
            ),

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,

        "TP":
            tp
    })


    predictions_df[
        f"prediction_threshold_{threshold_percent}"
    ] = predictions


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


results_output_path = os.path.join(
    OUTPUT_DIR,
    "all_thresholds_comparison.tsv"
)

predictions_output_path = os.path.join(
    OUTPUT_DIR,
    "all_threshold_predictions.tsv"
)


results_df.to_csv(
    results_output_path,
    sep="\t",
    index=False
)

predictions_df.to_csv(
    predictions_output_path,
    sep="\t",
    index=False
)


# ============================================================
# INTEGRITY CHECKS
# ============================================================

assert len(
    results_df
) == len(
    thresholds
), (
    "Unexpected number of threshold results."
)


assert len(
    predictions_df
) == len(
    df
), (
    "Prediction file does not contain "
    "the expected number of records."
)


prediction_columns = [
    f"prediction_threshold_{threshold}"
    for threshold in thresholds
]


assert predictions_df[
    prediction_columns
].notna().all().all(), (
    "Missing rule-based predictions detected."
)


# ============================================================
# SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "SIMPLE RULE EVALUATION COMPLETED"
)

print(
    "========================================"
)


print(
    "\nPerformance across majority thresholds:\n"
)

print(
    results_df[
        [
            "threshold_percent",
            "accuracy",
            "f1_macro"
        ]
    ].to_string(
        index=False
    )
)


print(
    "\nResults saved to:"
)

print(
    results_output_path
)

print(
    predictions_output_path
)