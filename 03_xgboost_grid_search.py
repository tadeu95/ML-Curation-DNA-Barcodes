# XGBoost Grid Search for DNA Barcode Curation (GroupKFold)

import os
import numpy as np
import pandas as pd
import itertools
import random
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import xgboost as xgb
import json
import matplotlib.pyplot as plt

#Set seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Load dataset
file_path = r'G:\O meu disco\ML_training_testing\labelled_for_nn_ready.tsv'
df = pd.read_csv(file_path, sep='\t')


group_column = 'BIN_nn'
id_column = 'processid'
target_column = 'ground_truth_label'

X_full = df.drop(columns=[target_column, id_column, group_column])
y_full = df[target_column].values
groups = df[group_column].values
ids = df[id_column].values

# Feature subsets
percent_col = ['percent_of_bin_records_belonging_to_species_nn', 'genus_prop_in_bin_nn']
count_cols = ['frequency_species_nn', 'total_records_in_bin_nn', 'species_records_in_bin_nn',
              'unique_identifiers_nn', 'unique_institutions_nn',
              'species_per_bin_nn', 'bin_per_species_nn']
entropy_col = ['shannon_entropy_nn']

# Hyperparameter grid for XGBoost
param_grid = { 'n_estimators': [75, 100, 125, 150], 
              'max_depth': [3, 4, 5], 
              'learning_rate': [0.01, 0.05, 0.1], 
              'subsample': [0.8, 1.0], 
              'colsample_bytree': [0.8, 1.0], 
              'min_child_weight': [1, 5]}

all_combos = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]


start = 0
end = 288
search_space = all_combos[start:end]

# Output directory
output_dir = r'G:\O meu disco\ML_training_testing\ANN_grid_search_XGB'
os.makedirs(output_dir, exist_ok=True)

# Save test BINs by fold
test_bins_df = []
folds = GroupKFold(n_splits=10)
for fold, (_, test_idx) in enumerate(folds.split(X_full, y_full, groups)):
    test_groups = groups[test_idx]
    fold_bins = pd.DataFrame({'fold': fold + 1, 'BIN': np.unique(test_groups)})
    test_bins_df.append(fold_bins)

test_bins_df = pd.concat(test_bins_df, ignore_index=True)
test_bins_df.to_csv(os.path.join(output_dir, 'test_bins_by_fold.tsv'), sep='\t', index=False)

# Run grid search
results = []

for config_number, params in enumerate(search_space, start=start + 1):
    print(f"\nRunning config {config_number}/{len(all_combos)}: {params}")
    val_scores = []

    fold_train_losses = []
    fold_val_losses = []

    for fold, (train_val_idx, test_idx) in enumerate(folds.split(X_full, y_full, groups)):
        X_test = X_full.iloc[test_idx].copy()
        y_test = y_full[test_idx]

        X_pool = X_full.iloc[train_val_idx].copy()
        y_pool = y_full[train_val_idx]
        pool_groups = groups[train_val_idx]

        rng = np.random.RandomState(SEED + fold)
        unique_bins = np.unique(pool_groups)
        rng.shuffle(unique_bins)
        n_val = int(0.2 * len(unique_bins))
        val_bins = unique_bins[:n_val]

        val_mask = np.isin(pool_groups, val_bins)
        train_mask = ~val_mask

        X_train = X_pool[train_mask].copy()
        y_train = y_pool[train_mask]
        X_val = X_pool[val_mask].copy()
        y_val = y_pool[val_mask]

        # Preprocessing
        scaler_percent = StandardScaler()
        scaler_entropy = StandardScaler()

        X_train[percent_col] = scaler_percent.fit_transform(X_train[percent_col])
        X_val[percent_col] = scaler_percent.transform(X_val[percent_col])
        X_test[percent_col] = scaler_percent.transform(X_test[percent_col])

        X_train[entropy_col] = scaler_entropy.fit_transform(X_train[entropy_col])
        X_val[entropy_col] = scaler_entropy.transform(X_val[entropy_col])
        X_test[entropy_col] = scaler_entropy.transform(X_test[entropy_col])

        X_train[count_cols] = np.log1p(X_train[count_cols])
        X_val[count_cols] = np.log1p(X_val[count_cols])
        X_test[count_cols] = np.log1p(X_test[count_cols])

        # Train model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            use_label_encoder=False,
            seed=SEED,
            eval_metric='logloss',
            **params
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        evals_result = model.evals_result()
        train_loss = evals_result['validation_0']['logloss']
        val_loss = evals_result['validation_1']['logloss']

        fold_train_losses.append(train_loss)
        fold_val_losses.append(val_loss)

        # Validation scores
        y_val_pred = model.predict(X_val)
        f1_per_class = f1_score(y_val, y_val_pred, average=None, zero_division=0)
        precision_per_class = precision_score(y_val, y_val_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_val, y_val_pred, average=None, zero_division=0)

        val_scores.append({
            'f1_macro': f1_score(y_val, y_val_pred, average='macro'),
            'f1_weighted': f1_score(y_val, y_val_pred, average='weighted'),
            'accuracy': accuracy_score(y_val, y_val_pred),
            'precision_macro': precision_score(y_val, y_val_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_val, y_val_pred, average='macro', zero_division=0),
            'precision_weighted': precision_score(y_val, y_val_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_val, y_val_pred, average='weighted', zero_division=0),
            'f1_class_0': f1_per_class[0],
            'f1_class_1': f1_per_class[1],
            'precision_class_0': precision_per_class[0],
            'precision_class_1': precision_per_class[1],
            'recall_class_0': recall_per_class[0],
            'recall_class_1': recall_per_class[1]
        })

    # Aggregate loss curves across folds
    min_len = min(map(len, fold_val_losses))
    avg_train_loss = np.mean([loss[:min_len] for loss in fold_train_losses], axis=0)
    avg_val_loss = np.mean([loss[:min_len] for loss in fold_val_losses], axis=0)

    # Save learning curves
    curve_data = {
        'train_logloss': avg_train_loss.tolist(),
        'val_logloss': avg_val_loss.tolist()
    }
    curve_path = os.path.join(output_dir, f'config{config_number}_learning_curve.json')
    with open(curve_path, 'w') as f:
        json.dump(curve_data, f)

    plt.figure()
    plt.plot(avg_train_loss, label='Train Logloss')
    plt.plot(avg_val_loss, label='Validation Logloss')
    plt.title(f'Logloss Learning Curve - Config {config_number}')
    plt.xlabel('Boosting Round')
    plt.ylabel('Logloss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'config{config_number}_learning_curve.png'))
    plt.close()

    # Save average metrics
    avg_scores = pd.DataFrame(val_scores).mean().to_dict()
    avg_scores['config_number'] = config_number
    avg_scores['params'] = params
    results.append(avg_scores)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(output_dir, f'xgb_grid_search_summary_{start+1}_to_{end}.tsv'),
    sep='\t',
    index=False
)

# Print best configuration
best_config = results_df.sort_values(by='f1_macro', ascending=False).iloc[0]
print("\nBest configuration based on validation F1 (macro):")
print(best_config)
