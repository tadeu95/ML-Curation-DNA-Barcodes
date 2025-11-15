# -*- coding: utf-8 -*-
"""
Created on Wed Aug  6 18:18:49 2025

@author: Utilizador
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  5 18:34:24 2025
@author: Utilizador
"""

import os
import numpy as np
import pandas as pd
import itertools
import random
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import json

#Set seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Load dataset
file_path = r'G:\O meu disco\ML_training_testing\labelled_for_nn_ready.tsv'
df = pd.read_csv(file_path, sep='\t')

# Key columns
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

# Hyperparameters
param_grid = {
    'hidden_layers': [[16], [32], [64], [32, 16], [64, 32]],
    'learning_rate': [1e-4, 1e-3, 3e-3],
    'batch_size': [16, 32, 64],
    'activation': ['relu', 'elu'],
    'epochs': [10, 20, 30]
}

all_combos = [dict(zip(param_grid.keys(), values)) for values in itertools.product(*param_grid.values())]


start = 0   
end = 270   
search_space = all_combos[start:end]
# ------------------------------------------

# Output path
output_dir = r'G:\O meu disco\ML_training_testing\ANN_grid_search'
os.makedirs(output_dir, exist_ok=True)

# Create subfolder for learning curves
curves_dir = os.path.join(output_dir, 'learning_curves')
os.makedirs(curves_dir, exist_ok=True)

# Cross-validation setup
folds = GroupKFold(n_splits=10)

# Save test BINs for all folds
test_bins_df = []
for fold, (_, test_idx) in enumerate(folds.split(X_full, y_full, groups)):
    test_groups = groups[test_idx]
    fold_bins = pd.DataFrame({'fold': fold + 1, 'BIN': np.unique(test_groups)})
    test_bins_df.append(fold_bins)

test_bins_df = pd.concat(test_bins_df, ignore_index=True)
test_bins_df.to_csv(os.path.join(output_dir, 'test_bins_by_fold.tsv'), sep='\t', index=False)

# Evaluate all hyperparameter combinations
results = []

for config_number, params in enumerate(search_space, start=start + 1):
    print(f"\nRunning config {config_number}/{len(all_combos)}: {params}")
    val_scores = []

    history_acc = []
    history_val_acc = []
    history_loss = []
    history_val_loss = []

    for fold, (train_val_idx, test_idx) in enumerate(folds.split(X_full, y_full, groups)):
        X_test = X_full.iloc[test_idx].copy()
        y_test = y_full[test_idx]
        test_ids = ids[test_idx]
        test_groups = groups[test_idx]

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

        # Model
        model = Sequential()
        model.add(Dense(params['hidden_layers'][0], input_dim=X_train.shape[1], activation=params['activation']))
        for units in params['hidden_layers'][1:]:
            model.add(Dense(units, activation=params['activation']))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=params['learning_rate']),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=params['epochs'],
            batch_size=params['batch_size'],
            verbose=0
        )

        history_path = os.path.join(output_dir, f'config{config_number}_fold{fold+1}_history.json')
        with open(history_path, 'w') as f:
            json.dump(history.history, f)

        history_acc.append(history.history['accuracy'])
        history_val_acc.append(history.history['val_accuracy'])
        history_loss.append(history.history['loss'])
        history_val_loss.append(history.history['val_loss'])

        y_val_pred = (model.predict(X_val).flatten() > 0.5).astype(int)
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

    avg_scores = pd.DataFrame(val_scores).mean().to_dict()
    avg_scores['config_number'] = config_number
    avg_scores['params'] = params
    results.append(avg_scores)

    # Plot learning curves
    avg_acc = np.mean(np.array(history_acc), axis=0)
    avg_val_acc = np.mean(np.array(history_val_acc), axis=0)
    avg_loss = np.mean(np.array(history_loss), axis=0)
    avg_val_loss = np.mean(np.array(history_val_loss), axis=0)

    plt.figure()
    plt.plot(avg_acc, label='Train Accuracy')
    plt.plot(avg_val_acc, label='Validation Accuracy')
    plt.title(f'Average Accuracy - Config {config_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, f'config{config_number}_avg_accuracy.png'))
    plt.close()

    plt.figure()
    plt.plot(avg_loss, label='Train Loss')
    plt.plot(avg_val_loss, label='Validation Loss')
    plt.title(f'Average Loss - Config {config_number}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(curves_dir, f'config{config_number}_avg_loss.png'))
    plt.close()

# Save results 
results_df = pd.DataFrame(results)
results_df.to_csv(
    os.path.join(output_dir, f'grid_search_summary_{start+1}_to_{end}.tsv'),
    sep='\t',
    index=False
)

# Print best config
best_config = results_df.sort_values(by='f1_macro', ascending=False).iloc[0]
print("\nBest configuration based on validation F1:")
print(best_config)
