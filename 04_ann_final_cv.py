#Final training and testing using selected ANN configuration

import os
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import classification_report

import time
start_time = time.time()

import platform

env_info = {
    "python": platform.python_version(),
    "platform": platform.platform(),
    "processor": platform.processor(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "tensorflow": tf.__version__,
}

#Set seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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

# Chosen configuration
best_params = {
    'hidden_layers': [16],
    'learning_rate': 0.0001,
    'batch_size': 64,
    'activation': 'elu',
    'epochs': 20
}

# Output directory
output_dir = r'G:\O meu disco\ML_training_testing\ANN_final_training'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'models'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'histories'), exist_ok=True)


# Cross-validation setup
folds = GroupKFold(n_splits=10)
#folds = GroupKFold(n_splits=5)


# Save test BINs per fold
test_bins_df = []
for fold, (_, test_idx) in enumerate(folds.split(X_full, y_full, groups)):
    test_groups = groups[test_idx]
    fold_bins = pd.DataFrame({
        'fold': fold + 1,
        'BIN': np.unique(test_groups)
    })
    test_bins_df.append(fold_bins)
pd.concat(test_bins_df, ignore_index=True).to_csv(os.path.join(output_dir, 'test_bins_by_fold.tsv'), sep='\t', index=False)

# Store predictions and metrics
all_predictions = []
all_metrics = []

# Final model training/testing
for fold, (train_idx, test_idx) in enumerate(folds.split(X_full, y_full, groups), start=1):
    print(f"Running fold {fold}...")
    train_bins = set(groups[train_idx])
    test_bins = set(groups[test_idx])
    assert len(train_bins & test_bins) == 0, f"Leak in fold {fold}: {train_bins & test_bins}"
    print("All folds: No BINs overlap between train/test sets.")
    X_train = X_full.iloc[train_idx].copy()
    y_train = y_full[train_idx]
    X_test = X_full.iloc[test_idx].copy()
    y_test = y_full[test_idx]
    test_ids = ids[test_idx]
    test_groups = groups[test_idx]

    # Scaling
    scaler_percent = StandardScaler()
    scaler_entropy = StandardScaler()

    X_train[percent_col] = scaler_percent.fit_transform(X_train[percent_col])
    X_test[percent_col] = scaler_percent.transform(X_test[percent_col])

    X_train[entropy_col] = scaler_entropy.fit_transform(X_train[entropy_col])
    X_test[entropy_col] = scaler_entropy.transform(X_test[entropy_col])

    X_train[count_cols] = np.log1p(X_train[count_cols])
    X_test[count_cols] = np.log1p(X_test[count_cols])

    # Model
    model = Sequential()
    model.add(Dense(best_params['hidden_layers'][0], input_dim=X_train.shape[1], activation=best_params['activation']))
    for units in best_params['hidden_layers'][1:]:
        model.add(Dense(units, activation=best_params['activation']))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=Adam(learning_rate=best_params['learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
        X_train, y_train,
        epochs=best_params['epochs'],
        batch_size=best_params['batch_size'],
        verbose=0
    )

    # Save model + history
    model.save(os.path.join(output_dir, 'models', f'fold{fold}_model.h5'))

    # Predict on test set
    y_probs = model.predict(X_test).flatten()
    y_preds = (y_probs > 0.5).astype(int)

    # Save predictions
    fold_preds = pd.DataFrame({
        'processid': test_ids,
        'ground_truth': y_test,
        'predicted': y_preds,
        'probability': y_probs,
        'fold': fold
    })
    all_predictions.append(fold_preds)

    # Metrics
    acc = accuracy_score(y_test, y_preds)
    f1_macro = f1_score(y_test, y_preds, average='macro')
    f1_weighted = f1_score(y_test, y_preds, average='weighted')
    precision_macro = precision_score(y_test, y_preds, average='macro', zero_division=0)
    recall_macro = recall_score(y_test, y_preds, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_preds)
    tn, fp, fn, tp = cm.ravel()
    report = classification_report(y_test, y_preds, output_dict=True, zero_division=0)


    all_metrics.append({
            'fold': fold,
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'TP': tp,
            # Per-class metrics
            'precision_class_0': report['0']['precision'],
            'recall_class_0': report['0']['recall'],
            'f1_class_0': report['0']['f1-score'],
            'support_class_0': report['0']['support'],
            'precision_class_1': report['1']['precision'],
            'recall_class_1': report['1']['recall'],
            'f1_class_1': report['1']['f1-score'],
            'support_class_1': report['1']['support']
        })


# Save all predictions
all_predictions_df = pd.concat(all_predictions, ignore_index=True)
all_predictions_df.to_csv(os.path.join(output_dir, 'result_file.tsv'), sep='\t', index=False)

# Save all metrics with averages
metrics_df = pd.DataFrame(all_metrics)
mean_metrics = metrics_df.mean(numeric_only=True)
std_metrics = metrics_df.std(numeric_only=True)
se_metrics = metrics_df.sem(numeric_only=True)

summary_df = pd.concat([
    metrics_df,
    pd.DataFrame([mean_metrics], index=['mean']),
    pd.DataFrame([std_metrics], index=['std']),
    pd.DataFrame([se_metrics], index=['se'])
])

summary_df.to_csv(os.path.join(output_dir, 'metrics_summary.tsv'), sep='\t')

total_seconds = time.time() - start_time
mins, secs = divmod(int(total_seconds), 60)
print(f"Total runtime: {mins} min {secs} s ({total_seconds:.2f} s)")

# Save to a small log file too
with open(os.path.join(output_dir, "runtime.txt"), "w") as f:
    f.write(f"Total runtime: {mins} min {secs} s ({total_seconds:.2f} s)\n")
