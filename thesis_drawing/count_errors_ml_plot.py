import numpy as np
import os
import matplotlib.pyplot as plt
from joblib import load
from machine_learning.ml_method.calibrated_label_ranking import (
    calibrated_label_ranking_classifier,
)
from machine_learning.ml_method.ml_classifier_chain import classifier_chain_classifier

from machine_learning.ml_method.ml_knn_method import multilabel_ml_knn_classifier
from machine_learning.ml_method.xgb_method import multilabel_xgb_classifier
from machine_learning.ml_method.random_forest import multilabel_rf_classifier
from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
)


def load_data(dataset_type="cnn_features"):
    if dataset_type == "cnn_features":
        data_dir1 = f"machine_learning/data/Ischemia_Dataset/*/*/d64_features_dataset/"
        data_dir2 = f"machine_learning/data/Ischemia_Dataset/*/*/d64_cnn128_dataset/"
        X1, y1, meta = load_dataset(data_dir1, meta_required=True)
        X2, y2, _ = load_dataset(data_dir2)
        if X1.ndim == 3:
            X1 = X1.reshape(X1.shape[0], -1)
        if X2.ndim == 3:
            X2 = X2.reshape(X2.shape[0], -1)

        X = np.concatenate([X1, X2], axis=1)
        y = y1
    else:
        data_dir = (
            f"machine_learning/data/Ischemia_Dataset/*/*/d64_{dataset_type}_dataset/"
        )

        X, y, meta = load_dataset(data_dir, meta_required=True)

        if dataset_type == "processed":
            X = X[:, 200:400, :]

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

    return X, y, meta


if __name__ == "__main__":

    dataset_type = "cnn_features"
    X, y, meta = load_data(dataset_type)

    _, X_test, _, y_test, _, idx_test = split_dataset(X, y, random_state=42)

    methods = {
        'ML-KNN': multilabel_ml_knn_classifier,
        'Classifier Chain': classifier_chain_classifier,
        'Calibrated Label Ranking': calibrated_label_ranking_classifier,
        'Binary Relevance-XGB': multilabel_xgb_classifier,
        'Binary Relevance-Random Forest': multilabel_rf_classifier,
    }

    # Store results for plotting
    model_results = {}
    max_errors = 0

    total_samples = len(y_test)
    print(f"Total test samples: {total_samples}")

    for model_name, model_func in methods.items():
        print(f"--- Analyzing model: {model_name} ---")

        # The path construction depends on how models were saved.
        model_path = f"machine_learning/data/model/{dataset_type}/ml_model/{model_func.__name__}.joblib"

        if not os.path.exists(model_path):
            print(f"Model path does not exist: {model_path}")
            continue

        try:
            clf = load(model_path)
            # Depending on how the model was saved, it might be the classifier object itself
            # or a dictionary. Assuming it's the classifier object based on previous file.
            y_pred = clf.predict(X_test)
        except Exception as e:
            print(f"Error loading or predicting with model {model_name}: {e}")
            continue

        # Calculate mismatches per sample
        # y_pred and y_test are (N_samples, N_labels)
        # We want to sum mismatch over axis 1 (number of wrong labels per sample)
        mismatches = np.sum(y_test != y_pred, axis=1)

        # Count occurrences of each mismatch count
        unique, counts = np.unique(mismatches, return_counts=True)
        result_dict = dict(zip(unique, counts))

        # Calculate percentages
        prob_dict = {k: (v / total_samples) * 100 for k, v in result_dict.items()}
        model_results[model_name] = prob_dict

        current_max = max(unique) if len(unique) > 0 else 0
        if current_max > max_errors:
            max_errors = current_max

        print(f"Processed {model_name}")

    if not model_results:
        print("No models were successfully processed.")
        exit()

    # --- Plotting ---
    print("Generating plot...")

    # Define error categories (0 errors, 1 error, 2 errors, ..., max_errors)
    # If max_errors is very large, might want to limit it, but for multi-label usually it's small-ish number of labels.
    # Assuming number of labels is relatively small (e.g. < 20).

    # X-axis categories: 0, 1, 2, and optionally >=3 grouped together
    has_ge3_bin = max_errors >= 3
    if has_ge3_bin:
        x_labels = [0, 1, 2, 3]
        x_tick_labels = ['0', '1', '2', '>=3']
    else:
        x_labels = list(range(max_errors + 1))
        x_tick_labels = [str(x) for x in x_labels]

    # Prepare data for plotting
    # num_models = len(model_results)
    # bar_width = 0.8 / num_models
    # indices = np.arange(len(x_labels))

    plt.figure(figsize=(12, 6))

    # Using bar plot
    # To place bars side by side, we need to shift positions

    model_names = list(model_results.keys())
    num_models = len(model_names)
    bar_width = 0.15
    indices = np.arange(len(x_labels))

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for i, model_name in enumerate(model_names):
        data = model_results[model_name]
        if has_ge3_bin:
            # Aggregate all bins >=3 into one column.
            ge3_value = sum(v for k, v in data.items() if k >= 3)
            y_values = [data.get(0, 0), data.get(1, 0), data.get(2, 0), ge3_value]
        else:
            y_values = [data.get(err, 0) for err in x_labels]

        # Position of bars
        pos = indices + i * bar_width - (num_models * bar_width) / 2 + bar_width / 2

        plt.bar(
            pos,
            y_values,
            width=bar_width,
            label=model_name,
            color=colors[i % len(colors)],
        )

    plt.xlabel('Number of Misclassified Labels per Sample', fontsize=14)
    plt.ylabel('Percentage of Samples (%)', fontsize=14)
    # plt.title('Error Distribution by Model', fontsize=14)
    plt.xticks(indices, x_tick_labels)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Determine save path
    save_dir = "machine_learning/check_result"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, "error_distribution.png")
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")
    # plt.show()
