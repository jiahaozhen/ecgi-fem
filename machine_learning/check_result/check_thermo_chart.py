# 查看模型的训练结果热力图

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    train_model,
)
from machine_learning.ml_method.xgb_method import multilabel_xgb_classifier


# dataset_type = "statistical_features"
dataset_type = "features"


if dataset_type == "statistical_features":
    data_dir = [
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_statistical_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_statistical_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_statistical_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_statistical_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_statistical_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_statistical_features_dataset/",
    ]
    model_save_dir = "machine_learning/data/model/statistical_features/ml_model"
    feature_names = [
        'ST_level_60_mean',
        'ST_level_60_std',
        'ST_level_80_mean',
        'ST_level_80_std',
        'ST_slope_mean',
        'ST_slope_std',
        'ST_area_mean',
        'ST_area_std',
        'ST_min_mean',
        'ST_min_std',
        'ST_mean_mean',
        'ST_mean_std',
        'T_peak_amplitude_mean',
        'T_peak_amplitude_std',
        'T_width_mean',
        'T_width_std',
        'T_area_mean',
        'T_area_std',
    ]

elif dataset_type == "features":
    data_dir = [
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_features_dataset/",
    ]
    model_save_dir = "machine_learning/data/model/features/ml_model"
    base_names = [
        'R_time',
        'J_time',
        'T_peak_time',
        'ST_level_60',
        'ST_level_80',
        'ST_slope',
        'ST_area',
        'ST_min',
        'ST_mean',
        'T_peak_amplitude',
        'T_peak_latency',
        'T_width',
        'T_area',
        'T_sign',
    ]
    # Prefix each feature name with the lead number (1..64)
    feature_names = [
        f"Lead{lead}_{name}" for lead in range(1, 65) for name in base_names
    ]

else:
    raise ValueError(f"Unknown dataset_type: {dataset_type}")

X, y, _ = load_dataset(data_dir)

if X.ndim == 3:
    X = X.reshape(X.shape[0], -1)


def get_correlation_data(X, y, feature_names=None, label_names=None, top_n=20):
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(X.shape[1])]
    if label_names is None:
        label_names = [f"Class {i}" for i in range(y.shape[1])]

    # Calculate correlation between each feature and each label
    n_features = X.shape[1]
    n_labels = y.shape[1]
    corr_matrix = np.zeros((n_features, n_labels))

    # Safe correlation handling constant columns
    for i in range(n_features):
        feature_vec = X[:, i]
        if np.std(feature_vec) == 0:
            continue
        for j in range(n_labels):
            label_vec = y[:, j]
            if np.std(label_vec) == 0:
                continue

            # Calculate Pearson correlation
            corr_matrix[i, j] = np.corrcoef(feature_vec, label_vec)[0, 1]

    # Initialize matrices for the ranked results
    ranked_corr_values = np.zeros((top_n, n_labels))
    ranked_feature_labels = np.empty((top_n, n_labels), dtype=object)

    for j in range(n_labels):
        # Get correlations for this label
        label_corrs = corr_matrix[:, j]

        # Sort by absolute correlation (descending)
        top_indices = np.argsort(np.abs(label_corrs))[::-1][:top_n]

        # Fill the rank matrices
        ranked_corr_values[:, j] = label_corrs[top_indices]
        for rank, idx in enumerate(top_indices):
            # Annotate with Feature Name and Correlation Value
            ranked_feature_labels[rank, j] = (
                f"{feature_names[idx]}\n({label_corrs[idx]:.2f})"
            )

    return ranked_corr_values, ranked_feature_labels, label_names


def plot_comparison(data_gt, data_pred, top_n):
    vals_gt, labels_gt, col_names = data_gt
    vals_pred, labels_pred, _ = data_pred

    n_labels = len(col_names)

    # Create a figure with two subplots
    _, axes = plt.subplots(2, 1, figsize=(max(8, n_labels * 0.8), max(6, top_n * 1.0)))

    # Plot Ground Truth
    sns.heatmap(
        vals_gt,
        cmap="coolwarm",
        center=0,
        ax=axes[0],
        xticklabels=col_names,
        yticklabels=[f"Rank {i+1}" for i in range(top_n)],
        cbar_kws={'label': 'Pearson Correlation'},
        vmin=-1,
        vmax=1,
        annot=labels_gt,
        fmt="",
        annot_kws={"size": 6},
    )
    axes[0].set_title(f"[Ground Truth] Top {top_n} Correlated Features per Label")
    axes[0].set_ylabel("Correlation Rank")

    # Plot Model Predictions
    sns.heatmap(
        vals_pred,
        cmap="coolwarm",
        center=0,
        ax=axes[1],
        xticklabels=col_names,
        yticklabels=[f"Rank {i+1}" for i in range(top_n)],
        cbar_kws={'label': 'Pearson Correlation'},
        vmin=-1,
        vmax=1,
        annot=labels_pred,
        fmt="",
        annot_kws={"size": 6},
    )
    axes[1].set_title(f"[Model Predictions] Top {top_n} Correlated Features per Label")
    axes[1].set_xlabel("Results (Labels)")
    axes[1].set_ylabel("Correlation Rank")

    plt.tight_layout()
    plt.show()


# feature_names is already set based on dataset_type

# Split Dataset
print("Splitting Dataset...")
X_train, X_test, y_train, y_test, _, _ = split_dataset(X, y)

# 1. Ground Truth (Original Labels on Test Set)
print("Calculating correlations for Ground Truth...")
data_gt = get_correlation_data(
    X_test,
    y_test,
    feature_names=feature_names,
    top_n=5,
)

# 2. Training Results (Predicted Labels on Test Set)
print("Training Model (XGBoost)...")
# Using XGBoost as default
clf = train_model(
    multilabel_xgb_classifier(),
    X_train,
    y_train,
    load_path=f"{model_save_dir}/multilabel_xgb_classifier.joblib",
)
print("Predicting on Test Set...")
y_pred = clf.predict(X_test)

print("Calculating correlations for Training Results...")
data_pred = get_correlation_data(
    X_test,
    y_pred,
    feature_names=feature_names,
    top_n=5,
)

print("Plotting comparison...")
plot_comparison(data_gt, data_pred, top_n=5)
