import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.machine_learning_tools import (
    load_dataset,
)
from utils.signal_processing_tools import (
    get_feature_names,
    get_statistical_feature_names,
)


# dataset_type = "statistical_features"
dataset_type = "features"
# dataset_type = "combined_features"
lead_config = 64  # Options: 64, 12


if dataset_type == "statistical_features":
    data_dir = [
        f"machine_learning/data/Ischemia_Dataset/normal_male/mild/d{lead_config}_statistical_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male/severe/d{lead_config}_statistical_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male/healthy/d{lead_config}_statistical_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male2/mild/d{lead_config}_statistical_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male2/severe/d{lead_config}_statistical_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d{lead_config}_statistical_features_dataset/",
    ]
    feature_names = get_statistical_feature_names()

elif dataset_type == "features":
    data_dir = [
        f"machine_learning/data/Ischemia_Dataset/normal_male/mild/d{lead_config}_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male/severe/d{lead_config}_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male/healthy/d{lead_config}_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male2/mild/d{lead_config}_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male2/severe/d{lead_config}_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d{lead_config}_features_dataset/",
    ]
    base_names = get_feature_names()
    # Prefix each feature name with the lead number
    feature_names = [
        f"Lead{lead}_{name}"
        for lead in range(1, lead_config + 1)
        for name in base_names
    ]
elif dataset_type == "combined_features":
    data_dir = [
        f"machine_learning/data/Ischemia_Dataset/normal_male/mild/d{lead_config}_combined_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male/severe/d{lead_config}_combined_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male/healthy/d{lead_config}_combined_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male2/mild/d{lead_config}_combined_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male2/severe/d{lead_config}_combined_features_dataset/",
        f"machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d{lead_config}_combined_features_dataset/",
    ]
    base_names = get_feature_names()
    # Prefix each feature name with the lead number
    feature_names_1 = [
        f"Lead{lead}_{name}"
        for lead in range(1, lead_config + 1)
        for name in base_names
    ]
    feature_names_2 = get_statistical_feature_names()
    feature_names = feature_names_1 + feature_names_2

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


def plot_comparison(data_gt, top_n):
    vals_gt, labels_gt, col_names = data_gt

    n_labels = len(col_names)

    # Create a figure with one subplot
    _, axes = plt.subplots(1, 1, figsize=(max(8, n_labels * 0.8), max(6, top_n * 1.0)))

    # Plot Ground Truth
    sns.heatmap(
        vals_gt,
        cmap="coolwarm",
        center=0,
        ax=axes,
        xticklabels=col_names,
        yticklabels=[f"Rank {i+1}" for i in range(top_n)],
        cbar_kws={'label': 'Pearson Correlation'},
        vmin=-1,
        vmax=1,
        annot=labels_gt,
        fmt="",
        annot_kws={"size": 6},
    )
    axes.set_title(f"[Ground Truth] Top {top_n} Correlated Features per Label")
    axes.set_ylabel("Correlation Rank")
    plt.tight_layout()
    plt.show()


# 1. Ground Truth (Original Labels on Test Set)
print("Calculating correlations for Ground Truth...")
data_gt = get_correlation_data(
    X,
    y,
    feature_names=feature_names,
    top_n=5,
)


print("Plotting comparison...")
plot_comparison(data_gt, top_n=5)
