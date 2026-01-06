# 查看标签共现（Label Co-occurrence）热力图
from utils.machine_learning_tools import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = [
    "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_features_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_features_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_features_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_features_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_features_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_features_dataset/",
]

print("Loading dataset...")
X, y, _ = load_dataset(data_dir)

print(f"Dataset shape: {y.shape}")
n_labels = y.shape[1]
label_names = [f"Class {i}" for i in range(n_labels)]

# Calculate Co-occurrence Matrix
# y is (n_samples, n_labels), so y.T @ y gives (n_labels, n_labels)
# Entry [i, j] counts how many samples have both label i and label j
cooccurrence_matrix = np.dot(y.T, y)

# It's often useful to normalize this somewhat, e.g., to conditional probabilities or Jaccard
# For a pure co-occurrence heatmap, raw counts or normalized by diagonal (self-occurrence) is common.
# Let's show raw counts first, but maybe handle diagonal separately if it dominates.

# We will zero out the diagonal for the heatmap visualization to see off-diagonal relationships better,
# or keep it to see label balance. Let's keep it but maybe use a max value for color scaling if needed.

plt.figure(figsize=(12, 10))
sns.heatmap(
    cooccurrence_matrix,
    annot=True,
    fmt="d",
    cmap="YlOrRd",
    xticklabels=label_names,
    yticklabels=label_names,
)
plt.title(
    "Label Co-occurrence Matrix (Raw Counts)\n(How often do two labels appear together?)"
)
plt.xlabel("Labels")
plt.ylabel("Labels")
plt.tight_layout()
plt.show()

# Optional: Normalized Co-occurrence (Correlation)
# This shows if the presence of one label is correlated with another, removing the effect of label frequency.
print("Calculating Label Correlation Matrix...")
corr_matrix = np.corrcoef(y.T)

plt.figure(figsize=(12, 10))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    xticklabels=label_names,
    yticklabels=label_names,
    vmin=-1,
    vmax=1,
)
plt.title("Label Correlation Matrix\n(Pearson Correlation between Labels)")
plt.xlabel("Labels")
plt.ylabel("Labels")
plt.tight_layout()
plt.show()
