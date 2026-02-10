import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
import sys
import os

from utils.machine_learning_tools import load_dataset, split_dataset, train_model
from machine_learning.ml_method.xgb_method import multilabel_xgb_classifier


def plot_confusion_heatmap(
    X,
    y,
    func,
    model_save_dir,
    save_path="machine_learning/check_result/xgb_confusion_matrix.png",
):
    """
    Load model and data, predict, and plot a multi-label confusion heatmap.
    matrix[i, j] represents the fraction of samples where label j was predicted
    given that label i was true. (Row-normalized P(Pred=j | True=i))
    """

    # 2. Split Data (Ensure same split as training)
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test, _, _ = split_dataset(X, y, random_state=42)

    # 3. Load Model
    model_path = os.path.join(model_save_dir, func.__name__ + ".joblib")
    print(f"Loading model from: {model_path}")
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    clf = func()
    clf = train_model(clf, X_train, y_train, save_path=model_path, load_path=model_path)

    # 4. Predict
    print("Predicting on test set...")
    y_pred = clf.predict(X_test)

    # Ensure y_pred is numpy array
    if hasattr(y_pred, "toarray"):
        y_pred = y_pred.toarray()

    n_samples, n_labels = y_test.shape
    print(f"Test samples: {n_samples}, Labels: {n_labels}")

    # 5. Compute 'Confusion' Matrix (True Label vs Predicted Label Co-occurrence)
    # Element (i, j) = Count where True Label is i AND Predicted Label is j
    # Ideally, we want high values on diagonal (i=j)

    # Use float for division
    cm = np.dot(y_test.T, y_pred).astype(float)

    # Row Normalization: Divide by how many times label i was actually true
    # Result: P(Predicted = j | True = i)
    # "When label i is true, how often does the model predict label j?"

    true_label_counts = y_test.sum(axis=0)

    # Handle division by zero if a label never appears in test set
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_norm = cm / true_label_counts[:, None]

    cm_norm = np.nan_to_num(cm_norm, nan=0.0)

    # 6. Plotting
    plt.figure(figsize=(14, 12))

    # Create segment labels (assuming 17 segments for heart modeling)
    # If 17 labels, use indices 1-17. If different, use indices.
    if n_labels == 17:
        tick_labels = [str(i + 1) for i in range(n_labels)]
    else:
        tick_labels = [str(i) for i in range(n_labels)]

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        square=True,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("XGB Confusion Matrix (Row-Normalized: P(Pred=j | True=i))")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Confusion matrix heatmap saved to {save_path}")

    # Also save the raw counts for reference
    raw_save_path = save_path.replace(".png", "_raw_counts.png")
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".0f",
        cmap="Reds",
        square=True,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    # plt.title("XGB Confusion Matrix (Raw Counts)")
    plt.tight_layout()
    plt.savefig(raw_save_path)
    print(f"✅ Raw counts heatmap saved to {raw_save_path}")


if __name__ == "__main__":
    dataset_type = "cnn_features"
    # dataset_type = "features"
    # dataset_type = "processed"
    # dataset_type = "cnn128"

    if dataset_type == "cnn_features":
        data_dir1 = f"machine_learning/data/Ischemia_Dataset/*/*/d64_features_dataset/"
        data_dir2 = f"machine_learning/data/Ischemia_Dataset/*/*/d64_cnn128_dataset/"
        model_save_dir = f"machine_learning/data/model/cnn_features/ml_model/"
        X1, y1, _ = load_dataset(data_dir1)
        X2, y2, _ = load_dataset(data_dir2)
        if X1.ndim == 3:
            X1 = X1.reshape(X1.shape[0], -1)
        if X2.ndim == 3:
            X2 = X2.reshape(X2.shape[0], -1)
        import numpy as np

        X = np.concatenate([X1, X2], axis=1)
        y = y1
    else:
        data_dir = (
            f"machine_learning/data/Ischemia_Dataset/*/*/d64_{dataset_type}_dataset/"
        )
        model_save_dir = f"machine_learning/data/model/{dataset_type}/ml_model/"

        X, y, _ = load_dataset(data_dir)

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

    func = multilabel_xgb_classifier

    plot_confusion_heatmap(X, y, func=func, model_save_dir=model_save_dir)
