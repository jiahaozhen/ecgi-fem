from skmultilearn.problem_transform import BinaryRelevance
from sklearn.ensemble import RandomForestClassifier

from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    train_model,
    evaluate_model,
)


def binary_relevance_classifier(base_classifier=None):
    """
    Binary Relevance method transforms a multi-label classification problem with L labels
    into L single-label binary classification problems.
    """
    if base_classifier is None:
        base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    classifier = BinaryRelevance(
        classifier=base_classifier, require_dense=[False, True]
    )
    return classifier


if __name__ == "__main__":
    data_dir = [
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_features_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_features_dataset/",
    ]

    X, y, _ = load_dataset(data_dir)

    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    print(f"Data shape: {X.shape}")
    print(f"Labels shape: {y.shape}")

    X_train, X_test, y_train, y_test, _, _ = split_dataset(X, y)

    # Test Binary Relevance
    print("\n--- Testing Binary Relevance ---")
    clf = binary_relevance_classifier()
    clf = train_model(clf, X_train, y_train)
    evaluate_model(clf, X_test, y_test)
