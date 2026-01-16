from skmultilearn.problem_transform import ClassifierChain
from sklearn.ensemble import RandomForestClassifier

from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    train_model,
    evaluate_model,
)


def classifier_chain_classifier(base_classifier=None):
    """
    Classifier Chains method constructs a chain of binary classifiers C0, C1, ..., Cn.
    Each classifier predicts the binary association of label li given the feature space
    plus all the label predictions of the previous classifiers in the chain.
    """
    if base_classifier is None:
        base_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    classifier = ClassifierChain(
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

    # Test Classifier Chain
    print("\n--- Testing Classifier Chain ---")
    clf = classifier_chain_classifier()
    clf = train_model(clf, X_train, y_train)
    evaluate_model(clf, X_test, y_test)
