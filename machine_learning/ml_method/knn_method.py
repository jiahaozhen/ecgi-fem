from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier

from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    evaluate_model,
)


def multilabel_knn_ovr_classifier(X_train, y_train, k=10):
    base_clf = KNeighborsClassifier(
        n_neighbors=k, weights="distance", metric="minkowski"  # 比 uniform 稳定
    )

    clf = OneVsRestClassifier(base_clf)
    clf.fit(X_train, y_train)
    return clf


if __name__ == "__main__":
    data_dir = [
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_processed_dataset/",
    ]

    X, y = load_dataset(data_dir)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    clf = multilabel_knn_ovr_classifier(X_train, y_train)

    evaluate_model(clf, X_test, y_test)
