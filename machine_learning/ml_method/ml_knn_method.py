import skmultilearn.adapt.mlknn
from sklearn.neighbors import NearestNeighbors as SkNearestNeighbors
from skmultilearn.adapt import MLkNN

from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    train_model,
    evaluate_model,
)


# Monkey patch NearestNeighbors to handle positional arguments
class PatchedNearestNeighbors(SkNearestNeighbors):
    def __init__(
        self,
        n_neighbors=5,
        radius=1.0,
        algorithm='auto',
        leaf_size=30,
        metric='minkowski',
        p=2,
        metric_params=None,
        n_jobs=None,
    ):
        super().__init__(
            n_neighbors=n_neighbors,
            radius=radius,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric=metric,
            p=p,
            metric_params=metric_params,
            n_jobs=n_jobs,
        )


skmultilearn.adapt.mlknn.NearestNeighbors = PatchedNearestNeighbors


def multilabel_ml_knn_classifier(k=5, s=1.0):
    classifier = MLkNN(k=k, s=s)
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

    print(X.shape)

    X_train, X_test, y_train, y_test, _, _ = split_dataset(X, y)

    clf = multilabel_ml_knn_classifier()

    clf = train_model(clf, X_train, y_train)

    evaluate_model(clf, X_test, y_test)
