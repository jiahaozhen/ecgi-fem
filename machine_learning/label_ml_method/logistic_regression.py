from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    evaluate_model,
)


def multilabel_logistic_classifier(X_train, y_train):
    base_clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  # 处理标签不平衡
        n_jobs=1,
    )

    clf = OneVsRestClassifier(base_clf)
    clf.fit(X_train, y_train)
    return clf


if __name__ == "__main__":
    data_dir = [
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
    ]

    X, y = load_dataset(data_dir)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    clf = multilabel_logistic_classifier(X_train, y_train)

    evaluate_model(clf, X_test, y_test)
