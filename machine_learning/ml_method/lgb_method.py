from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier

from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    evaluate_model,
)


def multilabel_lgb_classifier(X_train, y_train):
    base_clf = LGBMClassifier(
        objective="binary",
        n_estimators=60,  # ↓ 150 → 60
        learning_rate=0.1,
        num_leaves=15,  # ↓ 31 → 15
        max_depth=4,  # ↓ 6 → 4
        min_data_in_leaf=50,  # 🔥 关键：限制叶子
        subsample=0.7,
        colsample_bytree=0.7,
        class_weight=None,  # ❗先关掉
        n_jobs=1,
        random_state=42,
        verbose=-1,
    )

    clf = OneVsRestClassifier(base_clf, n_jobs=1)
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

    clf = multilabel_lgb_classifier(X_train, y_train)

    evaluate_model(clf, X_test, y_test)
