from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier

from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    train_model,
    evaluate_model,
)


def multilabel_lgb_classifier():
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
    return clf


if __name__ == "__main__":
    # data_dir = [
    #     "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
    #     "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_processed_dataset/",
    #     "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_processed_dataset/",
    #     "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
    #     "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_processed_dataset/",
    #     "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_processed_dataset/",
    # ]

    data_dir = ["machine_learning/data/Ischemia_Dataset_DR_flatten/"]

    model_path = f"machine_learning/data/model/ml_model/{multilabel_lgb_classifier.__name__}.joblib"

    X, y, _ = load_dataset(data_dir)

    X_train, X_test, y_train, y_test, _, _ = split_dataset(X, y)

    clf = multilabel_lgb_classifier()

    clf = train_model(clf, X_train, y_train, save_path=model_path, load_path=model_path)

    evaluate_model(clf, X_test, y_test)
