from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    train_model,
    evaluate_model,
)


def multilabel_xgb_classifier():

    base_clf = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,  # 300 → 120
        max_depth=6,  # 6 → 4
        learning_rate=0.05,  # 0.05 → 0.1
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        n_jobs=2,
    )

    clf = OneVsRestClassifier(base_clf)

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

    model_path = f"machine_learning/data/model/ml_model/{multilabel_xgb_classifier.__name__}.joblib"

    X, y, _ = load_dataset(data_dir)

    X_train, X_test, y_train, y_test, _, _ = split_dataset(X, y)

    clf = multilabel_xgb_classifier()

    clf = train_model(clf, X_train, y_train, save_path=model_path, load_path=model_path)

    evaluate_model(clf, X_test, y_test)
