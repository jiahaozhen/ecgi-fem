import numpy as np
import os
from joblib import load
from machine_learning.ml_method.knn_method import multilabel_knn_ovr_classifier
from machine_learning.ml_method.lgb_method import multilabel_lgb_classifier
from machine_learning.ml_method.xgb_method import multilabel_xgb_classifier
from machine_learning.ml_method.linear_svm import multilabel_svm_classifier
from machine_learning.ml_method.random_forest import multilabel_rf_classifier
from machine_learning.ml_method.logistic_regression import (
    multilabel_logistic_classifier,
)
from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
)
from utils.deep_learning_tools import save_wrong_samples


if __name__ == "__main__":

    dataset_type = "cnn_features"
    # dataset_type = "features"
    # dataset_type = "processed"
    # dataset_type = "cnn128"

    if dataset_type == "cnn_features":
        data_dir1 = f"machine_learning/data/Ischemia_Dataset/*/*/d64_features_dataset/"
        data_dir2 = f"machine_learning/data/Ischemia_Dataset/*/*/d64_cnn128_dataset/"
        X1, y1, meta = load_dataset(data_dir1, meta_required=True)
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

        X, y, meta = load_dataset(data_dir, meta_required=True)

        if dataset_type == "processed":
            X = X[:, 200:400, :]

        if X.ndim == 3:
            X = X.reshape(X.shape[0], -1)

    _, X_test, _, y_test, _, idx_test = split_dataset(X, y)

    assert meta is not None, "Meta information is required for error analysis"

    methods = {
        'KNN': multilabel_knn_ovr_classifier,
        'LightGBM': multilabel_lgb_classifier,
        'XGB': multilabel_xgb_classifier,
        'SVM': multilabel_svm_classifier,
        'Random Forest': multilabel_rf_classifier,
        'Logistic Regression': multilabel_logistic_classifier,
    }

    model_name = "XGB"
    model_func = methods[model_name]
    model_path = f"machine_learning/data/model/{dataset_type}/ml_model/{model_func.__name__}.joblib"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    clf = model_func()
    clf = load(model_path)

    y_pred = clf.predict(X_test)

    wrong_mask = ~(y_pred == y_test).all(axis=1)
    wrong_indices = np.where(wrong_mask)[0]
    wrong_info = []

    wrong_info = [
        {
            "file": meta["file"][idx_test[i]],
            "sample_idx": meta["sample_idx"][idx_test[i]],
            "y_true": y_test[i],
            "y_pred": y_pred[i],
        }
        for i in wrong_indices
    ]

    error_samples_dir = f"machine_learning/data/error_samples/{dataset_type}/ml/"

    os.makedirs(error_samples_dir, exist_ok=True)

    save_path = os.path.join(error_samples_dir, f"{model_func.__name__}.h5")

    save_wrong_samples(wrong_info, save_path)
