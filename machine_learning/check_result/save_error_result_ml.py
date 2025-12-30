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

    data_dir = ["machine_learning/data/Ischemia_Dataset_DR_flatten/"]

    X, y, meta = load_dataset(data_dir)
    assert meta is not None

    _, X_test, _, y_test, _, idx_test = split_dataset(X, y)

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
    model_path = f"machine_learning/data/model/ml_model/{model_func.__name__}.joblib"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    clf = model_func()
    clf = load(model_path)

    y_pred = clf.predict(X_test)

    wrong_mask = ~(y_pred == y_test).all(axis=1)
    wrong_indices = np.where(wrong_mask)[0]
    wrong_info = []

    src_file_ids = meta['src_file_id']
    src_indices = meta['src_index']
    file_names = meta['file_names']

    wrong_info = [
        {
            "file": file_names[src_file_ids[idx_test[i]]],
            "sample_idx": src_indices[idx_test[i]],
            "y_true": y_test[i],
            "y_pred": y_pred[i],
        }
        for i in wrong_indices
    ]

    error_samples_dir = "machine_learning/data/error_samples/ml"

    os.makedirs(error_samples_dir, exist_ok=True)

    save_path = os.path.join(error_samples_dir, f"{model_func.__name__}.h5")

    save_wrong_samples(wrong_info, save_path)
