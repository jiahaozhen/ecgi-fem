import time
from machine_learning.ml_method.ml_knn_method import multilabel_ml_knn_classifier
from machine_learning.ml_method.ml_classifier_chain import classifier_chain_classifier
from machine_learning.ml_method.ml_label_powerset import label_powerset_classifier
from machine_learning.ml_method.xgb_method import multilabel_xgb_classifier
from machine_learning.ml_method.random_forest import multilabel_rf_classifier
from machine_learning.ml_method.calibrated_label_ranking import (
    calibrated_label_ranking_classifier,
)
from machine_learning.ml_method.random_k_labelsets import random_k_labelsets_classifier
from machine_learning.ml_method.ml_decision_tree import (
    multilabel_decision_tree_classifier,
)

from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    train_model,
    evaluate_model,
)

# dataset_type = "cnn_features"
# dataset_type = "features"
dataset_type = "processed"
# dataset_type = "cnn128"

if dataset_type == "cnn_features":
    data_dir1 = f"machine_learning/data/Ischemia_Dataset/*/*/d64_features_dataset/"
    data_dir2 = f"machine_learning/data/Ischemia_Dataset/*/*/d64_cnn128_dataset/"
    model_save_dir = f"machine_learning/data/model/cnn_features/ml_model/"
    X1, y1, _ = load_dataset(data_dir1)
    X2, y2, _ = load_dataset(data_dir2)
    if X1.ndim == 3:
        X1 = X1.reshape(X1.shape[0], -1)
    if X2.ndim == 3:
        X2 = X2.reshape(X2.shape[0], -1)
    import numpy as np

    X = np.concatenate([X1, X2], axis=1)
    y = y1
else:
    data_dir = f"machine_learning/data/Ischemia_Dataset/*/*/d64_{dataset_type}_dataset/"
    model_save_dir = f"machine_learning/data/model/{dataset_type}/ml_model/"

    X, y, _ = load_dataset(data_dir)

    if dataset_type == "processed":
        X = X[:, 200:400, :]

    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)

    print(X.shape)

X_train, X_test, y_train, y_test, _, _ = split_dataset(X, y)

methods = [
    # ('ML-KNN', multilabel_ml_knn_classifier),
    # ('Classifier Chain', classifier_chain_classifier),
    # ('Calibrated Label Ranking', calibrated_label_ranking_classifier),
    # ('Random k-Labelsets', random_k_labelsets_classifier),
    # ('Multi-Lablel Decision Tree', multilabel_decision_tree_classifier),
    ('Binary Relevance-XGB', multilabel_xgb_classifier),
    # ('Binary Relevance-Random Forest', multilabel_rf_classifier),
    # ('Label Powerset', label_powerset_classifier),
]

results = {}

for name, func in methods:
    print(f'\n训练 {name}...')
    model_path = f"{model_save_dir}/{func.__name__}.joblib"
    start_time = time.time()
    try:
        clf = func()
        clf = train_model(
            clf, X_train, y_train, save_path=model_path, load_path=model_path
        )
        elapsed = time.time() - start_time
        print(f'{name}: 训练时间 = {elapsed:.4f}s')
        # 评估模型并记录准确度
        print(f'{name} 测试结果:')
        metrics = evaluate_model(clf, X_test, y_test)
        results[name] = metrics
    except Exception as e:
        elapsed = time.time() - start_time
        print(f'{name}: 错误: {e}')

print('\n训练完成:')
for name, metrics in results.items():
    print(f"Method: {name}, Metrics: {metrics}")
