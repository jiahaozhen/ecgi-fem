import time
from machine_learning.ml_method.ml_knn_method import multilabel_ml_knn_classifier
from machine_learning.ml_method.ml_classifier_chain import classifier_chain_classifier
from machine_learning.ml_method.ml_label_powerset import label_powerset_classifier
from machine_learning.ml_method.knn_method import multilabel_knn_ovr_classifier
from machine_learning.ml_method.lgb_method import multilabel_lgb_classifier
from machine_learning.ml_method.xgb_method import multilabel_xgb_classifier
from machine_learning.ml_method.linear_svm import multilabel_svm_classifier
from machine_learning.ml_method.random_forest import multilabel_rf_classifier
from machine_learning.ml_method.logistic_regression import (
    multilabel_logistic_classifier,
)
from machine_learning.ml_method.calibrated_label_ranking import (
    calibrated_label_ranking_classifier,
)
from machine_learning.ml_method.random_k_labelsets import random_k_labelsets_classifier
from machine_learning.ml_method.ml_decision_tree import (
    multilabel_decision_tree_classifier,
)
from machine_learning.ml_method.cml_method import multilabel_cml_classifier

from utils.machine_learning_tools import (
    load_dataset,
    split_dataset,
    train_model,
    evaluate_model,
)

dataset_type = "features"
# dataset_type = "processed"
# dataset_type = 'cnn_ae'
# dataset_type = 'cnn_ae_features'

data_dir = f"machine_learning/data/Ischemia_Dataset/*/*/d64_{dataset_type}_dataset/"
model_save_dir = f"machine_learning/data/model/{dataset_type}/ml_model/"

X, y, _ = load_dataset(data_dir)

if X.ndim == 3:
    X = X.reshape(X.shape[0], -1)

print(X.shape)

X_train, X_test, y_train, y_test, _, _ = split_dataset(X, y)

methods = [
    ('ML-KNN', multilabel_ml_knn_classifier),
    ('Classifier Chain', classifier_chain_classifier),
    ('Calibrated Label Ranking', calibrated_label_ranking_classifier),
    ('Random k-Labelsets', random_k_labelsets_classifier),
    ('Multi-Lablel Decision Tree', multilabel_decision_tree_classifier),
    ('Collective Multi-Label Classifier', multilabel_cml_classifier),
    ('Binary Relevance-KNN', multilabel_knn_ovr_classifier),
    ('Binary Relevance-LightGBM', multilabel_lgb_classifier),
    ('Binary Relevance-XGB', multilabel_xgb_classifier),
    # ('Binary Relevance-SVM', multilabel_svm_classifier),
    ('Binary Relevance-Random Forest', multilabel_rf_classifier),
    ('Binary Relevance-Logistic Regression', multilabel_logistic_classifier),
    ('Label Powerset', label_powerset_classifier),
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
