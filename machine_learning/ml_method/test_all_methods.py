import time
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
    train_model,
    evaluate_model,
)

data_dir = [
    "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_features_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_features_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_features_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_features_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_features_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_features_dataset/",
]

# data_dir = ["machine_learning/data/Ischemia_Dataset_flat_pca/"]

# data_dir = [
#     "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
#     "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_processed_dataset/",
#     "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_processed_dataset/",
#     "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
#     "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_processed_dataset/",
#     "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_processed_dataset/",
# ]

X, y, _ = load_dataset(data_dir)

if X.ndim == 3:
    X = X.reshape(X.shape[0], -1)

print(X.shape)

X_train, X_test, y_train, y_test, _, _ = split_dataset(X, y)

methods = [
    ('KNN', multilabel_knn_ovr_classifier),
    ('LightGBM', multilabel_lgb_classifier),
    ('XGB', multilabel_xgb_classifier),
    # ('SVM', multilabel_svm_classifier),
    ('Random Forest', multilabel_rf_classifier),
    ('Logistic Regression', multilabel_logistic_classifier),
]

results = {}

for name, func in methods:
    print(f'\n训练 {name}...')
    model_path = f"machine_learning/data/model/features/ml_model/{func.__name__}.joblib"
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
