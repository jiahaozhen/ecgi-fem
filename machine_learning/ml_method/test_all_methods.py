import time
from machine_learning.ml_method.knn_method import multilabel_knn_ovr_classifier
from machine_learning.ml_method.lgb_method import multilabel_lgb_classifier
from machine_learning.ml_method.xgb_method import multilabel_xgb_classifier
from machine_learning.ml_method.linear_svm import multilabel_svm_classifier
from machine_learning.ml_method.random_forest import multilabel_rf_classifier
from machine_learning.ml_method.logistic_regression import (
    multilabel_logistic_classifier,
)

from utils.machine_learning_tools import load_dataset, exclude_classes, split_dataset

data_dir = [
    "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_processed_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_processed_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_processed_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_processed_dataset/",
]
X, y = load_dataset(data_dir)
X, y = exclude_classes(X, y, exclude_labels=[-1])
X_train, X_test, y_train, y_test = split_dataset(X, y)

methods = [
    ('KNN', multilabel_knn_ovr_classifier),
    ('LightGBM', multilabel_lgb_classifier),
    ('XGB', multilabel_xgb_classifier),
    ('SVM', multilabel_svm_classifier),
    ('Random Forest', multilabel_rf_classifier),
    ('Logistic Regression', multilabel_logistic_classifier),
]

results = []

for name, func in methods:
    print(f'\n训练 {name}...')
    start_time = time.time()
    try:
        clf = func(X_train, y_train)
        elapsed = time.time() - start_time
        print(f'{name}: 训练时间 = {elapsed:.4f}s')
        # 评估模型并记录准确度
        print(f'{name} 测试结果:')
        from sklearn.metrics import accuracy_score

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print('Accuracy:', acc)
        results.append(
            {'method': name, 'time': elapsed, 'accuracy': acc, 'error': None}
        )
    except Exception as e:
        elapsed = time.time() - start_time
        results.append(
            {'method': name, 'time': elapsed, 'accuracy': None, 'error': str(e)}
        )
        print(f'{name}: 错误: {e}')

print('\n训练完成:')
for r in results:
    print(r)
