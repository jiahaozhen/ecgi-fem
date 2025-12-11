# ----------------- 机器学习通用工具 -----------------
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import h5py


def load_dataset(data_dir):
    file_list = sorted([f for f in os.listdir(data_dir) if f.endswith('.h5')])
    X_list, y_list = [], []
    for fname in file_list:
        with h5py.File(os.path.join(data_dir, fname), 'r') as data:
            X = data['X'][:] if 'X' in data else data[list(data.keys())[0]]
            y = data['y'][:] if 'y' in data else data[list(data.keys())[-1]]
            X_list.append(X)
            y_list.append(y)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    if X.ndim == 3:
        X = X.reshape(X.shape[0], -1)
    return X, y


def exclude_classes(X, y, exclude_labels):
    mask = ~np.isin(y, exclude_labels)
    return X[mask], y[mask]


def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def get_train_test(data_dir, test_size=0.2, random_state=42, test_only=False):
    X, y = load_dataset(data_dir)
    if test_only:
        # 全部数据作为测试集
        return None, X, None, y
    else:
        return split_dataset(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(clf, X_test, y_test):
    y_pred_label = clf.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred_label))
    print(classification_report(y_test, y_pred_label, digits=4))
