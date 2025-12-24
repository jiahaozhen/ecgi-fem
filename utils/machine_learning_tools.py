import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    hamming_loss,
)
import h5py


def load_dataset(data_dir):
    if isinstance(data_dir, str):
        data_dir = [data_dir]

    X_list, y_list = [], []

    for d in data_dir:
        assert os.path.isdir(d), f"{d} not found"

        for f in sorted(os.listdir(d)):
            if f.endswith(".h5"):
                with h5py.File(os.path.join(d, f), "r") as data:
                    X_list.append(data["X"][:])
                    y_list.append(data["y"][:])
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


def flatten_data(data: np.ndarray):
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    return data


def split_dataset(X, y, test_size=0.2, random_state=42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )


def get_train_test(data_dir, test_size=0.2, random_state=42, test_only=False):
    X, y = load_dataset(data_dir)
    if test_only:
        # 全部数据作为测试集
        return None, X, None, y
    else:
        return split_dataset(X, y, test_size=test_size, random_state=random_state)


def evaluate_model(clf, X_test, y_test, threshold=0.5):
    # 1️⃣ 预测概率（每个标签一个概率）
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)
        y_pred = (y_score >= threshold).astype(int)
    else:
        y_score = clf.decision_function(X_test)
        y_pred = (y_score > 0).astype(int)

    h_loss = hamming_loss(y_test, y_pred)
    f1_score_micro = f1_score(y_test, y_pred, average="micro")
    f1_score_macro = f1_score(y_test, y_pred, average="macro")
    a_score = accuracy_score(y_test, y_pred)

    # 3️⃣ 评估
    print("Hamming loss:", h_loss)
    print("Micro F1:", f1_score_micro)
    print("Macro F1:", f1_score_macro)
    print("Accuracy Score:", a_score)
    # print(classification_report(y_test, y_pred, digits=4))

    return {
        "Hamming loss": h_loss,
        "Micro F1": f1_score_micro,
        "Macro F1": f1_score_macro,
        "accuracy score": a_score,
    }
