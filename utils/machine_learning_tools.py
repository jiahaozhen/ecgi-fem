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
from joblib import dump, load


def load_dataset(data_dir):
    if isinstance(data_dir, str):
        data_dir = [data_dir]

    X_list, y_list = [], []

    src_file_id_list = []
    src_index_list = []
    file_names = None

    has_meta = True  # 只要有一个 h5 缺字段，就整体判定没有 meta

    for d in data_dir:
        assert os.path.isdir(d), f"{d} not found"

        for fname in sorted(os.listdir(d)):
            if not fname.endswith(".h5"):
                continue

            path = os.path.join(d, fname)

            with h5py.File(path, "r") as f:
                X_list.append(f["X"][:])
                y_list.append(f["y"][:])

                # ===== 检查 meta 是否存在 =====
                if "src_file_id" in f and "src_index" in f and "file_names" in f:
                    src_file_id_list.append(f["src_file_id"][:])
                    src_index_list.append(f["src_index"][:])

                    if file_names is None:
                        file_names = [
                            n.decode() if isinstance(n, bytes) else str(n)
                            for n in f["file_names"][:]
                        ]
                else:
                    has_meta = False

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    if not has_meta:
        return X, y, None

    meta = {
        "src_file_id": np.concatenate(src_file_id_list),
        "src_index": np.concatenate(src_index_list),
        "file_names": file_names,
    }

    return X, y, meta


def train_model(clf, X_train, y_train, save_path=None, load_path=None):
    if load_path is not None and os.path.exists(load_path):
        clf = load(load_path)
        print(f"📥 Loaded model from {load_path}")
        return clf

    clf.fit(X_train, y_train)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        dump(clf, save_path)
        print(f"💾 Saved model to {save_path}")

    return clf


def flatten_data(data: np.ndarray):
    if data.ndim == 3:
        data = data.reshape(data.shape[0], -1)
    return data


def split_dataset(X, y, test_size=0.2, random_state=42):
    idx = np.arange(len(X))
    return train_test_split(
        X,
        y,
        idx,
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
