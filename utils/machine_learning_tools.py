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


def evaluate_model(clf, X_test, y_test, threshold=None):
    # 1️⃣ 预测概率（每个标签一个概率）
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_test)

        # Handle sklearn's list output for multi-label (e.g. Decision Tree / RF)
        if isinstance(y_score, list):
            output_list = []
            for i, preds in enumerate(y_score):
                if preds.shape[1] == 2:
                    # Normal case: [prob_0, prob_1]
                    output_list.append(preds[:, 1])
                elif preds.shape[1] == 1:
                    # Single class case
                    if hasattr(clf, 'classes_') and len(clf.classes_[i]) == 1:
                        if clf.classes_[i][0] == 1:
                            output_list.append(preds[:, 0])
                        else:
                            output_list.append(np.zeros_like(preds[:, 0]))
                    else:
                        # Fallback
                        output_list.append(np.zeros_like(preds[:, 0]))
                else:
                    # >2 classes? Take last one?
                    output_list.append(preds[:, -1])
            y_score = np.array(output_list).T

        if hasattr(y_score, "toarray"):
            y_score = y_score.toarray()

        if threshold is None:
            best_f1 = -1
            best_th = 0.5
            for th in np.arange(0.01, 1.0, 0.01):
                y_pred_tmp = (y_score >= th).astype(int)
                score = f1_score(y_test, y_pred_tmp, average="micro")
                if score > best_f1:
                    best_f1 = score
                    best_th = th
            threshold = best_th
            print(f"Best threshold: {threshold:.3f}, F1-Micro: {best_f1:.4f}")

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
