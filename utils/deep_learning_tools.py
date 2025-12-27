import os
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, Dataset
from sklearn.metrics import (
    classification_report,
    hamming_loss,
    f1_score,
    accuracy_score,
)


import os
import h5py
import torch
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        """
        data_dirs: str 或 list[str]
            一个或多个包含 .h5 文件的目录
        """
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        # 收集所有 h5 文件
        self.data_files = []
        for d in data_dirs:
            self.data_files.extend(
                [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".h5")]
            )

        self.data_files = sorted(self.data_files)
        if len(self.data_files) == 0:
            raise ValueError(f"No .h5 files found in: {data_dirs}")

        self.transform = transform

        # ----------- 构建 index_map（不做任何过滤） -----------
        self.index_map = []  # (file_idx, sample_idx)

        for f_idx, fpath in enumerate(self.data_files):
            with h5py.File(fpath, "r") as f:
                X_key = "X" if "X" in f else list(f.keys())[0]
                n_samples = f[X_key].shape[0]

                for i in range(n_samples):
                    self.index_map.append((f_idx, i))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        file_idx, sample_idx = self.index_map[idx]
        fpath = self.data_files[file_idx]

        with h5py.File(fpath, "r") as f:
            X = f["X"][sample_idx] if "X" in f else f[list(f.keys())[0]][sample_idx]
            X = torch.tensor(X, dtype=torch.float32)

            if "y" in f:
                y = torch.tensor(f["y"][sample_idx], dtype=torch.long)
            else:
                y = torch.tensor(-1, dtype=torch.long)

        if self.transform:
            X = self.transform(X)

        meta = {
            "file": fpath,
            "file_idx": file_idx,
            "sample_idx": sample_idx,
            "global_idx": idx,
        }

        return X, y, meta


def build_train_test_loaders(
    data_dir,
    batch_size=32,
    test_ratio=0.2,
    num_workers=4,
    transform=None,
    seed=42,
):
    full_dataset = H5Dataset(data_dir, transform=transform)
    total_len = len(full_dataset)
    test_len = int(total_len * test_ratio)
    train_len = total_len - test_len

    # 固定随机种子
    g = torch.Generator()
    g.manual_seed(seed)

    train_set, test_set = random_split(full_dataset, [train_len, test_len], generator=g)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def train_model(
    model,
    loader,
    epochs=30,
    lr=1e-3,
    device="cuda",
    save_path=None,
    load_path=None,
):
    model.to(device)

    # --------------------
    # Load model (optional)
    # --------------------
    if load_path is not None and os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path, map_location=device))
        print(f"📥 Loaded model from {load_path}")
        return model

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # --------------------
    # Training
    # --------------------
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for X, y, _ in loader:
            X = X.to(device)
            y = y.to(device).float()

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # --------------------
    # Save LAST model
    # --------------------
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved last model to {save_path}")

    return model


def evaluate_model(model, loader, threshold=0.5, device="cuda"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y, _ in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    h_loss = hamming_loss(y_true, y_pred)
    f1_score_micro = f1_score(y_true, y_pred, average="micro")
    f1_score_macro = f1_score(y_true, y_pred, average="macro")
    a_score = accuracy_score(y_true, y_pred)

    # 3️⃣ 评估
    # print("Threshold:", threshold)
    print("Hamming loss:", h_loss)
    print("Micro F1:", f1_score_micro)
    print("Macro F1:", f1_score_macro)
    print("Accuracy Score:", a_score)
    # print(classification_report(y_true, y_pred, digits=4))

    return {
        "Hamming loss": h_loss,
        "Micro F1": f1_score_micro,
        "Macro F1": f1_score_macro,
        "accuracy score": a_score,
    }


def find_wrong_samples(model, dataloader, device="cuda", threshold=0.5):
    model.eval()

    wrong_samples = []

    with torch.no_grad():
        for x, y, meta in dataloader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            y_pred = (probs > threshold).int()  # (B, 17)

            # 任一标签预测错误即算错样本
            wrong_mask = (y_pred != y).any(dim=1)  # (B,)
            wrong_indices = torch.nonzero(wrong_mask, as_tuple=False).squeeze(1)

            for i in wrong_indices:
                wrong_samples.append(
                    {
                        "file": meta["file"][i],
                        "sample_idx": meta["sample_idx"][i].item(),
                        "x": x[i].detach().cpu(),
                        "y_pred": y_pred[i].detach().cpu(),
                        "y_true": y[i].detach().cpu(),
                    }
                )

    return wrong_samples


def save_wrong_samples(wrong_samples, save_path):

    with h5py.File(save_path, "w") as f:

        y_true = []
        y_pred = []
        sample_idx = []
        file = []

        for i, s in enumerate(wrong_samples):

            # ---- 标签统一成 numpy ----
            y_true.append(np.asarray(s["y_true"]))
            y_pred.append(np.asarray(s["y_pred"]))
            sample_idx.append(s.get("sample_idx", -1))
            file.append(s.get("file", ""))

        f.create_dataset("y_true", data=np.stack(y_true))
        f.create_dataset("y_pred", data=np.stack(y_pred))
        f.create_dataset("sample_idx", data=np.asarray(sample_idx))
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("file", data=np.asarray(file, dtype=object), dtype=dt)

    print(f"Saved {len(file)} samples to {save_path}")
