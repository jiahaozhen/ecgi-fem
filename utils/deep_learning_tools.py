import os
import h5py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader, Dataset
from sklearn.metrics import classification_report, hamming_loss, f1_score


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

        return X, y


def build_train_test_loaders(
    data_dir, batch_size=32, test_ratio=0.2, num_workers=4, transform=None
):
    full_dataset = H5Dataset(data_dir, transform=transform)
    total_len = len(full_dataset)
    test_len = int(total_len * test_ratio)
    train_len = total_len - test_len

    train_set, test_set = random_split(full_dataset, [train_len, test_len])

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


def train_model(model, loader, epochs=30, lr=1e-3, device="cuda"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for X, y in loader:
            X = X.to(device)
            y = y.to(device).float()  # (B, n_classes)

            optimizer.zero_grad()
            logits = model(X)  # (B, n_classes)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss / len(loader):.4f}")

    return model


def evaluate_model(model, loader, threshold=0.5, device="cuda"):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            probs = torch.sigmoid(logits)
            preds = (probs > threshold).int()

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()
    print("Hamming loss:", hamming_loss(y_true, y_pred))
    print("Micro F1:", f1_score(y_true, y_pred, average="micro"))
    print("Macro F1:", f1_score(y_true, y_pred, average="macro"))
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    return f1_score(y_true, y_pred, average="macro")
