import os
import h5py
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split, DataLoader, Dataset
from sklearn.metrics import accuracy_score


import os
import h5py
import torch
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    def __init__(self, data_dirs, transform=None):
        """
        data_dirs: str 或 [str]
            一个或多个包含 .h5 文件的目录
        """
        # ----------- 处理多个目录 -----------
        if isinstance(data_dirs, str):
            data_dirs = [data_dirs]

        # 收集所有目录中的 h5 文件
        self.data_files = []
        for d in data_dirs:
            h5_files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.h5')]
            self.data_files.extend(h5_files)

        self.data_files = sorted(self.data_files)

        if len(self.data_files) == 0:
            raise ValueError(f"No .h5 files found in: {data_dirs}")

        self.transform = transform

        # ----------- 构建 index_map -----------
        self.index_map = []  # [(file_idx, sample_idx), ...]
        self.file_sample_counts = []  # 每个文件的样本数

        for f_idx, fpath in enumerate(self.data_files):
            with h5py.File(fpath, "r") as f:
                X = f["X"] if "X" in f else f[list(f.keys())[0]]
                n_samples = X.shape[0]

                # 读取 y，过滤掉 -1
                if "y" in f:
                    y = f["y"][:]
                else:
                    y = -1 * torch.ones(n_samples)

                for i in range(n_samples):
                    if y[i] != -1:
                        self.index_map.append((f_idx, i))

                self.file_sample_counts.append(n_samples)

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
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0

        for X, y in loader:
            X, y = X.to(device), y.to(device)

            optim.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            loss.backward()
            optim.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss / len(loader):.4f}")

    return model


def evaluate_model(model, loader, device="cuda"):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred_label = pred.argmax(dim=1)
            correct += (pred_label == y).sum().item()
            total += y.size(0)

    print(f"Test Accuracy: {correct / total:.4f}")
    return correct / total
