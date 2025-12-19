import torch.nn as nn
import torch.nn.functional as F
from utils.deep_learning_tools import (
    build_train_test_loaders,
    train_model,
    evaluate_model,
)


# --------------------
# Residual Block
# --------------------
class ResidualBlock(nn.Module):
    def __init__(self, dim, kernel=5):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2)
        self.ln1 = nn.LayerNorm(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=kernel, padding=kernel // 2)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, T)
        residual = x
        out = self.conv1(x)
        out = F.relu(self.ln1(out.transpose(1, 2)).transpose(1, 2))
        out = self.conv2(out)
        out = self.ln2(out.transpose(1, 2)).transpose(1, 2)
        return F.relu(out + residual)


# --------------------
# Improved CNN Classifier
# --------------------
class ImprovedCNN(nn.Module):
    def __init__(self, input_dim, n_classes=17):
        super().__init__()

        hidden = 128

        self.proj = nn.Conv1d(input_dim, hidden, kernel_size=3, padding=1)
        self.block1 = ResidualBlock(hidden)
        self.block2 = ResidualBlock(hidden)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, D) → (B, D, T)
        x = F.relu(self.proj(x))
        x = self.block1(x)
        x = self.block2(x)
        x = x.mean(dim=2)  # GAP
        x = self.dropout(x)
        return self.fc(x)


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    data_dir = [
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_processed_dataset/",
    ]

    # 🔥 使用你之前写好的随机划分函数
    train_loader, test_loader = build_train_test_loaders(
        data_dir=data_dir, batch_size=32, test_ratio=0.2, num_workers=4
    )

    # 自动推断 input_dim（从 train_loader 第一个 batch）
    X_sample, _ = next(iter(train_loader))
    input_dim = X_sample.shape[-1]

    model = ImprovedCNN(input_dim)
    model = train_model(model, train_loader, epochs=30, lr=1e-3)
    evaluate_model(model, test_loader)
