import torch.nn as nn
import torch.nn.functional as F
from utils.deep_learning_tools import (
    build_train_test_loaders,
    train_model,
    evaluate_model,
)


# --------------------
# Temporal Convolutional Network (TCN) Block
# --------------------
class TCNBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=5, dilation=1, dropout=0.3):
        super().__init__()
        self.conv = nn.Conv1d(
            input_dim,
            hidden_dim,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation // 2,
            dilation=dilation,
        )
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D) -> (B, D, T)
        x = x.transpose(1, 2)
        out = self.conv(x)
        out = out.transpose(1, 2)
        out = F.relu(self.ln(out))
        out = self.dropout(out)
        return out


# --------------------
# TCN Classifier
# --------------------
class TCNClassifier(nn.Module):
    def __init__(self, input_dim, n_classes=17, hidden_dim=128, layers=3):
        super().__init__()
        self.proj = nn.Conv1d(input_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                TCNBlock(hidden_dim, hidden_dim, kernel_size=5, dilation=2**i)
                for i in range(layers)
            ]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)  # (B, D, T)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x.transpose(1, 2)).transpose(1, 2)
        x = x.mean(dim=2)  # Global Average Pooling over time
        x = self.dropout(x)
        return self.fc(x)


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    data_dir = (
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/"
    )

    # 🔥 使用你之前写好的随机划分函数
    train_loader, test_loader = build_train_test_loaders(
        data_dir=data_dir, batch_size=32, test_ratio=0.2, num_workers=4
    )

    # 自动推断 input_dim（从 train_loader 第一个 batch）
    X_sample, _ = next(iter(train_loader))
    input_dim = X_sample.shape[-1]

    model = TCNClassifier(input_dim)
    model = train_model(model, train_loader, epochs=30, lr=1e-3)
    evaluate_model(model, test_loader)
