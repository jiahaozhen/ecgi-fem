import torch.nn as nn
from utils.deep_learning_tools import (
    build_train_test_loaders,
    train_model,
    evaluate_model,
)


# --------------------
# GRU Block
# --------------------
class GRUBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.3):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.gru(x)
        return self.dropout(out)


# --------------------
# BiGRU Classifier
# --------------------
class BiGRUClassifier(nn.Module):
    def __init__(self, input_dim, n_classes=17, hidden_dim=128, num_layers=2):
        super().__init__()
        self.bigru = GRUBlock(input_dim, hidden_dim, num_layers)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, n_classes)

    def forward(self, x):
        # x: (B, T, D)
        out = self.bigru(x)  # (B, T, 2*hidden_dim)
        out = out.mean(dim=1)  # GAP over time -> (B, 2*hidden_dim)
        out = self.dropout(out)
        return self.fc(out)  # (B, n_classes)


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
    X_sample, _, _ = next(iter(train_loader))
    input_dim = X_sample.shape[-1]

    # 构建模型
    model = BiGRUClassifier(input_dim=input_dim)
    model = train_model(model, train_loader, epochs=30, lr=1e-3)
    evaluate_model(model, test_loader)
