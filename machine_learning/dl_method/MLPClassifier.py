import torch.nn as nn
import torch.nn.functional as F
from utils.deep_learning_tools import (
    build_train_test_loaders,
    train_model,
    evaluate_model,
)


# --------------------
# MLP Block
# --------------------
class MLPBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (B, T, D)
        x = self.fc1(x)
        x = F.relu(self.ln1(x))
        x = self.fc2(x)
        x = F.relu(self.ln2(x))
        x = self.dropout(x)
        return x


# --------------------
# MLP Classifier
# --------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, n_classes=17, hidden_dim=128, layers=2):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                MLPBlock(input_dim if i == 0 else hidden_dim, hidden_dim)
                for i in range(layers)
            ]
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim, n_classes)

    def forward(self, x):
        # x: (B, T, D)
        x = x.mean(dim=1)  # Pool over time
        for block in self.blocks:
            x = block(x)
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

    model = MLPClassifier(input_dim)
    model = train_model(model, train_loader, epochs=30, lr=1e-3)
    evaluate_model(model, test_loader)
