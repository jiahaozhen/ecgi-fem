import torch.nn as nn
from utils.deep_learning_tools import (
    build_train_test_loaders,
    train_model,
    evaluate_model,
)


# --------------------
# Transformer Block
# --------------------
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Linear(dim * 4, dim),
        )
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, D)
        attn_out, _ = self.attn(x, x, x)
        x = self.ln1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.ln2(x + self.dropout(ff_out))
        return x


# --------------------
# Transformer Classifier
# --------------------
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, n_classes=17, hidden=128, layers=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden)
        self.blocks = nn.ModuleList([TransformerBlock(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden, n_classes)

    def forward(self, x):
        # x: (B, T, D)
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        x = x.mean(dim=1)  # Global Average Pooling over time
        x = self.dropout(x)
        return self.fc(x)


# --------------------
# Main
# --------------------
if __name__ == "__main__":
    data_dir = [
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
    ]

    # 🔥 使用你之前写好的随机划分函数
    train_loader, test_loader = build_train_test_loaders(
        data_dir=data_dir, batch_size=32, test_ratio=0.2, num_workers=4
    )

    # 自动推断 input_dim（从 train_loader 第一个 batch）
    X_sample, _ = next(iter(train_loader))
    input_dim = X_sample.shape[-1]
    model = TransformerClassifier(input_dim)
    model = train_model(model, train_loader, epochs=30, lr=1e-3)
    evaluate_model(model, test_loader)
