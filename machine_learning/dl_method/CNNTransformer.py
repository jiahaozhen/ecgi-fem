import torch
import torch.nn as nn
import math
from utils.deep_learning_tools import (
    build_train_test_loaders,
    train_model,
    evaluate_model,
)


# -------- 位置编码 --------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, T, D]

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        return x + self.pe[:, : x.size(1)]


# -------- 主模型 --------
class CNNTransformer(nn.Module):
    def __init__(
        self,
        input_channels,  # ECG 通道数
        cnn_channels=64,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=256,
        num_labels=17,
        dropout=0.3,
    ):
        super().__init__()

        # -------- CNN 特征提取 --------
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, d_model, kernel_size=5, padding=2),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

        # -------- 位置编码 --------
        self.pos_encoder = PositionalEncoding(d_model)

        # -------- Transformer Encoder --------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # -------- 分类头 --------
        self.fc = nn.Linear(d_model, num_labels)

    def forward(self, x):
        """
        x: [B, T, C]
        """
        # CNN: [B, C, T]
        x = x.permute(0, 2, 1)
        x = self.cnn(x)  # [B, D, T']

        # Transformer: [B, T', D]
        x = x.permute(0, 2, 1)
        x = self.pos_encoder(x)

        x = self.transformer(x)  # [B, T', D]

        # -------- 时间维池化 --------
        x = torch.mean(x, dim=1)  # [B, D]

        logits = self.fc(x)  # [B, num_labels]

        return logits


# -------------------
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

    model = CNNTransformer(input_dim)
    model = train_model(model, train_loader, epochs=30, lr=1e-3)
    evaluate_model(model, test_loader)
