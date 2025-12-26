import torch
import torch.nn as nn
from utils.deep_learning_tools import (
    build_train_test_loaders,
    train_model,
    evaluate_model,
)


class CNNBiLSTM(nn.Module):
    def __init__(
        self,
        input_channels,  # ECG 通道数（如 1 / 12）
        cnn_channels=64,
        lstm_hidden=128,
        lstm_layers=2,
        num_labels=17,
        dropout=0.3,
    ):
        super().__init__()

        # -------- CNN 特征提取 --------
        self.cnn = nn.Sequential(
            nn.Conv1d(input_channels, cnn_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(dropout),
        )

        # -------- BiLSTM --------
        self.bilstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
            bidirectional=True,
        )

        # -------- 分类头 --------
        self.fc = nn.Linear(2 * lstm_hidden, num_labels)

    def forward(self, x):
        """
        x: [B, T, C]  (batch, time, channel)
        """
        # CNN 需要 [B, C, T]
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.cnn(x)  # [B, C', T']

        # LSTM 需要 [B, T', C']
        x = x.permute(0, 2, 1)  # [B, T', C']

        lstm_out, _ = self.bilstm(x)  # [B, T', 2H]

        # -------- 时间维池化（推荐）--------
        x = torch.mean(lstm_out, dim=1)  # [B, 2H]

        logits = self.fc(x)  # [B, num_labels]
        return logits


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

    model = CNNBiLSTM(input_dim)
    model = train_model(model, train_loader, epochs=30, lr=1e-3)
    evaluate_model(model, test_loader)
