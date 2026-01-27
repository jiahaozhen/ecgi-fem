import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.deep_learning_tools import (
    build_train_test_loaders,
    train_model,
    evaluate_model,
)


class Image2DClassifier(nn.Module):
    def __init__(self, n_classes=17):
        super(Image2DClassifier, self).__init__()

        # 类似VGG的简单结构，把 (T, D) 当作单通道图像处理
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 输出 (B, 256, 1, 1)
        )

        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, n_classes))

    def forward(self, x):
        # x shape: (Batch, Time, Dimension)
        # 增加一个 channel 维度 -> (Batch, 1, Time, Dimension)
        x = x.unsqueeze(1)

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # 数据集路径
    data_dir = [
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_processed_dataset/",
    ]

    # 构建 DataLoader
    train_loader, test_loader = build_train_test_loaders(
        data_dir=data_dir, batch_size=32, test_ratio=0.2, num_workers=4
    )

    # 实例化模型
    # 注意：Image2DClassifier 不需要知道 input_dim，因为它使用了 AdaptiveAvgPool2d
    model = Image2DClassifier(n_classes=17)

    # 训练模型
    model = train_model(model, train_loader, epochs=30, lr=1e-3)

    # 评估模型
    evaluate_model(model, test_loader)
