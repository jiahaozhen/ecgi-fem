import time
import os
from machine_learning.dl_method.BiGRUClassifier import BiGRUClassifier
from machine_learning.dl_method.BiLSTMClassifier import BiLSTMClassifier
from machine_learning.dl_method.CNNBiLSTM import CNNBiLSTM
from machine_learning.dl_method.CNNTransformer import CNNTransformer
from machine_learning.dl_method.CNNClassifier import ResCNNClassifier
from machine_learning.dl_method.TCNClassifier import TCNClassifier
from machine_learning.dl_method.TransformerClassifier import TransformerClassifier
from utils.deep_learning_tools import (
    build_train_test_loaders,
    train_model,
    evaluate_model,
)


methods = [
    ('BiGRUClassifier', BiGRUClassifier),
    ('BiLSTMClassifier', BiLSTMClassifier),
    ('ResCNNClassifier', ResCNNClassifier),
    ('CNNBiLSTM', CNNBiLSTM),
    ('CNNTransformer', CNNTransformer),
    ('TCNClassifier', TCNClassifier),
    ('TransformerClassifier', TransformerClassifier),
]


def test_all_classifiers():
    # dataset_type = "features"
    dataset_type = "processed"
    # dataset_type = "cnn_ae"
    # dataset_type = "cnn_ae_feature_concat"

    if dataset_type == "features":
        data_dir = [
            "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_features_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_features_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_features_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_features_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_features_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_features_dataset/",
        ]
        model_save_dir = "machine_learning/data/model/features/dl_model"

    elif dataset_type == "processed":
        data_dir = [
            "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_processed_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_processed_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_processed_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_processed_dataset/",
        ]
        model_save_dir = "machine_learning/data/model/processed/dl_model"
    elif dataset_type == "cnn_ae":
        data_dir = [
            "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_cnn_ae_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_cnn_ae_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_cnn_ae_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_cnn_ae_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_cnn_ae_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_cnn_ae_dataset/",
        ]
        model_save_dir = "machine_learning/data/model/cnn_ae/dl_model"
    elif dataset_type == "cnn_ae_feature_concat":
        data_dir = [
            "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_cnn_ae_features_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_cnn_ae_features_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_cnn_ae_features_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_cnn_ae_features_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_cnn_ae_features_dataset/",
            "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_cnn_ae_features_dataset/",
        ]
        model_save_dir = "machine_learning/data/model/all_concat/dl_model"
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # 🔥 使用你之前写好的随机划分函数
    train_loader, test_loader = build_train_test_loaders(
        data_dir=data_dir, batch_size=32, test_ratio=0.2, num_workers=4
    )

    # 自动推断 input_dim（从 train_loader 第一个 batch）
    X_sample, _, _ = next(iter(train_loader))
    input_dim = X_sample.shape[-1]

    results = {}

    os.makedirs(model_save_dir, exist_ok=True)

    for name, method in methods:
        print(f'\n训练 {name}...')
        start_time = time.time()
        try:
            model = method(input_dim)
            save_path = os.path.join(model_save_dir, f"{method.__name__}.pth")
            load_path = save_path if os.path.exists(save_path) else None

            model = train_model(
                model,
                train_loader,
                epochs=30,
                lr=1e-3,
                load_path=load_path,
                save_path=save_path,
            )

            elapsed = time.time() - start_time
            print(f'{name}: 训练时间 = {elapsed:.4f}s')
            # 评估模型并记录准确度
            print(f'{name} 测试结果:')
            metrics = evaluate_model(model, test_loader)
            results[name] = metrics
        except Exception as e:
            elapsed = time.time() - start_time
            print(f'{name}: 错误: {e}')

    print('\n训练完成:')
    for name, metrics in results.items():
        print(f"Method: {name}, Metrics: {metrics}")


if __name__ == '__main__':

    test_all_classifiers()
