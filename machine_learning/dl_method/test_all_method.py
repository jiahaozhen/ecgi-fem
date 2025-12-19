import time
from machine_learning.dl_method.BiGRUClassifier import BiGRUClassifier
from machine_learning.dl_method.BiLSTMClassifier import BiLSTMClassifier
from machine_learning.dl_method.CNNBiLSTM import CNNBiLSTM
from machine_learning.dl_method.CNNTransformer import CNNTransformer
from machine_learning.dl_method.CNNClassifier import ImprovedCNN
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
    ('ImprovedCNN', ImprovedCNN),
    ('CNNBiLSTM', CNNBiLSTM),
    ('CNNTransformer', CNNTransformer),
    ('TCNClassifier', TCNClassifier),
    ('TransformerClassifier', TransformerClassifier),
]


def test_all_classifiers():
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

    results = []

    for name, method in methods:
        print(f'\n训练 {name}...')
        start_time = time.time()
        try:
            model = method(input_dim)
            model = train_model(model, train_loader, epochs=30, lr=1e-3)
            elapsed = time.time() - start_time
            print(f'{name}: 训练时间 = {elapsed:.4f}s')
            # 评估模型并记录准确度
            print(f'{name} 测试结果:')
            f1_score = evaluate_model(model, test_loader)
            print('f1_score:', f1_score)
            results.append(
                {'method': name, 'time': elapsed, 'f1_score': f1_score, 'error': None}
            )
        except Exception as e:
            elapsed = time.time() - start_time
            results.append(
                {'method': name, 'time': elapsed, 'f1_score': None, 'error': str(e)}
            )
            print(f'{name}: 错误: {e}')

    print('\n训练完成:')
    for r in results:
        print(r)


if __name__ == '__main__':

    test_all_classifiers()
