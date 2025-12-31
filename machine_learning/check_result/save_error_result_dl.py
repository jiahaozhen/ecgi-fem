import torch
import os
from machine_learning.dl_method.BiGRUClassifier import BiGRUClassifier
from machine_learning.dl_method.BiLSTMClassifier import BiLSTMClassifier
from machine_learning.dl_method.CNNBiLSTM import CNNBiLSTM
from machine_learning.dl_method.CNNTransformer import CNNTransformer
from machine_learning.dl_method.CNNClassifier import ImprovedCNN
from machine_learning.dl_method.TCNClassifier import TCNClassifier
from machine_learning.dl_method.TransformerClassifier import TransformerClassifier
from utils.deep_learning_tools import (
    build_train_test_loaders,
    find_wrong_samples,
    save_wrong_samples,
)


if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"

    d_data_dirs = [
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_processed_dataset/",
    ]

    methods = {
        'BiGRUClassifier': BiGRUClassifier,
        'BiLSTMClassifier': BiLSTMClassifier,
        'ImprovedCNN': ImprovedCNN,
        'CNNBiLSTM': CNNBiLSTM,
        'CNNTransformer': CNNTransformer,
        'TCNClassifier': TCNClassifier,
        'TransformerClassifier': TransformerClassifier,
    }

    model_name = "TransformerClassifier"
    model_func = methods[model_name]
    model_path = f"machine_learning/data/model/dl_model/{model_func.__name__}.pth"
    assert os.path.exists(model_path), f"Model not found: {model_path}"

    train_loader, test_loader = build_train_test_loaders(
        d_data_dirs, batch_size=32, test_ratio=0.2, num_workers=4
    )

    X_sample, _, _ = next(iter(train_loader))
    input_dim = X_sample.shape[-1]

    model = model_func(input_dim=input_dim)
    model.to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))

    wrong_samples = find_wrong_samples(
        model,
        test_loader,
        device=device,
    )

    error_samples_dir = "machine_learning/data/error_samples/dl"

    os.makedirs(error_samples_dir, exist_ok=True)

    save_path = os.path.join(error_samples_dir, f"{model_func.__name__}.h5")

    save_wrong_samples(wrong_samples, save_path)
