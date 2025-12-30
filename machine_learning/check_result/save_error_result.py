import torch
from machine_learning.dl_method.BiGRUClassifier import BiGRUClassifier
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

    parameter_path = "machine_learning/data/model/BiGRUClassifier_last.pth"

    train_loader, test_loader = build_train_test_loaders(
        d_data_dirs, batch_size=32, test_ratio=0.2, num_workers=4
    )

    X_sample, _, _ = next(iter(train_loader))
    input_dim = X_sample.shape[-1]

    model = BiGRUClassifier(
        input_dim=input_dim,
    )

    model.load_state_dict(torch.load(parameter_path, map_location=device))

    wrong_samples = find_wrong_samples(
        model,
        test_loader,
        device=device,
    )

    wrong_save_path = "machine_learning/data/wrong/dl_wrong_samples.h5"

    save_wrong_samples(wrong_samples, wrong_save_path)
