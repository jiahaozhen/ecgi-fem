from utils.deep_learning_tools import build_train_test_loaders


def test_train_test_loaders():
    data_dir = [
        "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
        "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
    ]

    train_loader, test_loader = build_train_test_loaders(
        data_dir=data_dir, batch_size=4, test_ratio=0.2, num_workers=0
    )

    print("Train & test loaders created.")
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches:  {len(test_loader)}")

    # 查看第1个训练 batch
    for X, y in train_loader:
        print("Train X shape:", X.shape)
        print("Train y:", y)
        break

    # 查看第1个测试 batch
    for X, y in test_loader:
        print("Test X shape:", X.shape)
        print("Test y:", y)
        break


if __name__ == "__main__":

    test_train_test_loaders()
