import os
import h5py

from utils.signal_processing_tools import batch_extract_features


def test_batch_extraction():
    # Path to data
    file_path = 'machine_learning/data/Ischemia_Dataset/normal_male/healthy/d12_noisy_dataset/d_part_000.h5'

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return

    print(f"Loading data from {file_path}")
    with h5py.File(file_path, 'r') as f:
        # Load a small batch for testing
        data_batch = f['X'][:]  # Take first 5 records (5, T, D)

    print(f"Data shape: {data_batch.shape}")

    fs = 1000
    print("Extracting features...")
    features_array, feature_names = batch_extract_features(data_batch, fs=fs)

    print(f"Features shape: {features_array.shape}")
    if feature_names:
        print(f"Feature names ({len(feature_names)}): {feature_names}")

    # Print some sample values
    if features_array.shape[0] > 0 and features_array.shape[2] > 0:
        print("\nSample features for Record 0, Lead 0:")
        for i, name in enumerate(feature_names):
            print(f"  {name}: {features_array[0, 0, i]}")


if __name__ == "__main__":
    test_batch_extraction()
