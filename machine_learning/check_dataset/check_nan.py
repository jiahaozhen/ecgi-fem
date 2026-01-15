import numpy as np
import sys
import os

# Add project root to sys.path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from utils.machine_learning_tools import load_dataset


def check_dataset_for_nans(dataset_type, lead_config):
    print(f"\nChecking dataset: Type={dataset_type}, Leads={lead_config}")

    subjects = ["normal_male", "normal_male2"]
    severities = ["severe", "mild", "healthy"]

    data_dir = [
        f"machine_learning/data/Ischemia_Dataset/{subject}/{severity}/d{lead_config}_{dataset_type}_dataset/"
        for subject in subjects
        for severity in severities
    ]

    try:
        X, y, _ = load_dataset(data_dir)

        if X is None or len(X) == 0:
            print("  Dataset is empty or could not be loaded.")
            return

        print(f"  Loaded shape: X={X.shape}, y={y.shape}")

        # Flatten if 3D
        if X.ndim == 3:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X

        # Check for NaNs
        nan_count = np.sum(np.isnan(X_flat))

        if nan_count > 0:
            nan_samples = np.sum(np.isnan(X_flat).any(axis=1))
            print(
                f"  ⚠️  WARNING: Dataset contains {nan_count} NaNs in {nan_samples}/{len(X)} samples!"
            )
        else:
            print("  ✅  Dataset is clean (no NaNs).")

    except Exception as e:
        print(f"  Error loading dataset: {e}")


if __name__ == "__main__":
    configs = [
        ("features", 64),
        ("features", 12),
        ("statistical_features", 64),
        ("statistical_features", 12),
        ("combined_features", 64),
        ("combined_features", 12),
    ]

    for dtype, leads in configs:
        check_dataset_for_nans(dtype, leads)
