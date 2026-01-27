import torch
import torch.nn as nn
import pandas as pd
import os
from utils.ECGDimReducer_tools import ECGReducerFactory
from utils.deep_learning_tools import (
    build_train_test_loaders,
    train_model,
    evaluate_model,
)

# Import Models
from machine_learning.dl_method.BiGRUClassifier import BiGRUClassifier
from machine_learning.dl_method.BiLSTMClassifier import BiLSTMClassifier
from machine_learning.dl_method.CNNBiLSTM import CNNBiLSTM
from machine_learning.dl_method.CNNTransformer import CNNTransformer
from machine_learning.dl_method.CNNClassifier import ResCNNClassifier
from machine_learning.dl_method.TCNClassifier import TCNClassifier
from machine_learning.dl_method.TransformerClassifier import TransformerClassifier

# -------------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------------

# Datasets (Processed as requested)
DATA_DIR_LIST = [
    "machine_learning/data/Ischemia_Dataset/normal_male/mild/d64_processed_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male/severe/d64_processed_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male/healthy/d64_processed_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/mild/d64_processed_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/severe/d64_processed_dataset/",
    "machine_learning/data/Ischemia_Dataset/normal_male2/healthy/d64_processed_dataset/",
]

BATCH_SIZE = 32
TRAIN_EPOCHS = 10  # Moderate epochs for comparison
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = "machine_learning/checkpoints"

# Methods to Test (From test_all_method.py)
MODELS_TO_TEST = [
    ('BiGRU', BiGRUClassifier),
]

# Reducers to Compare
REDUCERS_CONFIG = [
    ("No Reduction", None, {}),
    ("Flatten PCA (n=64)", "flat_pca", {"out_dim": 128}),
    ("Temporal Downsample (step=5)", "temporal_downsample", {"step": 5}),
    ("Temporal Pooling (k=4)", "temporal_pooling", {"kernel_size": 4, "mode": "mean"}),
    (
        "Temporal ST Segment (120-400)",
        "temporal_st_segment",
        {"start_idx": 120, "end_idx": 400},
    ),
]

# -------------------------------------------------------------------------
# ADAPTER
# -------------------------------------------------------------------------


class ModelInputAdapter(nn.Module):
    """
    Wraps any model to ensure input is (B, T, D).
    If input is (B, D) from flatten/stats, unsqueezes to (B, 1, D).
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # x shape: (B, *)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, D)
        return self.model(x)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("DIMENSIONALITY REDUCTION COMPARISON ACROSS METHODS")
    print("=" * 80)
    print(f"Data Sources: {len(DATA_DIR_LIST)} directories")
    print(f"Models: {len(MODELS_TO_TEST)}")
    print(f"Reducers: {len(REDUCERS_CONFIG)}")
    print(f"Device: {DEVICE}")
    print("-" * 80)

    all_results = []

    # 1. Iterate over Models
    for model_name, model_class in MODELS_TO_TEST:
        print(f"\n[Testing Model Architecture: {model_name}]")
        print("~" * 40)

        # 2. Iterate over Reducers
        for reducer_name, reducer_key, reducer_kwargs in REDUCERS_CONFIG:
            print(f"\n   >> Reducer: {reducer_name}")

            # A. Create Reducer
            reducer = None
            if reducer_key is not None:
                try:
                    reducer = ECGReducerFactory.create(reducer_key, **reducer_kwargs)
                except Exception as e:
                    print(f"      [SKIP] Failed to create reducer {reducer_key}: {e}")
                    continue

            # B. Build Loaders (Fit Reducer)
            try:
                # Note: build_train_test_loaders will fit the reducer on the full dataset
                train_loader, test_loader = build_train_test_loaders(
                    DATA_DIR_LIST, batch_size=BATCH_SIZE, reducer=reducer, num_workers=4
                )
            except Exception as e:
                print(f"      [SKIP] Error loading data: {e}")
                continue

            # C. Determine Input Dim
            try:
                X_sample, _, _ = next(iter(train_loader))
                input_shape = X_sample.shape  # (B, T, D) or (B, D)

                if len(input_shape) == 2:
                    input_dim = input_shape[1]  # Features
                elif len(input_shape) == 3:
                    input_dim = input_shape[2]  # Leads/Channels
                else:
                    print(f"      [SKIP] Unexpected shape {input_shape}")
                    continue
            except StopIteration:
                print("      [SKIP] Empty loader")
                continue

            # D. Initialize Model
            try:
                # Instantiate model with input dimension
                base_model = model_class(input_dim=input_dim)

                # Wrap with adapter to handle shape mismatches
                model = ModelInputAdapter(base_model)
            except Exception as e:
                print(f"      [SKIP] Error initializing model: {e}")
                continue

            # E. Train
            print(f"      Training ({input_shape[1:]})...")

            # Prepare Save/Load Path
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)
            # Sanitize names for filename
            safe_model_name = model_name.replace(" ", "_")
            safe_reducer_name = (
                reducer_name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("=", "_")
            )
            save_path = os.path.join(
                CHECKPOINT_DIR, f"{safe_model_name}_{safe_reducer_name}.pth"
            )

            try:
                train_model(
                    model,
                    train_loader,
                    epochs=TRAIN_EPOCHS,
                    device=DEVICE,
                    load_path=save_path,
                    save_path=save_path,
                )
            except Exception as e:
                print(f"      [FAIL] Training failed: {e}")
                continue

            # F. Evaluate
            try:
                metrics = evaluate_model(model, test_loader, device=DEVICE)
            except Exception as e:
                print(f"      [FAIL] Evaluation failed: {e}")
                continue

            # G. Log Results
            res = {
                "Model": model_name,
                "Reducer": reducer_name,
                "Input Shape": str(tuple(input_shape[1:])),
                "Micro F1": metrics.get("Micro F1", 0),
                "Macro F1": metrics.get("Macro F1", 0),
                "Acc": metrics.get("accuracy score", 0),
                "Hamming": metrics.get("Hamming loss", 0),
            }
            all_results.append(res)

            # H. Cleanup
            del model, base_model, reducer, train_loader, test_loader
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    if len(all_results) > 0:
        df = pd.DataFrame(all_results)
        print("\n" + "=" * 100)
        print("FINAL RESULTS TABLE")
        print("=" * 100)
        print(df.to_string(index=False))

        save_path = "dim_reduction_comparison_results.csv"
        df.to_csv(save_path, index=False)
        print(f"\nSaved to {save_path}")
    else:
        print("\nNo results collected.")


if __name__ == "__main__":

    main()
