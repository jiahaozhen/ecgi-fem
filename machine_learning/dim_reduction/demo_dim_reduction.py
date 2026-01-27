import numpy as np

from utils.ECGDimReducer_tools import ECGReducerFactory


def run_demo():
    # Simulate a batch of data
    # Shape: (Batch, Time, Leads)
    # User said "64*500". Assuming 64 leads, 500 time steps.
    # Existing models take input_dim=64, so it matches (B, T, 64).
    # Let's create a batch of 32 samples.
    # Note: ECGReducerFactory expects (B, T, D)

    B, T, D = 10000, 500, 64
    print(f"Creating dummy data with shape (Batch={B}, Time={T}, Leads={D})")
    X = np.random.randn(B, T, D).astype(np.float32)

    configs = [
        {
            "type": "temporal_downsample",
            "kwargs": {"step": 5},
            "label": "Temporal Downsampling (step=5)",
        },
        {
            "type": "temporal_pooling",
            "kwargs": {"kernel_size": 4, "mode": "mean"},
            "label": "Temporal Pooling (kernel=4)",
        },
        {
            "type": "flat",
            "kwargs": {},
            "label": "Flatten",
        },
        {
            "type": "lead_pca",
            "kwargs": {"n_components": 32},
            "label": "Lead PCA (n=32)",
        },
        {
            "type": "temporal_pca",
            "kwargs": {"n_components": 8},
            "label": "Temporal PCA (n=8)",
        },
        {
            "type": "cnn_ae",
            "kwargs": {},
            "label": "CNN Autoencoder",
        },
        {
            "type": "drpca",
            "kwargs": {},
            "label": "DRPCA",
        },
    ]

    for cfg in configs:
        reducer_type = cfg["type"]
        kwargs = cfg["kwargs"]
        label = cfg["label"]

        print(f"\n[{label}]")
        try:
            reducer = ECGReducerFactory.create(reducer_type, **kwargs)
            reducer.fit(X)
            X_out = reducer.transform(X)
            print(f"Output shape: {X_out.shape}")
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    run_demo()
