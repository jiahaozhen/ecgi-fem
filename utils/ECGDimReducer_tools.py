from typing import Optional

import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from utils.SMPG import smpg


class ECGReducerBase:
    """
    Base class for ECG reducers
    Input shape: (B, T, D)
    Output shape: (B, *)
    """

    def fit(self, X: np.ndarray):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


class FlattenReducer(ECGReducerBase):
    """
    (B, T, D) -> (B, T*D)
    """

    def transform(self, X: np.ndarray) -> np.ndarray:
        B, T, D = X.shape
        return X.reshape(B, T * D)


class FlattenPCAReducer(ECGReducerBase):
    """
    (B, T, D) -> flatten -> PCA -> (B, out_dim)
    """

    def __init__(self, out_dim: int = 256):
        self.out_dim = out_dim
        self.pca = None

    def fit(self, X: np.ndarray):
        X_flat = self._flatten(X)
        self.pca = PCA(n_components=self.out_dim)
        self.pca.fit(X_flat)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("FlattenPCAReducer.transform() called before fit().")
        X_flat = self._flatten(X)
        return self.pca.transform(X_flat)

    @staticmethod
    def _flatten(X: np.ndarray) -> np.ndarray:
        B, T, D = X.shape
        return X.reshape(B, T * D)


class TemporalDownsampleReducer(ECGReducerBase):
    """
    (B, T, D) -> (B, T', D)
    """

    def __init__(self, step: int = 5):
        self.step = step

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError("Expected input shape (B, T, D)")
        return X[:, :: self.step, :]


class TemporalPoolingReducer(ECGReducerBase):
    """
    (B, T, D) -> (B, T', D)
    """

    def __init__(self, kernel_size=4, mode="mean"):
        assert mode in ("mean", "max")
        self.kernel_size = kernel_size
        self.mode = mode

    def transform(self, X: np.ndarray) -> np.ndarray:
        B, T, D = X.shape
        T_new = T // self.kernel_size
        X = X[:, : T_new * self.kernel_size, :]
        X = X.reshape(B, T_new, self.kernel_size, D)

        if self.mode == "mean":
            return X.mean(axis=2)
        else:
            return X.max(axis=2)


class LeadPCAReducer(ECGReducerBase):
    """
    (B, T, D) -> (B, T, D')
    """

    def __init__(self, n_components: int = 8):
        self.n_components = n_components
        self.pca = None

    def fit(self, X: np.ndarray):
        B, T, D = X.shape
        X_2d = X.reshape(B * T, D)
        self.pca = PCA(n_components=self.n_components, random_state=42)
        self.pca.fit(X_2d)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        B, T, D = X.shape
        X_2d = X.reshape(B * T, D)
        if self.pca is None:
            raise RuntimeError("LeadPCAReducer.transform() called before fit().")
        X_pca = self.pca.transform(X_2d)
        return X_pca.reshape(B, T, self.n_components)


class TemporalPCAReducer(ECGReducerBase):
    """
    (B, T, D) -> (B, T', D)
    """

    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.pcas = []

    def fit(self, X: np.ndarray):
        B, T, D = X.shape
        self.pcas = []

        for d in range(D):
            pca = PCA(n_components=self.n_components, random_state=42)
            pca.fit(X[:, :, d])
            self.pcas.append(pca)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        feats = []
        for d, pca in enumerate(self.pcas):
            feat_d = pca.transform(X[:, :, d])
            feats.append(feat_d[..., None])

        return np.concatenate(feats, axis=2)


class TimeLeadPCAReducer(ECGReducerBase):
    """
    PCA on time axis, then PCA on lead axis
    (B, T, D) -> (B, T', D')
    """

    def __init__(self, t_components: int = 64, d_components: int = 8):
        self.t_components = t_components
        self.d_components = d_components
        self.time_pcas = []
        self.lead_pca = None

    def fit(self, X: np.ndarray):
        B, T, D = X.shape

        # 1️⃣ Time PCA (per lead)
        self.time_pcas = []
        X_time_reduced = []

        for d in range(D):
            pca_t = PCA(n_components=self.t_components, random_state=42)
            pca_t.fit(X[:, :, d])
            self.time_pcas.append(pca_t)
            X_td = pca_t.transform(X[:, :, d])  # (B, T')
            X_time_reduced.append(X_td[..., None])

        X_time_reduced = np.concatenate(X_time_reduced, axis=2)  # (B, T', D)

        # 2️⃣ Lead PCA
        X_2d = X_time_reduced.reshape(B * self.t_components, D)
        self.lead_pca = PCA(n_components=self.d_components, random_state=42)
        self.lead_pca.fit(X_2d)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        B, T, D = X.shape

        # Time PCA
        X_time = []
        for d, pca_t in enumerate(self.time_pcas):
            X_td = pca_t.transform(X[:, :, d])
            X_time.append(X_td[..., None])
        X_time = np.concatenate(X_time, axis=2)  # (B, T', D)

        # Lead PCA
        X_2d = X_time.reshape(B * self.t_components, D)
        assert self.lead_pca is not None
        X_ld = self.lead_pca.transform(X_2d)

        return X_ld.reshape(B, self.t_components, self.d_components)


class TemporalSTSegmentReducer(ECGReducerBase):
    """
    Extract ST segment (e.g., samples 200 to 400) from ECG signals.
    (B, T, D) -> (B, T', D) where T' is the length of the ST segment.
    """

    def __init__(self, start_idx: int = 120, end_idx: int = 400):
        self.start_idx = start_idx
        self.end_idx = end_idx

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError("Expected input shape (B, T, D)")
        return X[:, self.start_idx : self.end_idx, :]


class CNNAutoEncoder(nn.Module):
    """
    Time-compression CNN AutoEncoder
    D (channels) is fixed to 64
    Bottleneck: (B, 64, T')
    """

    def __init__(self):
        super().__init__()

        # ---------------- Encoder ----------------
        # Input: (B, 1, D, T)
        self.encoder = nn.Sequential(
            # (B, 1, D, T) -> (B, 16, D, T)
            nn.Conv2d(1, 16, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # T -> T/2
            # (B, 16, D, T/2) -> (B, 32, D, T/2)
            nn.Conv2d(16, 32, kernel_size=(1, 5), padding=(0, 2)),
            nn.ReLU(),
            nn.MaxPool2d((1, 2)),  # T -> T/4
            # (B, 32, D, T/4) -> (B, 64, D, T/4)
            nn.Conv2d(32, 64, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
        )

        # ---------------- Decoder ----------------
        self.decoder = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 2), mode="nearest"),  # T/4 -> T/2
            nn.Conv2d(32, 16, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Upsample(scale_factor=(1, 2), mode="nearest"),  # T/2 -> T
            nn.Conv2d(16, 1, kernel_size=(1, 3), padding=(0, 1)),
        )

    def forward(self, x):
        """
        x: (B, 1, D, T)
        """
        x_enc = self.encoder(x)  # (B, D, D, T')
        z = x_enc.mean(dim=2)  # (B, D, T')  ← 编码结果

        x_rec = self.decoder(x_enc)  # (B, 1, D, T)

        # 解决输入为奇数时，下采样再上采样造成的尺寸不一致问题
        if x_rec.shape[-1] != x.shape[-1]:
            x_rec = nn.functional.interpolate(
                x_rec,
                size=(x.shape[2], x.shape[3]),
                mode="bilinear",
                align_corners=False,
            )

        return x_rec, z


class CNNAutoEncoderReducer:
    def __init__(
        self,
        epochs=20,
        batch_size=32,
        lr=1e-3,
        device=None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    def fit(self, X: np.ndarray):
        """
        X: (B, T, 64)
        """
        B, T, D = X.shape
        assert D == 64, "Input feature dimension must be 64"

        self.model = CNNAutoEncoder().to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        X_tensor = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32).unsqueeze(
            1
        )  # (B, 1, 64, T)

        loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)

                optimizer.zero_grad()
                x_rec, _ = self.model(x_batch)
                loss = criterion(x_rec, x_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x_batch.size(0)

            print(f"Epoch {epoch+1}/{self.epochs}, " f"Loss = {total_loss / B:.6f}")

        self.model.eval()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Return encoded representation
        Output: (B, 64, T')
        """
        if self.model is None:
            raise RuntimeError("Call fit() first.")

        X_tensor = torch.tensor(X.transpose(0, 2, 1), dtype=torch.float32).unsqueeze(1)

        loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=False,
        )

        feats = []
        self.model.eval()
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                _, z = self.model(x_batch)
                feats.append(z.cpu().numpy().transpose(0, 2, 1))  # (B, T', D)

        return np.concatenate(feats, axis=0)


class CNN128AutoEncoder(nn.Module):
    """
    1D CNN AutoEncoder treating ECG as a sequence of vectors.
    Input: (B, D, T) -> Latent: (B, 128)
    Processes temporal dimension with Conv1d, while mixing channel information.
    """

    def __init__(self, in_channels):
        super().__init__()

        # Encoder: (B, D, T) -> (B, 128)
        self.encoder = nn.Sequential(
            # (B, D, T)
            nn.Conv1d(
                in_channels, 32, kernel_size=5, stride=1, padding=2
            ),  # (B, 32, T)
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),  # (B, 32, T/2)
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),  # (B, 64, T/2)
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),  # (B, 64, T/4)
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),  # (B, 128, T/4)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.MaxPool1d(2),  # (B, 128, T/8)
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),  # (B, 256, T/8)
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, 128, 1)
        )

        # Decoder: (B, 128 + 1, T/8) -> (B, D, T)
        # Input channel increased by 1 for Positional Encoding
        self.decoder = nn.Sequential(
            nn.Conv1d(128 + 1, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode="linear", align_corners=False),
            nn.Conv1d(16, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        """
        x: (B, D, T)
        """
        B, D, T = x.shape
        z_enc = self.encoder(x)  # (B, 128, 1)
        z = z_enc.view(B, 128)

        # Expand z
        t_latent = max(1, T // 8)
        z_expanded = z_enc.expand(-1, -1, t_latent)  # (B, 128, T//8)

        # Add Positional Encoding (Linear Ramp from -1 to 1)
        pos = torch.linspace(-1, 1, t_latent, device=x.device, dtype=x.dtype)
        pos = pos.view(1, 1, t_latent).expand(B, -1, -1)  # (B, 1, T//8)

        z_in = torch.cat([z_expanded, pos], dim=1)  # (B, 129, T//8)

        x_rec = self.decoder(z_in)  # (B, D, T//8 * 8)

        if x_rec.shape[-1] != T:
            x_rec = nn.functional.interpolate(
                x_rec, size=T, mode="linear", align_corners=False
            )

        return x_rec, z


class CNN128Reducer:
    def __init__(
        self,
        epochs=20,
        batch_size=32,
        lr=1e-3,
        device=None,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.mean_ = None
        self.std_ = None

    def fit(self, X: np.ndarray):
        """
        X: (B, T, D)
        """
        B, T, D = X.shape

        # Standardization: (X - mean) / std
        # Compute mean/std over B and T for each channel D
        # X shape (B, T, D). Reshape to (B*T, D)
        X_reshaped = X.reshape(-1, D)
        self.mean_ = X_reshaped.mean(axis=0)
        self.std_ = X_reshaped.std(axis=0) + 1e-8

        # Apply normalization
        X_scaled = (X - self.mean_) / self.std_

        # Input to model expects (B, D, T)
        # Transpose X from (B, T, D) to (B, D, T)
        X_tensor = torch.tensor(X_scaled.transpose(0, 2, 1), dtype=torch.float32)

        self.model = CNN128AutoEncoder(in_channels=D).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)

                optimizer.zero_grad()
                x_rec, _ = self.model(x_batch)
                loss = criterion(x_rec, x_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * x_batch.size(0)

            print(f"Epoch {epoch+1}/{self.epochs}, " f"Loss = {total_loss / B:.6f}")

        self.model.eval()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Return encoded representation
        Output: (B, 128)
        """
        if self.model is None:
            raise RuntimeError("Call fit() first.")

        # Apply normalization
        # Handle cases where transform is called on data that might not be shape-compatible if fit was different?
        # But assuming X matches (B, T, D)
        X_scaled = (X - self.mean_) / self.std_

        # (B, T, D) -> (B, D, T)
        X_tensor = torch.tensor(X_scaled.transpose(0, 2, 1), dtype=torch.float32)

        loader = DataLoader(
            TensorDataset(X_tensor),
            batch_size=self.batch_size,
            shuffle=False,
        )

        feats = []
        self.model.eval()
        with torch.no_grad():
            for (x_batch,) in loader:
                x_batch = x_batch.to(self.device)
                _, z = self.model(x_batch)
                feats.append(z.cpu().numpy())  # (B, 128)

        return np.concatenate(feats, axis=0)


class DRPCAReducer(ECGReducerBase):
    """
    Dimensionality reduction using Smoothed Manifold Proximal Gradient (SMPG).
    Wraps the smpg function for sklearn-like usage.
    """

    def __init__(self, n_components=256, gamma=0.5, opts=None):
        """
        Args:
            n_components (int): Target dimensionality.
            gamma (float): Sparsity parameter.
            opts (dict): Dictionary of options for the SMPG solver (e.g., 'tol', 'maxiter', 'mode').
                         Default values will be used if not provided.
        """
        self.n_components = n_components
        self.gamma = gamma
        self.opts = opts if opts is not None else {}
        self.components_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.is_fitted_ = False
        self.training_results_ = None

    def _flatten_input(self, X):
        """Flatten (B, T, D) to (B, T*D) or pass through if already 2D."""
        if X.ndim == 3:
            B, T, D = X.shape
            return X.reshape(B, T * D)
        elif X.ndim == 2:
            return X
        else:
            raise ValueError(
                f"Input data must be 3D (B, T, D) or 2D (B, F). Got ndim={X.ndim}"
            )

    def fit(self, X: np.ndarray):
        """
        Fit the model with X.

        Args:
            X (np.ndarray): Training data with shape (B, T, D) or (B, Features).
        """
        print(f"[DRPCAReducer] Starting fit with input shape: {X.shape}")

        X = X[:, 120:400, :]  # Focus on ST segment
        X_flat = self._flatten_input(X)
        # Center the data
        self.mean_ = np.mean(X_flat, axis=0)
        X_centered = X_flat - self.mean_

        n_samples, n_features = X_centered.shape
        print(f"[DRPCAReducer] Flattened and centered data shape: {X_centered.shape}")

        if n_features < self.n_components:
            raise ValueError(
                f"Number of features ({n_features}) must be greater than n_components ({self.n_components})"
            )

        # Initialize Dictionary for SMPG
        solver_opts = self.opts.copy()
        solver_opts['dim'] = [n_features, self.n_components]
        # PCA Initialization logic (as per requirement)
        if 'x0' not in solver_opts:
            print("[DRPCAReducer] Initializing SMPG with Standard PCA...")
            u_pca, s_pca, vh_pca = np.linalg.svd(X_centered, full_matrices=False)
            x0_pca = vh_pca[: self.n_components, :].T
            solver_opts['x0'] = x0_pca

        # Run SMPG
        print(f"[DRPCAReducer] Running SMPG optimization (gamma={self.gamma})...")
        results = smpg(X_centered, self.gamma, solver_opts)
        self.components_ = results['xopt']  # Shape (n_features, n_components)
        self.training_results_ = results
        self.is_fitted_ = True

        print("[DRPCAReducer] Fitting complete.")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted_:
            raise RuntimeError("SMPGReducer must be fitted before calling transform.")
        # mean_ is guaranteed after fit(); keep assertion for type-checkers
        assert self.mean_ is not None

        X = X[:, 120:400, :]  # Focus on ST segment
        X_flat = self._flatten_input(X)
        # Check feature dimension compatibility
        if X_flat.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.mean_.shape[0]}, got {X_flat.shape[1]}"
            )

        X_centered = X_flat - self.mean_
        # Project data
        output = X_centered @ self.components_
        output = output.reshape(X.shape[0], self.n_components // 64, 64)
        return output


class ECGReducerFactory:
    """
    Central place to create reducers
    """

    registry = {
        "flat": FlattenReducer,
        "flat_pca": FlattenPCAReducer,
        "lead_pca": LeadPCAReducer,
        "temporal_pca": TemporalPCAReducer,
        "temporal_pooling": TemporalPoolingReducer,
        "temporal_downsample": TemporalDownsampleReducer,
        "temporal_lead_pca": TimeLeadPCAReducer,
        "temporal_st_segment": TemporalSTSegmentReducer,
        "cnn_ae": CNNAutoEncoderReducer,
        "cnn128": CNN128Reducer,
        "drpca": DRPCAReducer,
    }

    @classmethod
    def create(cls, method: str, **kwargs) -> ECGReducerBase:
        if method not in cls.registry:
            raise ValueError(f"Unknown reducer: {method}")
        return cls.registry[method](**kwargs)
