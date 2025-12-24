import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA


class ECGDimReducer:
    """
    ECG dimensionality reduction tool for multi-lead ECG signals.

    Input shape
    -----------
    (B, T, D)
        B : batch size / number of samples
        T : time length
        D : number of ECG leads

    Features
    --------
    - Time dimension reduction (pooling / downsampling)
    - Lead dimension reduction (PCA)
    - Optional flattening for classical ML models
    - NumPy / PyTorch automatic conversion
    """

    def __init__(
        self,
        time_method: str | None = "pool",
        time_factor: int = 4,
        lead_method: str | None = None,
        lead_dim: int | None = None,
        flatten: bool = False,
    ):
        """
        Parameters
        ----------
        time_method : {'pool', 'downsample', None}
            Method for time dimension reduction.
        time_factor : int
            Reduction factor for time dimension.
        lead_method : {'pca', None}
            Method for lead dimension reduction.
        lead_dim : int
            Target lead dimension after reduction.
        flatten : bool
            Whether to reshape output to (B, T*D).
        """
        self.time_method = time_method
        self.time_factor = time_factor
        self.lead_method = lead_method
        self.lead_dim = lead_dim
        self.flatten = flatten

        self.pca = None

    # ======================================================
    # Time dimension reduction (torch)
    # ======================================================
    def _reduce_time(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: torch.Tensor, shape (B, T, D)
        """
        if self.time_method is None:
            return x

        # (B, T, D) -> (B, D, T)
        x = x.permute(0, 2, 1)

        if self.time_method == "pool":
            x = F.avg_pool1d(x, kernel_size=self.time_factor)
        elif self.time_method == "downsample":
            x = x[:, :, :: self.time_factor]
        else:
            raise ValueError(f"Unknown time_method: {self.time_method}")

        # (B, D, T') -> (B, T', D)
        return x.permute(0, 2, 1)

    # ======================================================
    # Lead dimension reduction (numpy / PCA)
    # ======================================================
    def fit_lead(self, X: np.ndarray):
        """
        Fit PCA on lead dimension.

        X: np.ndarray, shape (B, T, D)
        """
        if self.lead_method != "pca":
            return

        B, T, D = X.shape
        X2d = X.reshape(B * T, D)

        self.pca = PCA(n_components=self.lead_dim)
        self.pca.fit(X2d)

    def _reduce_lead(self, X: np.ndarray) -> np.ndarray:
        """
        X: np.ndarray, shape (B, T, D)
        """
        if self.lead_method is None:
            return X

        if self.lead_method == "pca":
            if self.pca is None:
                raise RuntimeError("PCA is not fitted.")

            B, T, D = X.shape
            X2d = X.reshape(B * T, D)
            Xp = self.pca.transform(X2d)
            return Xp.reshape(B, T, -1)

        raise ValueError(f"Unknown lead_method: {self.lead_method}")

    # ======================================================
    # Public API
    # ======================================================
    def fit(self, X):
        """
        Fit reducer (only affects lead PCA).

        X: np.ndarray or torch.Tensor, shape (B, T, D)
        """
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()

        self.fit_lead(X)
        return self

    def transform(self, X):
        """
        Transform ECG data.

        X: np.ndarray or torch.Tensor, shape (B, T, D)

        Returns
        -------
        np.ndarray or torch.Tensor
            Reduced ECG data.
        """
        is_numpy = isinstance(X, np.ndarray)

        # -------- 1. Time reduction --------
        if self.time_method is not None:
            if is_numpy:
                X = torch.from_numpy(X).float()
                is_numpy = False

            X = self._reduce_time(X)

        # -------- 2. Lead reduction --------
        if self.lead_method is not None:
            if not is_numpy:
                X = X.detach().cpu().numpy()
                is_numpy = True

            X = self._reduce_lead(X)

        # -------- 3. Optional flatten --------
        if self.flatten:
            B = X.shape[0]
            X = X.reshape(B, -1)

        return X

    def fit_transform(self, X):
        """
        Fit + transform.
        """
        self.fit(X)
        return self.transform(X)
