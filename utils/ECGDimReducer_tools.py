import numpy as np
from sklearn.decomposition import PCA


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


class StatisticalReducer(ECGReducerBase):
    """
    Extract statistics along time axis
    (B, T, D) -> (B, D * n_stats)
    """

    def __init__(self, stats=("mean", "std", "min", "max")):
        self.stats = stats

    def transform(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 3:
            raise ValueError("Input must be (B, T, D)")

        feats = []

        if "mean" in self.stats:
            feats.append(X.mean(axis=1))
        if "std" in self.stats:
            feats.append(X.std(axis=1))
        if "min" in self.stats:
            feats.append(X.min(axis=1))
        if "max" in self.stats:
            feats.append(X.max(axis=1))
        if "ptp" in self.stats:  # peak-to-peak
            feats.append(X.ptp(axis=1))

        return np.concatenate(feats, axis=1)


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

    def __init__(self, n_components: int = 32):
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

    def __init__(self, t_components: int = 128, d_components: int = 32):
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
    }

    @classmethod
    def create(cls, method: str, **kwargs) -> ECGReducerBase:
        if method not in cls.registry:
            raise ValueError(f"Unknown reducer: {method}")
        return cls.registry[method](**kwargs)
