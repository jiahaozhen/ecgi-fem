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


# ==================================================
# 1. Flatten only
# ==================================================
class FlattenReducer(ECGReducerBase):
    """
    (B, T, D) -> (B, T*D)
    """

    def transform(self, X: np.ndarray) -> np.ndarray:
        B, T, D = X.shape
        return X.reshape(B, T * D)


# ==================================================
# 2. Flatten + PCA
# ==================================================
class FlattenPCAReducer(ECGReducerBase):
    """
    (B, T, D) -> flatten -> PCA -> (B, out_dim)
    """

    def __init__(self, out_dim: int):
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


# ==================================================
# 3. Factory (extension point)
# ==================================================
class ECGReducerFactory:
    """
    Central place to create reducers
    """

    registry = {
        "flat": FlattenReducer,
        "flat_pca": FlattenPCAReducer,
    }

    @classmethod
    def create(cls, method: str, **kwargs) -> ECGReducerBase:
        if method not in cls.registry:
            raise ValueError(f"Unknown reducer: {method}")
        return cls.registry[method](**kwargs)
