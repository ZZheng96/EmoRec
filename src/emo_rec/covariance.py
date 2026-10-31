import numpy as np


def regularized_covariances(segments: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Estimate channel covariance per segment with trace-scaled identity shrinkage."""
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[-1] < 2 or not np.isfinite(segments).all():
        raise ValueError("Expected finite (segments, channels, time) data with at least two time points")
    if not 0 < shrinkage <= 1:
        raise ValueError("shrinkage must be in (0, 1]")
    centered = segments - segments.mean(axis=-1, keepdims=True)
    covariances = centered @ centered.swapaxes(-1, -2) / (segments.shape[-1] - 1)
    scale = np.trace(covariances, axis1=-2, axis2=-1) / segments.shape[1]
    scale = np.maximum(scale, np.finfo(np.float64).tiny)
    return (1 - shrinkage) * covariances + shrinkage * scale[:, None, None] * np.eye(segments.shape[1])


def _matrix_function(matrices: np.ndarray, function) -> np.ndarray:
    symmetric = (matrices + matrices.swapaxes(-1, -2)) / 2
    values, vectors = np.linalg.eigh(symmetric)
    floor = np.maximum(values[..., -1:] * 1e-12, np.finfo(np.float64).tiny)
    values = np.maximum(values, floor)
    return (vectors * function(values)[..., None, :]) @ vectors.swapaxes(-1, -2)


class EuclideanTangentSpace:
    """Log-whitened covariance coordinates at a training-only arithmetic mean."""

    def fit(self, covariances: np.ndarray):
        self.reference_ = np.asarray(covariances, dtype=np.float64).mean(axis=0)
        self.whitener_ = _matrix_function(self.reference_, lambda values: values ** -0.5)
        return self

    def transform(self, covariances: np.ndarray) -> np.ndarray:
        whitened = self.whitener_ @ covariances @ self.whitener_
        logarithms = _matrix_function(whitened, np.log)
        rows, columns = np.triu_indices(logarithms.shape[-1])
        weights = np.where(rows == columns, 1.0, np.sqrt(2.0))
        return logarithms[:, rows, columns] * weights
