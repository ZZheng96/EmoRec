import numpy as np


def normalize_by_subject(
    X: np.ndarray,
    subject_ids: np.ndarray,
    method: str,
    near_zero_std: float = 1e-12,
) -> np.ndarray:
    """Apply label-free, transductive normalization independently within each subject."""
    X = np.asarray(X, dtype=np.float64)
    subject_ids = np.asarray(subject_ids)
    if X.ndim != 2:
        raise ValueError("X must be a two-dimensional array")
    if subject_ids.ndim != 1 or len(subject_ids) != len(X):
        raise ValueError("subject_ids must contain one ID per row of X")
    if method not in {"raw", "center", "zscore"}:
        raise ValueError(f"Unknown normalization method: {method}")
    if near_zero_std < 0:
        raise ValueError("near_zero_std must be non-negative")

    normalized = X.copy()
    if method == "raw":
        return normalized

    for subject_id in np.unique(subject_ids):
        mask = subject_ids == subject_id
        subject_X = X[mask]
        normalized[mask] = subject_X - subject_X.mean(axis=0)
        if method == "zscore":
            std = subject_X.std(axis=0)
            safe_std = np.where(std > near_zero_std, std, 1.0)
            normalized[mask] /= safe_std
    return normalized
