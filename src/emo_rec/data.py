from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


def load_train_segments(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one v7.3 training subject as 40 non-overlapping 10-second segments."""
    segments_by_emotion = []
    with h5py.File(path, "r") as mat:
        for key in ("EEG_data_neu", "EEG_data_pos"):
            if key not in mat:
                raise ValueError(f"{path} does not contain {key}")
            data = np.asarray(mat[key][()], dtype=np.float64).T
            if data.shape != (30, 50000):
                raise ValueError(f"{path}: {key} has shape {data.shape}, expected (30, 50000)")
            if not np.isfinite(data).all():
                raise ValueError(f"{path}: {key} contains non-finite values")
            segments_by_emotion.append(data.reshape(30, 20, 2500).transpose(1, 0, 2))

    segments = np.concatenate(segments_by_emotion, axis=0)
    labels = np.concatenate([np.zeros(20, dtype=np.int64), np.ones(20, dtype=np.int64)])
    trial_ids = np.concatenate([np.repeat(np.arange(1, 5), 5)] * 2)
    return segments, labels, trial_ids


def load_public_test_segments(path: Path) -> np.ndarray:
    """Load one public-test subject as eight 10-second trials."""
    mat = loadmat(path)
    candidates = [
        np.asarray(value, dtype=np.float64)
        for key, value in mat.items()
        if not key.startswith("__")
        and isinstance(value, np.ndarray)
        and np.issubdtype(value.dtype, np.number)
        and value.shape == (30, 20000)
    ]
    if len(candidates) != 1:
        raise ValueError(f"{path}: expected one 30x20000 EEG array, found {len(candidates)}")
    data = candidates[0]
    if not np.isfinite(data).all():
        raise ValueError(f"{path}: test EEG contains non-finite values")
    return data.reshape(30, 8, 2500).transpose(1, 0, 2)
