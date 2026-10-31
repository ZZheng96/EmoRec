import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

CHANNEL_NAMES = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
    "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
    "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
]

FREQUENCY_BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

ASYMMETRY_PAIRS = [
    ("FP1", "FP2"), ("F7", "F8"), ("F3", "F4"), ("FC3", "FC4"),
    ("FT7", "FT8"), ("T3", "T4"), ("C3", "C4"), ("TP7", "TP8"),
    ("CP3", "CP4"), ("T5", "T6"), ("P3", "P4"), ("O1", "O2"),
]


def extract_spectral_features(segments: np.ndarray, sampling_rate: int = 250) -> tuple[np.ndarray, list[str]]:
    """Extract channel log-bandpower and left-minus-right asymmetry features."""
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1] != len(CHANNEL_NAMES):
        raise ValueError("Expected segments with shape (n_segments, 30, n_samples)")
    if not np.isfinite(segments).all():
        raise ValueError("EEG segments contain non-finite values")

    nperseg = min(500, segments.shape[-1])
    frequencies, psd = welch(
        segments,
        fs=sampling_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    band_powers = []
    for low, high in FREQUENCY_BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        band_powers.append(trapezoid(psd[..., mask], frequencies[mask], axis=-1))
    band_powers = np.stack(band_powers, axis=-1)
    log_bandpower = np.log(np.maximum(band_powers, 1e-20))

    channel_indices = {name: index for index, name in enumerate(CHANNEL_NAMES)}
    asymmetry = np.stack(
        [
            log_bandpower[:, channel_indices[left], :] - log_bandpower[:, channel_indices[right], :]
            for left, right in ASYMMETRY_PAIRS
        ],
        axis=1,
    )

    features = np.concatenate(
        [log_bandpower.reshape(len(segments), -1), asymmetry.reshape(len(segments), -1)],
        axis=1,
    )
    names = [
        f"log_bandpower_{channel}_{band}"
        for channel in CHANNEL_NAMES
        for band in FREQUENCY_BANDS
    ]
    names.extend(
        f"asymmetry_{left}_{right}_{band}"
        for left, right in ASYMMETRY_PAIRS
        for band in FREQUENCY_BANDS
    )
    return features, names
