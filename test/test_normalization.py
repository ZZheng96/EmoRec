import numpy as np

from emo_rec.normalization import normalize_by_subject


def test_subject_center_uses_each_subjects_own_feature_means() -> None:
    X = np.asarray([[1, 10], [3, 14], [100, 1000], [104, 1008]], dtype=float)
    subject_ids = np.asarray(["A", "A", "B", "B"])

    normalized = normalize_by_subject(X, subject_ids, "center")

    np.testing.assert_allclose(normalized, [[-1, -2], [1, 2], [-2, -4], [2, 4]])
    np.testing.assert_allclose(X, [[1, 10], [3, 14], [100, 1000], [104, 1008]])


def test_subject_zscore_standardizes_each_subject_and_feature() -> None:
    X = np.asarray([[1, 10], [3, 14], [100, 1000], [104, 1008]], dtype=float)
    subject_ids = np.asarray(["A", "A", "B", "B"])

    normalized = normalize_by_subject(X, subject_ids, "zscore")

    np.testing.assert_allclose(normalized, [[-1, -1], [1, 1], [-1, -1], [1, 1]])


def test_zscore_safely_handles_constant_and_near_zero_variance_features() -> None:
    X = np.asarray([[1.0, 5.0, 2.0], [1.0 + 1e-15, 5.0, 4.0], [8.0, 9.0, 10.0]])
    subject_ids = np.asarray(["A", "A", "B"])

    normalized = normalize_by_subject(X, subject_ids, "zscore", near_zero_std=1e-12)

    assert np.isfinite(normalized).all()
    assert np.abs(normalized[:, 0]).max() < 1e-12
    np.testing.assert_allclose(normalized[:, 1], 0.0)
    np.testing.assert_allclose(normalized[:2, 2], [-1.0, 1.0])
    np.testing.assert_allclose(normalized[2], 0.0)


def test_raw_returns_an_unchanged_copy() -> None:
    X = np.arange(12, dtype=float).reshape(4, 3)
    subject_ids = np.asarray(["A", "A", "B", "B"])

    normalized = normalize_by_subject(X, subject_ids, "raw")

    np.testing.assert_array_equal(normalized, X)
    assert normalized is not X
