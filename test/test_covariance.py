import numpy as np
import pytest

from emo_rec.covariance import EuclideanTangentSpace, regularized_covariances
from emo_rec.normalization import normalize_by_subject


def test_covariance_formula_and_positive_definiteness():
    segments = np.random.default_rng(2026).normal(size=(6, 4, 100))
    segments[:, 3] = segments[:, 2]
    original = segments.copy()
    result = regularized_covariances(segments)
    sample = np.cov(segments[0])
    np.testing.assert_allclose(result[0], 0.9 * sample + 0.1 * np.trace(sample) / 4 * np.eye(4))
    assert np.all(np.linalg.eigvalsh(result) > 0)
    np.testing.assert_array_equal(segments, original)
    np.testing.assert_allclose(result, regularized_covariances(segments + 10), atol=1e-12)


def test_training_reference_and_tangent_geometry():
    matrices = np.asarray([np.diag([1., 4.]), np.diag([3., 2.])])
    space = EuclideanTangentSpace().fit(matrices)
    reference = space.reference_.copy()
    np.testing.assert_allclose(reference, np.diag([2., 3.]))
    np.testing.assert_allclose(space.transform(reference[None]), 0, atol=1e-12)
    values = space.transform(matrices)
    np.testing.assert_allclose(values[0], [np.log(0.5), 0, np.log(4 / 3)], atol=1e-12)
    space.transform(matrices * 100)
    np.testing.assert_array_equal(space.reference_, reference)
    covariance = np.asarray([[[np.cosh(1), np.sinh(1)], [np.sinh(1), np.cosh(1)]]])
    identity_space = EuclideanTangentSpace().fit(np.eye(2)[None])
    np.testing.assert_allclose(identity_space.transform(covariance), [[0, np.sqrt(2), 0]], atol=1e-12)


def test_degenerate_signals_are_finite():
    covariance = regularized_covariances(np.zeros((2, 3, 20)))
    transformed = EuclideanTangentSpace().fit(covariance).transform(covariance)
    assert transformed.shape == (2, 6)
    assert np.isfinite(transformed).all()


def test_subject_normalization_is_independent():
    X = np.random.default_rng(1).normal(size=(12, 6))
    ids = np.repeat(["train", "validation", "public"], 4)
    for method in ("center", "zscore"):
        result = normalize_by_subject(X, ids, method)
        changed = X.copy()
        changed[4:] *= 100
        np.testing.assert_allclose(normalize_by_subject(changed, ids, method)[:4], result[:4])
        np.testing.assert_allclose(result[:4].mean(axis=0), 0, atol=1e-12)


@pytest.mark.parametrize("shrinkage", [0, -0.1, 1.1])
def test_invalid_shrinkage(shrinkage):
    with pytest.raises(ValueError):
        regularized_covariances(np.ones((2, 3, 10)), shrinkage)
