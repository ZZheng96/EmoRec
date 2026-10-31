import numpy as np

from emo_rec.features import extract_spectral_features


def test_spectral_features_have_expected_shape_and_finite_values() -> None:
    segments = np.zeros((2, 30, 2500))
    features, names = extract_spectral_features(segments)
    assert features.shape == (2, 210)
    assert len(names) == 210
    assert len(set(names)) == 210
    assert np.isfinite(features).all()


def test_spectral_features_reject_wrong_channel_count() -> None:
    try:
        extract_spectral_features(np.zeros((1, 29, 2500)))
    except ValueError as error:
        assert "30" in str(error)
    else:
        raise AssertionError("Expected invalid channel count to raise ValueError")
