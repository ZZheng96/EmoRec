import numpy as np

from emo_rec.features import compute_relative_log_bandpower


def test_relative_log_bandpower_matches_log_normalized_power_without_mutating_input() -> None:
    power = np.asarray(
        [
            [[1, 2, 3, 4, 5], [2, 1, 4, 3, 5]],
            [[10, 20, 15, 5, 25], [7, 3, 9, 11, 13]],
        ],
        dtype=float,
    )
    log_power = np.log(power)
    original = log_power.copy()

    relative = compute_relative_log_bandpower(log_power)

    expected = np.log(power / power.sum(axis=-1, keepdims=True))
    np.testing.assert_allclose(relative, expected, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(np.exp(relative).sum(axis=-1), 1.0, rtol=1e-14, atol=1e-14)
    np.testing.assert_array_equal(log_power, original)


def test_relative_log_bandpower_is_stable_for_extreme_log_powers() -> None:
    base = np.arange(5, dtype=float)
    log_power = np.vstack([base + 10000.0, base - 10000.0])

    relative = compute_relative_log_bandpower(log_power)

    assert np.isfinite(relative).all()
    expected = np.log(np.exp(base) / np.exp(base).sum())
    np.testing.assert_allclose(relative, np.vstack([expected, expected]), rtol=1e-12, atol=1e-11)
    np.testing.assert_allclose(np.exp(relative).sum(axis=-1), 1.0, rtol=1e-12, atol=1e-11)


def test_relative_log_bandpower_is_invariant_to_common_channel_power_scale() -> None:
    log_power = np.asarray([[0.5, -1.0, 2.0, 0.0, -2.5]])

    relative = compute_relative_log_bandpower(log_power)
    scaled_relative = compute_relative_log_bandpower(log_power + np.log(123.0))

    np.testing.assert_allclose(relative, scaled_relative, rtol=1e-14, atol=1e-14)
