import importlib.util
import warnings
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parent / "experiments" / "004_labram_frozen" / "run_experiment.py"
SPEC = importlib.util.spec_from_file_location("experiment_004_labram_frozen", MODULE_PATH)
labram = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(labram)


def test_competition_channel_order_uses_official_indices_without_reordering() -> None:
    expected = [
        0, 1, 3, 16, 18, 20, 22, 24, 27, 29, 31, 33, 35, 89, 40, 42,
        44, 91, 49, 51, 53, 55, 57, 90, 62, 64, 66, 92, 81, 82, 83,
    ]
    assert labram.get_input_chans(labram.COMPETITION_CHANNELS) == expected
    assert labram.get_input_chans([name.lower() for name in labram.COMPETITION_CHANNELS]) == expected


def test_official_channel_list_recognizes_legacy_temporal_names() -> None:
    official_names = labram.load_official_standard_1020()
    for legacy_name in ("T3", "T4", "T5", "T6"):
        assert legacy_name in official_names


def test_processed_train_trial_is_split_into_five_ordered_segments() -> None:
    processed = np.arange(30 * 10000, dtype=np.float64).reshape(30, 10000)

    segments = labram.split_train_trial(processed)

    assert segments.shape == (5, 30, 2000)
    for index in range(5):
        np.testing.assert_array_equal(segments[index], processed[:, index * 2000 : (index + 1) * 2000])


def test_short_public_trial_preprocessing_is_finite_without_filter_length_warning() -> None:
    rng = np.random.default_rng(2026)
    time = np.arange(2500) / 250.0
    data = 5.0 * np.sin(2 * np.pi * 10.0 * time)[None, :]
    data = data + 0.1 * rng.standard_normal((30, 2500))

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        processed = labram.preprocess_continuous_trial(data)

    assert processed.shape == (30, 2000)
    assert np.isfinite(processed).all()


def test_model_input_uses_microvolt_division_by_100_and_ten_one_second_patches() -> None:
    segments = np.arange(2 * 30 * 2000, dtype=np.float32).reshape(2, 30, 2000)

    samples = labram.prepare_model_input(segments)

    assert tuple(samples.shape) == (2, 30, 10, 200)
    np.testing.assert_allclose(samples.numpy(), segments.reshape(2, 30, 10, 200) / 100.0)


def test_classification_metrics_include_binary_and_macro_f1_at_threshold() -> None:
    y_true = np.asarray([0, 0, 1, 1])
    probabilities = np.asarray([0.1, 0.8, 0.7, 0.9])

    metrics = labram.classification_metrics(y_true, probabilities, threshold=0.5)

    assert metrics["roc_auc"] == 0.75
    assert metrics["binary_f1"] == 0.8
    assert np.isclose(metrics["macro_f1"], 11.0 / 15.0)
    assert (metrics["tn"], metrics["fp"], metrics["fn"], metrics["tp"]) == (1, 1, 0, 2)
