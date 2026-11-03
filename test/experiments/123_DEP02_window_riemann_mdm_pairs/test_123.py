import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment123", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf8"))


def test_spd_shrinkage_and_training_isolation():
    rng = np.random.default_rng(123)
    a = rng.normal(size=(8, 5, 30, 35))
    cov = a @ a.transpose(0, 1, 3, 2)
    y = np.array([0, 0, 0, 1, 1, 1])
    train = experiment.shrink(cov[:6], 0.3)
    np.testing.assert_array_equal(train, experiment.shrink(cov[:6], 0.3))
    assert np.all(np.linalg.eigvalsh(train) > 0)
    assert np.isfinite(experiment.karcher_mean(train[y == 0])).all()
    first = experiment.predict(cov[:6], y, cov[6:], (0.3, "all", "riemann"))
    second = experiment.predict(cov[:6], y, cov[6:] * 100, (0.3, "all", "riemann"))
    np.testing.assert_allclose(first, second, rtol=1e-12, atol=1e-12)


def test_cached_predictions_match_uncached_and_self_distance():
    rng = np.random.default_rng(125)
    a = rng.normal(size=(6, 5, 30, 30)); cov = a @ a.transpose(0, 1, 3, 2)
    y = np.array([0, 0, 0, 1, 1, 1])
    valid = cov[:2]
    candidate = (0.3, "band_2", "riemann")
    cached = experiment.build_prediction_cache(cov, y, valid)
    np.testing.assert_allclose(
        experiment.predict(cov, y, valid, candidate),
        experiment.predict(None, None, None, candidate, cached),
        rtol=1e-12, atol=1e-12,
    )
    prototypes = np.stack([experiment.karcher_mean(experiment.shrink(cov[y == cls], 0.3)) for cls in (0, 1)])
    np.testing.assert_allclose(experiment.distances(prototypes, prototypes[0], "riemann")[0], 0.0, atol=1e-8)


def test_prototype_only_training_and_pair_order():
    rng = np.random.default_rng(124)
    a = rng.normal(size=(8, 5, 30, 30)); cov = a @ a.transpose(0, 1, 3, 2)
    y = np.array([0, 0, 0, 1, 1, 1])
    assert experiment.predict(cov[:6], y, cov[6:], (0, "band_1", "euclidean_logm")).shape == (2,)
    pairs = list(experiment.trial_pairs(np.arange(8)))
    assert len(pairs) == 16 and all(np.all(va[:-1] <= va[1:]) for _, va in pairs)


def test_candidate_count_and_blocks():
    assert len(experiment.CANDIDATES) == 48
    assert experiment.CANDIDATES[0] == (0.0, "all", "riemann")
    assert experiment.CANDIDATES[-1] == (0.5, "band_5", "euclidean_logm")


def test_saved_metrics_and_aggregation():
    metrics = read("metrics.json")
    out = pd.read_csv(HERE / "predictions/DEP02_window_riemann_mdm_pairs_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8 and len(pairs) == 32 and out.prediction_count.eq(4).all()
    np.testing.assert_allclose(out.prob_positive, pairs.groupby("trial_index").prob_positive.mean())
    assert np.isclose(metrics["pooled_oof"]["auc"], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(metrics["pooled_oof"]["f1"], f1_score(out.y, out.prediction_quota4))
    assert np.isclose(metrics["fixed_0.5"]["f1"], f1_score(out.y, out.prediction_fixed05))


def test_real_outer_selections_and_isolation():
    selections = read("input_metadata/selections.json")
    assert len(selections) == 16
    for item in selections:
        assert item["train_window_count"] == 30 and item["valid_window_count"] == 10
        assert len(item["inner_scores"]) == 96
        assert item["selection_source"].startswith("inner OOF")
        assert item["valid_trial_indices"] == sorted(item["valid_trial_indices"])


def test_freeze_provenance_and_config():
    config = read("config.json"); freeze = read("input_metadata/freeze.json"); provenance = read("input_metadata/provenance.json")
    assert config["candidate_count"] == 48 and config["candidate_direction_count"] == 96
    assert provenance["outer_validation_labels_used_for_selection"] is False
    assert provenance["freeze"] == freeze
    assert freeze["hashes"] == {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("run_experiment.py", "config.json", "冻结协议.md")}
