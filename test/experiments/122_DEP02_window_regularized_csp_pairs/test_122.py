import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment122", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf8"))


def test_candidate_count_and_order():
    assert len(experiment.CANDIDATES) == 320
    assert experiment.CANDIDATES[0] == (0.0, 2, "all", "linear_svc", 0.01)
    assert experiment.CANDIDATES[-1] == (0.5, 4, "band_4", "logistic_regression", 10)


def test_shrinkage_is_training_only_and_formula():
    rng = np.random.default_rng(122)
    z = rng.normal(size=(8, 5, 30, 35))
    cov = z @ z.transpose(0, 1, 3, 2)
    y = np.array([0, 0, 0, 1, 1, 1])
    first, _ = experiment.csp_fit_transform(cov[:6], y, cov[6:], 0.3, 2)
    second, _ = experiment.csp_fit_transform(cov[:6], y, cov[6:] * 100, 0.3, 2)
    np.testing.assert_array_equal(first, second)
    c = cov[:1, 0]
    shrunk = 0.7 * c + 0.3 * np.trace(c, axis1=1, axis2=2)[:, None, None] / 30 * np.eye(30)
    assert np.all(np.linalg.eigvalsh(shrunk[0]) > 0)


def test_csp_and_scaler_validation_isolation():
    rng = np.random.default_rng(123)
    cov = rng.normal(size=(8, 5, 30, 30))
    cov = cov @ cov.transpose(0, 1, 3, 2)
    y = np.array([0, 0, 0, 1, 1, 1])
    a, _ = experiment.csp_fit_transform(cov[:6], y, cov[6:], 0.1, 4)
    b, _ = experiment.csp_fit_transform(cov[:6], y, cov[6:] * 50, 0.1, 4)
    np.testing.assert_array_equal(a, b)


def test_window_trial_pair_alignment():
    trial = np.repeat(np.arange(8), 5)
    outer = list(experiment.trial_pairs(np.arange(8)))
    assert len(outer) == 16
    for train, valid in outer:
        tr = experiment.window_indices(trial, train)
        va = experiment.window_indices(trial, valid)
        assert len(tr) == 30 and len(va) == 10 and not set(tr) & set(va)
        assert set(trial[tr]) | set(trial[va]) == set(range(8))
        assert len(list(experiment.trial_pairs(train))) == 9


def test_saved_predictions_and_true_metric_recompute():
    metrics = read("metrics.json")
    out = pd.read_csv(HERE / "predictions/DEP02_window_regularized_csp_pairs_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8 and len(pairs) == 32
    assert out.trial_index.tolist() == list(range(8)) and out.prediction_count.eq(4).all()
    np.testing.assert_allclose(out.prob_positive, pairs.groupby("trial_index").prob_positive.mean())
    assert np.isclose(metrics["pooled_oof"]["auc"], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(metrics["pooled_oof"]["f1"], f1_score(out.y, out.prediction_quota4))
    assert np.isclose(metrics["fixed_0.5"]["f1"], f1_score(out.y, out.prediction_fixed05))


def test_selections_have_complete_inner_candidates_and_no_outer_labels():
    selections = read("input_metadata/selections.json")
    assert len(selections) == 16
    for item in selections:
        assert item["train_window_count"] == 30 and item["valid_window_count"] == 10
        assert item["selection_source"] == "inner OOF trial probabilities only"
        assert len(item["inner_scores"]) == 640
        assert {r["candidate_index"] for r in item["inner_scores"]} == set(range(320))
        assert {r["direction"] for r in item["inner_scores"]} == {"raw", "reverse"}


def test_freeze_provenance_and_config():
    config = read("config.json")
    freeze = read("input_metadata/freeze.json")
    provenance = read("input_metadata/provenance.json")
    assert config["candidate_count"] == 320 and config["candidate_direction_count"] == 640
    assert config["outer_validation_labels_used_for_selection"] is False
    assert provenance["outer_validation_labels_used_for_selection"] is False
    assert provenance["freeze"] == freeze
    assert freeze["hashes"] == {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("run_experiment.py", "config.json", "冻结协议.md")}
