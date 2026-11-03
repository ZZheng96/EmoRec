import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment121", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def read(name):
    return json.loads((HERE / name).read_text(encoding="utf8"))


def test_candidate_space_is_frozen_and_ordered():
    assert len(experiment.CANDIDATES) == 300
    assert experiment.CANDIDATES[0] == (0.0, 2, "all", "lda", "auto")
    assert experiment.CANDIDATES[-1] == (0.8, 4, "band_5", "logistic_regression", 10.0)


def test_pairs_and_nested_sizes():
    outer = list(experiment.trial_pairs(np.arange(8)))
    assert len(outer) == 16
    for train, valid in outer:
        assert len(train) == 6 and len(valid) == 2
        assert len(list(experiment.trial_pairs(train))) == 9


def test_shrinkage_formula_and_csp_validation_isolation():
    rng = np.random.default_rng(121)
    z = rng.normal(size=(8, 5, 30, 35))
    cov = z @ z.transpose(0, 1, 3, 2)
    y = np.array([0, 0, 0, 1, 1, 1])
    a, _ = experiment.feature_fit_transform(cov[:6], y, cov[6:], 0.3, 2)
    b, _ = experiment.feature_fit_transform(cov[:6], y, cov[6:] * 100, 0.3, 2)
    np.testing.assert_array_equal(a, b)
    c = cov[:1, 0]
    shrunk = 0.7 * c + 0.3 * np.trace(c, axis1=1, axis2=2)[:, None, None] / 30 * np.eye(30)
    assert np.all(np.linalg.eigvalsh(shrunk[0]) > 0)


def test_saved_predictions_recompute_metrics():
    metrics = read("metrics.json")
    out = pd.read_csv(HERE / "predictions/DEP02_trial_regularized_csp_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8 and len(pairs) == 32
    assert out.prediction_count.eq(4).all()
    np.testing.assert_allclose(out.prob_positive, pairs.groupby("trial_index").prob_positive.mean())
    assert np.isclose(metrics["pooled_oof"]["auc"], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(metrics["pooled_oof"]["f1"], f1_score(out.y, out.prediction_quota4))
    assert np.isclose(metrics["fixed_0.5"]["f1"], f1_score(out.y, out.prediction_fixed05))


def test_selection_records_have_complete_candidates_and_no_outer_labels():
    selections = read("input_metadata/selections.json")
    assert len(selections) == 16
    for item in selections:
        assert item["train_trial_count"] == 6 and item["valid_trial_count"] == 2
        assert item["selection_source"] == "inner OOF trial probabilities only"
        assert len(item["inner_scores"]) == 600
        assert {r["candidate_index"] for r in item["inner_scores"]} == set(range(300))
        assert {r["direction"] for r in item["inner_scores"]} == {"raw", "reverse"}


def test_provenance_freeze_and_protocol():
    config = read("config.json")
    provenance = read("input_metadata/provenance.json")
    freeze = read("input_metadata/freeze.json")
    assert config["candidate_count"] == 300
    assert config["outer_validation_labels_used_for_selection"] is False
    assert provenance["outer_validation_labels_used_for_selection"] is False
    assert provenance["freeze"] == freeze
    assert freeze["hashes"] == {n: __import__("hashlib").sha256((HERE / n).read_bytes()).hexdigest() for n in ("run_experiment.py", "config.json", "冻结协议.md")}
