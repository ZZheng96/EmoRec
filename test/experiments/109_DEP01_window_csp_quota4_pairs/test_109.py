import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment109", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def read_json(name):
    return json.loads((HERE / name).read_text(encoding="utf8"))


def test_quota_has_no_labels_and_stable_ties():
    p = np.array([0.1, 0.8, 0.2, 0.9, 0.3, 0.7, 0.4, 0.6])
    assert experiment.quota_predictions(p).tolist() == [0, 1, 0, 1, 0, 1, 0, 1]
    assert experiment.quota_predictions(np.full(8, 0.5)).tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    assert experiment.quota_predictions(np.zeros(8)).sum() == 4
    assert experiment.quota_predictions(np.ones(8)).sum() == 4


def test_saved_oof_metrics_and_aggregation():
    metrics = read_json("metrics.json")
    out = pd.read_csv(HERE / "predictions/DEP01_window_csp_pairs_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8 and len(pairs) == 32
    assert out.trial_index.tolist() == list(range(8))
    assert out.prediction_count.eq(4).all()
    assert pairs.groupby("trial_index").size().eq(4).all()
    assert pairs.window_count.eq(5).all()
    np.testing.assert_allclose(out.prob_positive, pairs.groupby("trial_index").prob_positive.mean())
    np.testing.assert_array_equal(out.prediction_quota4, experiment.quota_predictions(out.prob_positive))
    np.testing.assert_array_equal(out.prediction_fixed05, out.prob_positive >= 0.5)
    assert out.prediction_quota4.sum() == 4
    assert np.isclose(metrics["pooled_oof"]["auc"], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(metrics["pooled_oof"]["f1"], f1_score(out.y, out.prediction_quota4))
    assert np.isclose(metrics["fixed_0.5_comparison"]["f1"], f1_score(out.y, out.prediction_fixed05))
    assert metrics["positive_quota"] == 4
    assert metrics["quota_search"] is False and metrics["threshold_search"] is False


def test_outer_inner_trial_window_isolation():
    trial = np.repeat(np.arange(8), 5)
    outer = list(experiment.trial_pairs(np.arange(8)))
    assert len(outer) == 16
    selections = read_json("input_metadata/selections.json")
    for item, (train, valid) in zip(selections, outer, strict=True):
        assert item["train_trial_indices"] == train.tolist()
        assert item["valid_trial_indices"] == valid.tolist()
        tr = experiment.window_indices(trial, train)
        va = experiment.window_indices(trial, valid)
        assert len(tr) == item["train_window_count"] == 30
        assert len(va) == item["valid_window_count"] == 10
        assert not set(tr) & set(va)
        inner = list(experiment.trial_pairs(train))
        assert len(inner) == 9
        for inner_train, inner_valid in inner:
            itr = experiment.window_indices(trial, train[inner_train])
            iva = experiment.window_indices(trial, train[inner_valid])
            assert len(itr) == 20 and len(iva) == 10
            assert not set(itr) & set(iva)
            assert set(itr) | set(iva) <= set(tr)
        scores = item["inner_scores"]
        assert len(scores) == 240
        assert {(r["candidate_index"], r["direction"]) for r in scores} == {(i, d) for i in range(120) for d in ("raw", "reverse")}
        best = max(scores, key=lambda r: (r["auc"], r["f1"], -r["candidate_index"], -r["direction_index"]))
        assert item["selected_candidate"] == best["candidate"]
        assert item["selected_direction"] == best["direction"]


def test_csp_training_features_do_not_depend_on_validation():
    rng = np.random.default_rng(109)
    z = rng.normal(size=(8, 5, 30, 35))
    cov = z @ z.transpose(0, 1, 3, 2)
    labels = np.array([0, 0, 0, 1, 1, 1])
    first, _ = experiment.csp_fit_transform(cov[:6], labels, cov[6:], 1e-4, 2)
    second, _ = experiment.csp_fit_transform(cov[:6], labels, cov[6:] * 100, 1e-4, 2)
    np.testing.assert_array_equal(first, second)


def test_frozen_provenance():
    freeze = read_json("input_metadata/freeze.json")
    provenance = read_json("input_metadata/provenance.json")
    config = read_json("config.json")
    assert freeze["hashes"] == experiment.frozen_hashes()
    assert provenance["freeze"] == freeze
    assert provenance["independent_refit"] is True
    assert freeze["started_at_utc"] < provenance["completed_at_utc"]
    assert config["positive_quota"] == 4
    assert config["outer_validation_labels_used_for_selection"] is False
