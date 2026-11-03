import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment114", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def read_json(name):
    return json.loads((HERE / name).read_text(encoding="utf8"))


def test_logeuclidean_features_and_blocks():
    rng = np.random.default_rng(114)
    a = rng.normal(size=(6, 5, 30, 30))
    cov = np.einsum("nbci,nbdi->nbcd", a, a) + 1e-3 * np.eye(30)
    train, valid = experiment.logeuclidean_features(cov[:4], cov[4:])
    assert train.shape == (4, 2325) and valid.shape == (2, 2325)
    assert np.isfinite(train).all() and np.isfinite(valid).all()
    for block in experiment.BLOCKS:
        assert experiment.select_block(train, block).shape[1] == (2325 if block == "all" else 465)


def test_saved_oof_metrics_and_aggregation():
    metrics = read_json("metrics.json")
    out = pd.read_csv(HERE / "predictions/DEP02_trial_riemann_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8 and len(pairs) == 32 and out.prediction_count.eq(4).all()
    np.testing.assert_allclose(out.prob_positive, pairs.groupby("trial_index").prob_positive.mean())
    np.testing.assert_array_equal(out.prediction_quota4, experiment.quota_predictions(out.prob_positive))
    assert np.isclose(metrics["pooled_oof"]["auc"], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(metrics["pooled_oof"]["f1"], f1_score(out.y, out.prediction_quota4))


def test_outer_inner_trial_isolation():
    selections = read_json("input_metadata/selections.json")
    outer = list(experiment.trial_pairs(np.arange(8)))
    assert len(selections) == 16 and len(outer) == 16
    for item, (train, valid) in zip(selections, outer, strict=True):
        assert item["train_trial_indices"] == train.tolist() and item["valid_trial_indices"] == valid.tolist()
        assert len(list(experiment.trial_pairs(train))) == 9
        assert len(item["inner_scores"]) == 120


def test_frozen_provenance():
    freeze = read_json("input_metadata/freeze.json")
    provenance = read_json("input_metadata/provenance.json")
    config = read_json("config.json")
    assert freeze["hashes"] == experiment.frozen_hashes()
    assert provenance["freeze"] == freeze and provenance["outer_validation_labels_used_for_selection"] is False
    assert provenance["reference_fit_on_training_trials_only"] is True
    assert config["target_subject"] == "DEP02" and config["positive_quota"] == 4
