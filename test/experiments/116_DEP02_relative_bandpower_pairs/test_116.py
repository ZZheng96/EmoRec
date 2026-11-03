import importlib.util
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment116", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)

def read_json(name): return json.loads((HERE / name).read_text(encoding="utf8"))

def test_feature_blocks_and_signal_only():
    rng = np.random.default_rng(116)
    x = rng.normal(size=(2, 30, 2500))
    f = experiment.window_features(x)
    assert f.shape == (2, 210) and np.isfinite(f).all()
    assert experiment.select_block(f, "relative").shape[1] == 150
    np.testing.assert_allclose(f[:, :150].reshape(2, 5, 30).mean(axis=2), 0, atol=1e-10)
    assert experiment.select_block(f, "asymmetry").shape[1] == 60

def test_saved_oof_metrics_and_aggregation():
    metrics = read_json("metrics.json")
    out = pd.read_csv(HERE / "predictions/DEP02_trial_relative_bandpower_oof.csv")
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
        assert len(item["inner_scores"]) == 60

def test_frozen_provenance():
    freeze = read_json("input_metadata/freeze.json")
    provenance = read_json("input_metadata/provenance.json")
    config = read_json("config.json")
    assert freeze["hashes"] == experiment.frozen_hashes()
    assert provenance["freeze"] == freeze and provenance["outer_validation_labels_used_for_selection"] is False
    assert config["target_subject"] == "DEP02" and config["positive_quota"] == 4
