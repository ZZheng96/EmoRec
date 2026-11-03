import json
import importlib.util

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("experiment119", HERE / "run_experiment.py")
EXP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXP)


def test_nested_window_label_alignment():
    config = json.loads((HERE / "config.json").read_text(encoding="utf8"))
    x, ref, _ = EXP.load_subject(config)
    keys = ref.trial_key.drop_duplicates().tolist()
    ids = np.repeat(np.arange(8), 5)
    y = ref.y.to_numpy()[::5]
    assert ref.trial.nunique() == 4 and len(keys) == 8
    for tr, va in EXP.pair_splits(y):
        for indices in (tr, va):
            rows = EXP.trials_to_rows(indices, ids)
            np.testing.assert_array_equal(ids[rows], np.repeat(indices, 5))
            np.testing.assert_array_equal(ref.y.to_numpy()[rows], np.repeat(y[indices], 5))
            for aggregation in EXP.AGGREGATIONS:
                actual = EXP.aggregate_windows(x[rows], np.repeat(np.arange(len(indices)), 5), aggregation)
                expected = np.vstack([EXP.aggregate_windows(x[ids == i], np.zeros(5), aggregation) for i in indices])
                np.testing.assert_allclose(actual, expected)
        train_rows = EXP.trials_to_rows(tr, ids)
        inner_ids = np.repeat(np.arange(6), 5)
        for itr, iva in EXP.pair_splits(y[tr]):
            for indices in (itr, iva):
                rows = EXP.trials_to_rows(indices, inner_ids)
                np.testing.assert_array_equal(ids[train_rows][rows], np.repeat(tr[indices], 5))
                np.testing.assert_array_equal(ref.y.to_numpy()[train_rows][rows], np.repeat(y[tr][indices], 5))


def test_select_prediction_assignment(monkeypatch):
    y = np.array([0, 0, 0, 1, 1, 1])
    ids = np.repeat(np.arange(6), 5)
    x = np.column_stack([ids, np.repeat(y, 5)])
    monkeypatch.setattr(EXP, "CANDIDATES", EXP.CANDIDATES[:1])
    seen = []

    def checked_predict(xt, yt, tid, xv, vid, candidate):
        np.testing.assert_array_equal(xt[:, 1], np.repeat(yt, 5))
        features = EXP.aggregate_windows(xv, vid, "mean")
        seen.append(features[:, 0].astype(int))
        return (features[:, 0] + 1) / 10

    monkeypatch.setattr(EXP, "predict", checked_predict)
    _, _, records = EXP.select(x, y, ids)
    for actual, (_, va) in zip(seen, EXP.pair_splits(y)):
        np.testing.assert_array_equal(actual, va)
    expected = EXP.metric(y, (np.arange(6) + 1) / 10)
    assert records[0]["auc"] == expected["auc"]
    assert records[0]["f1"] == expected["f1"]


def test_aggregation_and_candidates():
    x = np.array([0, 1, 2, 3, 100])[:, None]
    for aggregation, expected in (("mean", 21.2), ("median", 2), ("trimmed_mean", 2)):
        np.testing.assert_allclose(EXP.aggregate_windows(x, np.zeros(5), aggregation), [[expected]])
    assert len(EXP.CANDIDATES) == len(set(EXP.CANDIDATES)) == 240


def test_saved_outputs():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    out = pd.read_csv(HERE / "predictions/DEP02_trial_robust_aggregation_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8 and len(pairs) == 32
    assert out.prediction_count.eq(4).all() and pairs.groupby("trial_index").size().eq(4).all()
    assert np.isfinite(out.prob_positive).all() and out.prediction_quota4.sum() == 4
    assert np.isclose(metrics["pooled_oof"]["auc"], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(metrics["pooled_oof"]["f1"], f1_score(out.y, out.prediction_quota4))
    assert np.isclose(metrics["fixed_0.5"]["f1"], f1_score(out.y, out.prediction_fixed05))
    assert metrics["simultaneously_reached_0.95"] == all(metrics["pooled_oof"][k] >= 0.95 for k in ("auc", "f1"))
    assert metrics["model_candidate_count"] == 240 and metrics["candidate_count"] == 480
    recomputed = [EXP.metric(g.y, g.prob_positive) for _, g in pairs.groupby("pair_id")]
    for k in ("auc", "f1"):
        np.testing.assert_allclose([m[k] for m in metrics["pair_metrics"]], [m[k] for m in recomputed])
        assert np.isclose(metrics["pair_mean"][k], np.mean([m[k] for m in recomputed]))
    np.testing.assert_allclose(out.prob_positive, pairs.groupby("trial_index").prob_positive.mean())
    selections = json.loads((HERE / "input_metadata/selections.json").read_text(encoding="utf8"))
    for selection in selections:
        assert len(selection["inner_scores"]) == 480
        assert not set(selection["train_indices"]) & set(selection["valid_indices"])
        assert selection["valid_indices"] == sorted(selection["valid_indices"])

def test_trial_structure_and_metadata():
    out = pd.read_csv(HERE / "predictions/DEP02_trial_robust_aggregation_oof.csv")
    assert out.y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert all(len(json.loads(rows)) == 5 for rows in out.window_rows)
    provenance = json.loads((HERE / "input_metadata/provenance.json").read_text(encoding="utf8"))
    assert provenance["metadata_rows"] == 40
    assert provenance["feature_dimensions"] == {"all": 720, "spectral": 150, "temporal": 450, "PE": 120}
    for path, digest in provenance["sources"].items():
        assert EXP.sha(path) == digest
    assert all(v == 5 for v in provenance["trial_window_counts"].values())

def test_protocol_candidate_space():
    config = json.loads((HERE / "config.json").read_text(encoding="utf8"))
    assert config["target_subject"] == "DEP02"
    assert config["outer_validation_labels_used_for_selection"] is False
    assert config["positive_quota"] == 4
    assert config["blocks"] == {"all": 720, "spectral": 150, "temporal": 450, "PE": 120}
