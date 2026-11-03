import json
import importlib.util

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("experiment120", HERE / "run_experiment.py")
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
    np.testing.assert_allclose(EXP.aggregate_windows(x, np.zeros(5), "mean"), [[21.2]])
    assert EXP.AGGREGATIONS == ("mean",)
    assert len(EXP.CANDIDATES) == len(set(EXP.CANDIDATES)) == 36
    assert EXP.CANDIDATES == tuple(("mean", "logistic_regression", b, c, n)
                                   for b in ("all", "spectral", "temporal", "PE")
                                   for n in (1, 2, 3) for c in (0.1, 1, 10))


def test_saved_outputs():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    out = pd.read_csv(HERE / "predictions/DEP02_trial_lowdim_pca_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8 and len(pairs) == 32
    assert out.prediction_count.eq(4).all() and pairs.groupby("trial_index").size().eq(4).all()
    assert np.isfinite(out.prob_positive).all() and out.prediction_quota4.sum() == 4
    assert np.isclose(metrics["pooled_oof"]["auc"], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(metrics["pooled_oof"]["f1"], f1_score(out.y, out.prediction_quota4))
    assert np.isclose(metrics["fixed_0.5"]["f1"], f1_score(out.y, out.prediction_fixed05))
    assert metrics["simultaneously_reached_0.95"] == all(metrics["pooled_oof"][k] >= 0.95 for k in ("auc", "f1"))
    assert metrics["model_candidate_count"] == 36 and metrics["candidate_count"] == 72
    recomputed = [EXP.metric(g.y, g.prob_positive) for _, g in pairs.groupby("pair_id")]
    for k in ("auc", "f1"):
        np.testing.assert_allclose([m[k] for m in metrics["pair_metrics"]], [m[k] for m in recomputed])
        assert np.isclose(metrics["pair_mean"][k], np.mean([m[k] for m in recomputed]))
    np.testing.assert_allclose(out.prob_positive, pairs.groupby("trial_index").prob_positive.mean())
    selections = json.loads((HERE / "input_metadata/selections.json").read_text(encoding="utf8"))
    for selection in selections:
        assert len(selection["inner_scores"]) == 72
        assert not set(selection["train_indices"]) & set(selection["valid_indices"])
        assert selection["valid_indices"] == sorted(selection["valid_indices"])

def test_trial_structure_and_metadata():
    out = pd.read_csv(HERE / "predictions/DEP02_trial_lowdim_pca_oof.csv")
    assert out.y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert all(len(json.loads(rows)) == 5 for rows in out.window_rows)
    provenance = json.loads((HERE / "input_metadata/provenance.json").read_text(encoding="utf8"))
    assert provenance["metadata_rows"] == 40
    assert provenance["feature_dimensions"] == {"all": 720, "spectral": 150, "temporal": 450, "PE": 120}
    for path, digest in provenance["sources"].items():
        assert EXP.sha(path) == digest
    assert all(v == 5 for v in provenance["trial_window_counts"].values())

def test_pca_train_only_and_dimensions(monkeypatch):
    rng = np.random.default_rng(120)
    xt = rng.normal(size=(20, 720))
    xv = rng.normal(size=(10, 720)) + 1000
    train_ids, valid_ids = np.repeat(np.arange(4), 5), np.repeat(np.arange(2), 5)
    original_make = EXP.make_model
    captured = []

    def capture(candidate):
        model = original_make(candidate)
        captured.append(model)
        return model

    monkeypatch.setattr(EXP, "make_model", capture)
    with EXP.threadpool_limits(limits=1):
        for candidate in EXP.CANDIDATES:
            EXP.predict(xt, np.array([0, 0, 1, 1]), train_ids, xv, valid_ids, candidate)
            first = captured[-1]
            EXP.predict(xt, np.array([0, 0, 1, 1]), train_ids, -xv, valid_ids, candidate)
            second = captured[-1]
            train = EXP.aggregate_windows(EXP.block_x(xt, candidate[2]), train_ids, "mean")
            scaler, pca, lr = first.steps[0][1], first.steps[1][1], first.steps[2][1]
            assert scaler.n_samples_seen_ == pca.n_samples_ == 4
            np.testing.assert_allclose(scaler.mean_, train.mean(axis=0))
            np.testing.assert_allclose(pca.mean_, scaler.transform(train).mean(axis=0), atol=1e-14)
            assert pca.n_components_ == candidate[4] <= 3
            assert pca.components_.shape == (candidate[4], train.shape[1])
            assert pca.svd_solver == "full" and pca.whiten is False
            assert lr.solver == "liblinear" and lr.C == candidate[3]
            np.testing.assert_allclose(pca.components_, second.named_steps["pca"].components_)
            np.testing.assert_allclose(scaler.mean_, second.named_steps["standardscaler"].mean_)


def test_real_predictions_and_inner_selection_recomputed():
    config = json.loads((HERE / "config.json").read_text(encoding="utf8"))
    x, ref, _ = EXP.load_subject(config)
    y, ids = ref.y.to_numpy()[::5], np.repeat(np.arange(8), 5)
    saved = json.loads((HERE / "input_metadata/selections.json").read_text(encoding="utf8"))
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    splits = list(EXP.pair_splits(y))
    assert len(splits) == len(saved) == 16
    assert [va.tolist() for _, va in splits] == [[n, p] for p in range(4, 8) for n in range(4)]
    with EXP.threadpool_limits(limits=1):
        for pair_id, ((tr, va), selection) in enumerate(zip(splits, saved)):
            xt, xv = x[EXP.trials_to_rows(tr, ids)], x[EXP.trials_to_rows(va, ids)]
            tid, vid = np.repeat(np.arange(6), 5), np.repeat(np.arange(2), 5)
            assert len(list(EXP.pair_splits(y[tr]))) == 9
            candidate, direction, scores = EXP.select(xt, y[tr], tid)
            assert list(candidate) == selection["selected_candidate"]
            assert direction == selection["selected_direction"]
            assert json.loads(json.dumps(scores)) == selection["inner_scores"]
            assert tr.tolist() == selection["train_indices"] and va.tolist() == selection["valid_indices"]
            prediction = EXP.predict(xt, y[tr], tid, xv, vid, candidate)
            if direction == "reverse":
                prediction = 1 - prediction
            rows = pairs[pairs.pair_id == pair_id]
            np.testing.assert_array_equal(rows.trial_index, va)
            np.testing.assert_array_equal(rows.y, y[va])
            np.testing.assert_allclose(rows.prob_positive, prediction, rtol=1e-12, atol=1e-14)


def test_protocol_candidate_space():
    config = json.loads((HERE / "config.json").read_text(encoding="utf8"))
    assert config["target_subject"] == "DEP02"
    assert config["outer_validation_labels_used_for_selection"] is False
    assert config["positive_quota"] == 4
    assert config["blocks"] == {"all": 720, "spectral": 150, "temporal": 450, "PE": 120}
