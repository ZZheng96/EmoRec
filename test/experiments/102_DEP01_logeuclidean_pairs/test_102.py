import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment102", HERE / "run_experiment.py")
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def test_saved_oof_and_selection():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    out = pd.read_csv(HERE / "predictions/DEP01_logeuclidean_pairs_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    selections = json.loads((HERE / "input_metadata/selections.json").read_text(encoding="utf8"))
    assert len(out) == 8 and len(pairs) == 32 and len(selections) == 16
    assert out.trial_key.is_unique and out.prediction_count.eq(4).all()
    assert pairs.groupby("trial_index").size().eq(4).all()
    assert np.isfinite(out.prob_positive).all() and out.prob_positive.between(0, 1).all()
    assert np.allclose(pairs.groupby("trial_index").prob_positive.mean(), out.prob_positive)
    assert np.isclose(metrics["pooled_oof"]["auc"], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(metrics["pooled_oof"]["f1"], f1_score(out.y, out.prob_positive >= 0.5))
    assert metrics["candidate_count"] == len(exp.CANDIDATES) == 64
    for selected, (train, valid) in zip(selections, exp.pairs(out.y.to_numpy())):
        assert selected["train_indices"] == train.tolist()
        assert selected["valid_indices"] == valid.tolist()
        assert not set(train) & set(valid)
        inner = list(exp.pairs(out.y.to_numpy()[train]))
        assert len(inner) == 9
        assert all(len(tr) == 4 and len(va) == 2 and not set(tr) & set(va) for tr, va in inner)
        assert len(selected["inner_scores"]) == 128
        best = max(selected["inner_scores"], key=lambda r: (r["auc"], r["f1"], -r["candidate_index"], -r["direction_index"]))
        assert selected["selected_candidate"] == best["candidate"]
        assert selected["selected_direction"] == best["direction"]
        assert set(pairs[pairs.pair_id == selected["pair_id"]].trial_index) == set(valid)


def test_training_only_reference_and_scaler(monkeypatch):
    rng = np.random.default_rng(2026)
    a = rng.normal(size=(6, 5, 3, 3))
    cov = a @ a.swapaxes(-1, -2) + np.eye(3)
    train, valid = cov[:4], cov[4:]
    reg, bands = 1e-6, (0, 1)
    features, transformed = exp.logeuclidean_fit_transform(train, valid, reg, bands)
    other_features, _ = exp.logeuclidean_fit_transform(train, valid * 100, reg, bands)
    assert np.array_equal(features, other_features)
    upper = np.triu_indices(3)
    reference = exp.spd_log(train[:, bands].mean(axis=0), reg)
    expected = (exp.spd_log(valid[:, bands], reg) - reference)[..., upper[0], upper[1]].reshape(2, -1)
    assert np.allclose(transformed, expected)
    fitted = []
    original = exp.StandardScaler.fit

    def fit(self, x, y=None, sample_weight=None):
        fitted.append(x.copy())
        return original(self, x, y, sample_weight=sample_weight)

    monkeypatch.setattr(exp.StandardScaler, "fit", fit)
    p = exp.predict(train, np.array([0, 0, 1, 1]), valid, ("logistic_regression", reg, bands, 1))
    q = exp.predict(train, np.array([0, 0, 1, 1]), valid[:1], ("logistic_regression", reg, bands, 1))
    assert all(np.allclose(x, features) for x in fitted)
    assert np.allclose(p[:1], q)


def test_trial_aggregation_and_provenance():
    cov = np.arange(40 * 5 * 3 * 3).reshape(40, 5, 3, 3)
    labels = np.repeat([0, 1], 20)
    trial = np.tile(np.repeat(np.arange(1, 5), 5), 2)
    aggregated, y, keys = exp.trial_covariances(cov, labels, trial)
    assert np.array_equal(aggregated, cov.reshape(8, 5, 5, 3, 3).mean(axis=1))
    assert y.tolist() == [0] * 4 + [1] * 4 and len(set(keys)) == 8
    provenance = json.loads((HERE / "input_metadata/provenance.json").read_text(encoding="utf8"))
    for key, name in [("script_sha256", "run_experiment.py"), ("config_sha256", "config.json"), ("protocol_sha256", "冻结协议.md")]:
        assert provenance[key] == hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    assert provenance["source_sha256"] == hashlib.sha256(Path(provenance["source"]).read_bytes()).hexdigest()
