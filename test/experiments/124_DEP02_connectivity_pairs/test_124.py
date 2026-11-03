import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

import run_experiment as r

HERE = Path(__file__).resolve().parent


def test_connectivity_symmetry_and_diagonal():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2, 30, 2500))
    f = r.window_features(x)
    assert f.shape == (2, 2325)
    for i in range(5):
        tri = f[:, i * 435:(i + 1) * 435]
        assert np.isfinite(tri).all()
    z = np.asarray([[1, 2, 3], [2, 1, 4], [3, 4, 1]], float)
    assert np.allclose(z, z.T) and np.allclose(np.diag(z), 1)


def test_feature_dimensions_and_candidates():
    assert len(r.CANDIDATES) == 48
    assert len(r.CANDIDATES) * 2 == 96
    assert r.block_x(np.zeros((1, 2325)), "connectivity_all").shape[1] == 2175
    assert r.block_x(np.zeros((1, 2325)), "bandpower").shape[1] == 150


def test_trial_pair_window_alignment():
    keys = np.array(["0::1"] * 5 + ["1::1"] * 5 + ["0::2"] * 5 + ["1::2"] * 5)
    assert all((keys == k).sum() == 5 for k in np.unique(keys))
    assert list(r.pair_splits(np.array([0, 0, 1, 1])))[0][1].tolist() == [0, 2]


def test_aggregation_and_standardization_fit_shape():
    x = np.arange(20, dtype=float).reshape(10, 2)
    ids = np.repeat(np.arange(2), 5)
    assert r.aggregate(x, ids, "mean").shape == (2, 2)
    assert r.aggregate(x, ids, "median").shape == (2, 2)
    model = r.make_model(("mean", "bandpower", "logistic_regression", 1))
    assert "standardscaler" in str(model).lower()


def test_saved_metrics_recompute():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    out = pd.read_csv(HERE / "predictions/DEP02_trial_connectivity_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8 and len(pairs) == 32
    assert out.prediction_count.eq(4).all() and out.prediction_quota4.sum() == 4
    assert np.isclose(metrics["pooled_oof"]["auc"], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(metrics["pooled_oof"]["f1"], f1_score(out.y, out.prediction_quota4))


def test_provenance_and_protocol():
    cfg = json.loads((HERE / "config.json").read_text(encoding="utf8"))
    prov = json.loads((HERE / "input_metadata/provenance.json").read_text(encoding="utf8"))
    assert cfg["candidate_count"] == 48 and cfg["candidate_direction_count"] == 96
    assert prov["metadata_rows"] == 40
    assert all(v == 5 for v in prov["trial_window_counts"].values())
    assert prov["outer_validation_labels_used_for_selection"] is False
