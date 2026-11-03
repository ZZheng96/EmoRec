import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("e143", ROOT / "src/143_HC24_csp_transfer_pairs/run_experiment.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def test_algorithm_unchanged_from_ast_reference():
    old = ast.parse((ROOT / "src/test/143_HC24_csp_transfer_pairs_ast_reference.py").read_text(encoding="utf8"))
    new = ast.parse(e.SOURCE.read_text(encoding="utf8"))
    names = {"sigmoid", "quota_predictions", "metric", "trial_pairs", "window_indices", "covariances", "csp_fit_transform", "predict", "select"}
    a = {n.name: ast.dump(n) for n in old.body if isinstance(n, ast.FunctionDef) and n.name in names}
    b = {n.name: ast.dump(n) for n in new.body if isinstance(n, ast.FunctionDef) and n.name in names}
    assert a == b


def test_cache_metadata_matches_raw():
    eeg, y, trial, source = e.load_hc24()
    x, ref = e.load_720()
    assert source.name == "HC24.mat" and eeg.shape == (40, 30, 2500)
    assert x.shape == (40, 720) and np.isfinite(x).all()
    np.testing.assert_array_equal(ref.y, y)
    np.testing.assert_array_equal(ref.trial - 1 + 4 * ref.y, trial)
    assert ref.subject.eq("HC24").all()


def test_nested_isolation_and_candidate_count():
    assert len(e.CANDIDATES) == len(set(e.CANDIDATES)) == 120
    selections = json.loads((HERE / "input_metadata/selections.json").read_text())
    pairs = list(e.trial_pairs(np.arange(8)))
    assert len(selections) == len(pairs) == 16
    coverage = np.zeros(8, dtype=int)
    for item, (tr, va) in zip(selections, pairs):
        assert item["train_trial_indices"] == tr.tolist()
        assert item["valid_trial_indices"] == va.tolist()
        assert len(tr) == 6 and len(va) == 2 and set(tr).isdisjoint(va)
        assert item["train_window_count"] == 30 and item["valid_window_count"] == 10
        coverage[va] += 1
        inner = list(e.trial_pairs(tr))
        assert len(inner) == 9
        for ti, vi in inner:
            assert len(ti) == 4 and len(vi) == 2
            assert set(tr[ti]).isdisjoint(tr[vi])
            assert set(tr[ti]).union(tr[vi]).isdisjoint(va)
        scores = item["inner_scores"]
        assert len(scores) == 240
        assert {(s["candidate_index"], s["direction"]) for s in scores} == {(i, d) for i in range(120) for d in ("raw", "reverse")}
        best = max(scores, key=lambda r: (r["auc"], r["f1"], -r["candidate_index"], -r["direction_index"]))
        assert item["selected_candidate"] == best["candidate"]
        assert item["selected_direction"] == best["direction"]
    assert coverage.tolist() == [4] * 8


def test_real_predictions_and_metrics_recompute():
    eeg, y, trial, _ = e.load_hc24()
    cov = e.covariances(eeg)
    selections = json.loads((HERE / "input_metadata/selections.json").read_text())
    pair = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    out = pd.read_csv(HERE / "predictions/HC24_window_csp_pairs_oof.csv")
    metrics = json.loads((HERE / "metrics.json").read_text())
    assert len(pair) == 32 and len(out) == 8
    with threadpool_limits(limits=1):
        for item in selections:
            tr = e.window_indices(trial, item["train_trial_indices"])
            va = e.window_indices(trial, item["valid_trial_indices"])
            p = e.predict(cov[tr], y[tr], cov[va], item["selected_candidate"])
            if item["selected_direction"] == "reverse":
                p = 1 - p
            rows = pair[pair.pair_id == item["pair_id"]].sort_values("trial_index")
            expected = [p[trial[va] == t].mean() for t in rows.trial_index]
            np.testing.assert_allclose(rows.prob_positive, expected, atol=1e-12)
    grouped = pair.groupby("trial_index").prob_positive
    np.testing.assert_allclose(out.prob_positive, grouped.mean(), atol=1e-12)
    np.testing.assert_array_equal(out.prediction_count, grouped.count())
    np.testing.assert_array_equal(out.prediction_fixed05, out.prob_positive >= .5)
    np.testing.assert_array_equal(out.prediction_quota4, e.quota_predictions(out.prob_positive))
    m = metrics["csp_window_pairs"]
    assert np.isclose(m["auc"], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(m["f1_fixed05"], f1_score(out.y, out.prediction_fixed05))
    assert np.isclose(m["f1_quota4"], f1_score(out.y, out.prediction_quota4))
    base = pd.read_csv(HERE / "predictions/baseline_720_linearsvc_oof.csv")
    assert len(base) == 40 and base.fold.nunique() == 8
    np.testing.assert_array_equal(base.fold, trial)
    np.testing.assert_array_equal(base.y, y)
    bm = metrics["baseline_720_linearsvc_fixed05"]
    assert np.isclose(bm["auc"], roc_auc_score(base.y, base.prob_positive))
    assert np.isclose(bm["f1"], f1_score(base.y, base.prob_positive >= .5))


def test_frozen_hashes_and_quota():
    frozen = json.loads((HERE / "input_metadata/freeze.json").read_text())
    assert frozen["hashes"] == e.frozen_hashes()
    assert e.SOURCE.relative_to(ROOT).as_posix() in frozen["hashes"]
    assert e.quota_predictions(np.full(8, .5)).tolist() == [1, 1, 1, 1, 0, 0, 0, 0]
    cfg = json.loads((HERE / "config.json").read_text())
    assert cfg["subject"] == "HC24" and cfg["candidate_count"] == 120
    assert cfg["threshold"] == .5 and cfg["positive_quota"] == 4
    assert cfg["threshold_search"] is False and cfg["quota_search"] is False
    assert cfg["outer_validation_labels_used_for_selection"] is False

