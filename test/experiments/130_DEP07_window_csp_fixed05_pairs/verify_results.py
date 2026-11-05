import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import run_experiment as run

HERE = Path(__file__).resolve().parent


def independent_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    positive, negative = p[y == 1], p[y == 0]
    auc = np.mean((positive[:, None] > negative).astype(float) + 0.5 * (positive[:, None] == negative))
    pred = p >= 0.5
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    return {"auc": float(auc), "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def main():
    template = HERE.parent / "129_DEP06_window_csp_fixed05_pairs"
    assert (HERE / "run_experiment.py").read_text(encoding="utf8") == (template / "run_experiment.py").read_text(encoding="utf8").replace("DEP06", "DEP07").replace("[129/16]", "[130/16]")
    expected = json.loads((template / "config.json").read_text(encoding="utf8"))
    expected.update(experiment_id=HERE.name, target_subject="DEP07")
    assert json.loads((HERE / "config.json").read_text(encoding="utf8")) == expected
    selections = json.loads((HERE / "input_metadata/selections.json").read_text(encoding="utf8"))
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    pooled = pd.read_csv(HERE / "predictions/DEP07_window_csp_pairs_oof.csv")
    x, y, trial, source = run.load_subject("DEP07")
    cov = run.covariances(x)
    assert len(selections) == 16 and len(pairs) == 32 and len(pooled) == 8
    assert {tuple(s["valid_trial_indices"]) for s in selections} == set(product(range(4, 8), range(4)))
    windows, splits = [], []
    with threadpool_limits(limits=1):
        for s in selections:
            pid = s["pair_id"]
            print(f"verify pair {pid + 1}/16", flush=True)
            train_trials = set(s["train_trial_indices"])
            valid_trials = set(s["valid_trial_indices"])
            assert train_trials.isdisjoint(valid_trials) and train_trials | valid_trials == set(range(8))
            tr = np.flatnonzero(np.isin(trial, list(train_trials)))
            va = np.flatnonzero(np.isin(trial, list(valid_trials)))
            assert len(tr) == s["train_window_count"] == 30 and len(va) == s["valid_window_count"] == 10
            inner_splits = []
            for pos, neg in product(sorted(train_trials & set(range(4, 8))), sorted(train_trials & set(range(4)))):
                iv = np.flatnonzero(np.isin(trial, [pos, neg]))
                it = np.setdiff1d(tr, iv)
                assert len(it) == 20 and len(iv) == 10
                assert not set(it) & set(iv) and not (set(it) | set(iv)) & set(va)
                inner_splits.append({"train_windows": it.tolist(), "valid_windows": iv.tolist()})
            assert len(inner_splits) == 9
            best = max(s["inner_scores"], key=lambda r: (r["auc"], r["f1"], -r["candidate_index"], -r["direction_index"]))
            assert len(s["inner_scores"]) == 240
            assert best["candidate"] == s["selected_candidate"] and best["direction"] == s["selected_direction"]
            p = run.predict(cov[tr], y[tr], cov[va], s["selected_candidate"])
            if s["selected_direction"] == "reverse":
                p = 1 - p
            for index, probability in zip(va, p):
                windows.append({"pair_id": pid, "window_index": int(index), "trial_index": int(trial[index]), "window_in_trial": int(index % 5), "y": int(y[index]), "prob_positive": float(probability)})
            splits.append({"pair_id": pid, "train_windows": tr.tolist(), "valid_windows": va.tolist(), "inner_splits": inner_splits})
    windows = pd.DataFrame(windows)
    assert len(windows) == 160 and windows.window_index.nunique() == 40
    assert windows.groupby("window_index").size().eq(4).all()
    reconstructed = windows.groupby(["pair_id", "trial_index"]).prob_positive.mean()
    actual = pairs.set_index(["pair_id", "trial_index"]).prob_positive.sort_index()
    assert np.allclose(reconstructed, actual, atol=1e-12, rtol=0)
    assert windows.groupby(["pair_id", "trial_index"]).size().eq(5).all()
    reconstructed_pooled = pairs.groupby("trial_index").prob_positive.mean()
    assert np.allclose(reconstructed_pooled, pooled.prob_positive, atol=1e-12, rtol=0)
    assert pairs.groupby("trial_index").size().eq(4).all()
    checked = independent_metrics(pooled.y, pooled.prob_positive)
    for key in checked:
        assert np.isclose(checked[key], metrics["pooled_oof"][key])
    for pid, group in pairs.groupby("pair_id"):
        recalculated = independent_metrics(group.y, group.prob_positive)
        assert all(np.isclose(value, metrics["pair_metrics"][pid][key]) for key, value in recalculated.items())
    diagnostics = {}
    for name, frame in (("window", windows), ("pair", pairs), ("pooled", pooled)):
        p = frame.prob_positive.to_numpy()
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        diagnostics[name] = {"min": float(p.min()), "max": float(p.max()), "std": float(p.std()), "unique_count": int(len(np.unique(p))), "near_half_count_at_1e-6": int(np.sum(abs(p - 0.5) <= 1e-6)), "saturated_count_at_1e-6": int(np.sum((p <= 1e-6) | (p >= 1 - 1e-6))), "positive_predictions": int(np.sum(p >= 0.5)), "constant": bool(np.ptp(p) <= 1e-12)}
    result = {"independent_pooled": checked, "template_only_allowed_substitutions": True, "outer_pairs": 16, "inner_pairs_per_outer": 9, "unique_windows": 40, "window_predictions": 160, "pair_predictions": 32, "pooled_predictions": 8, "all_isolation_checks_passed": True, "inner_only_selection_check_passed": True, "aggregation_checks_passed": True, "probability_diagnostics": diagnostics}
    windows.to_csv(HERE / "predictions/window_predictions.csv", index=False)
    (HERE / "input_metadata/split_audit.json").write_text(json.dumps(splits, indent=2), encoding="utf8")
    (HERE / "input_metadata/verification.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
