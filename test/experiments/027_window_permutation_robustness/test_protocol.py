import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment027", HERE / "run_experiment.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def test_determinism_and_five_groups_of_eight():
    subjects = np.repeat(["a", "b"], 40)
    for seed in CONFIG["permutation_seeds"]:
        batches = protocol.random_batches(subjects, seed)
        np.testing.assert_array_equal(batches, protocol.random_batches(subjects, seed))
        for s in np.unique(subjects):
            np.testing.assert_array_equal(np.bincount(batches[subjects == s]), [8] * 5)
    assert not np.array_equal(protocol.random_batches(subjects, 1101), protocol.random_batches(subjects, 2202))
    with pytest.raises(ValueError):
        protocol.random_batches(subjects[:-1], 1101)


def test_no_label_or_trial_balancing():
    assert list(inspect.signature(protocol.random_batches).parameters) == ["subjects", "seed"]
    subjects = np.repeat(["a", "b"], 40)
    labels = np.tile(np.repeat([0, 1], 20), 2)
    batches = protocol.random_batches(subjects, 1101)
    counts = [labels[(subjects == s) & (batches == b)].sum() for s in ("a", "b") for b in range(5)]
    assert any(c != 4 for c in counts)
    # Grouping does not access labels or trial metadata; arbitrary changes cannot affect it.
    labels[:] = 1 - labels
    np.testing.assert_array_equal(batches, protocol.random_batches(subjects, 1101))


def test_subject_and_group_isolation():
    subjects = np.repeat(["a", "b"], 40)
    batches = protocol.random_batches(subjects, 1101)
    np.testing.assert_array_equal(batches[:40], protocol.random_batches(subjects[:40], 1101))
    np.testing.assert_array_equal(batches[40:], protocol.random_batches(subjects[40:], 1101))
    X = np.random.default_rng(5).normal(size=(80, 4))
    normalized = protocol.normalize(X, subjects, batches, 8)
    mask = (subjects == "a") & (batches == 0)
    changed = X.copy()
    changed[~mask] += 1000
    np.testing.assert_array_equal(normalized[mask], protocol.normalize(changed, subjects, batches, 8)[mask])
    np.testing.assert_allclose(normalized[mask], (X[mask] - X[mask].mean(0)) / X[mask].std(0))


def test_fixed_folds_and_config():
    previous = json.loads((HERE.parent / "026_window_matched_hgb/config.json").read_text(encoding="utf-8"))
    for key in ("hgb", "feature_cache", "reference_oof", "excluded_subject", "outer_folds", "random_seed"):
        assert CONFIG[key] == previous[key]
    assert len(CONFIG["permutation_seeds"]) == len(set(CONFIG["permutation_seeds"])) == 10
    with np.load((HERE / CONFIG["feature_cache"]).resolve()) as cache:
        keep = cache["subject_ids"] != "HC43"
        data = {k: cache[k][keep] for k in ("subject_ids", "health_groups", "original_trial_ids", "y_train")}
    ref = pd.read_csv(HERE / CONFIG["reference_oof"])
    folds = protocol.align_reference(data, ref)
    assert len(folds) == 2760 and set(folds) == set(range(5))
    for s in np.unique(data["subject_ids"]):
        assert len(set(folds[data["subject_ids"] == s])) == 1
    with pytest.raises(ValueError):
        protocol.align_reference(data, ref.iloc[::-1])


def test_saved_outputs_and_recomputed_metrics():
    if not (HERE / "metrics.json").exists():
        pytest.skip("Run full experiment first")
    results = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    ref = pd.read_csv(HERE / CONFIG["reference_oof"])
    names = ["control_40_40", "baseline_8_8"] + [f"{c}_seed_{s}" for s in CONFIG["permutation_seeds"] for c in ("matched_random_8_8", "eval_only_40_8")]
    assert set(results["runs"]) == set(names)
    for name in names:
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        pd.testing.assert_frame_equal(oof[ref.columns.drop("prob_positive")], ref.drop(columns="prob_positive"))
        assert len(oof) == 2760 and oof.subject_id.nunique() == 69 and "HC43" not in set(oof.subject_id)
        assert oof.prob_positive.between(0, 1).all()
        assert results["runs"][name]["overall"] == pytest.approx(protocol.metrics(oof.y_true, oof.prob_positive))
        if "_seed_" in name:
            seed = int(name.rsplit("_", 1)[1])
            np.testing.assert_array_equal(oof.normalization_batch, protocol.random_batches(oof.subject_id.to_numpy(), seed))
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        protocol.validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
        assert len(public) == 80 and not public.duplicated(["user_id", "trial_id"]).any()
    for name in ("control_40_40", "baseline_8_8"):
        assert results["runs"][name]["reference_026_max_abs_probability_difference"] < 1e-8
    summary = pd.read_csv(HERE / "summary.csv")
    assert len(summary) == 22
    for condition, stats in results["seed_summary"].items():
        subset = summary[summary.condition == condition]
        assert len(subset) == 10
        for metric, values in stats.items():
            assert values["mean"] == pytest.approx(subset[metric].mean(), abs=1e-14)
            assert values["std"] == pytest.approx(subset[metric].std(ddof=1), abs=1e-14)
