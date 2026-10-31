import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment037", HERE / "run_experiment.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return e.prepare(CONFIG)


@pytest.mark.parametrize("block", [2, 5])
@pytest.mark.parametrize("seed", [None, 4101, 4102, 4103])
def test_features_subject_isolation_and_row_alignment(block, seed):
    rng = np.random.default_rng(7)
    x = rng.normal(size=(18, 150))
    subjects = np.array(["A", "B"] * 9)
    f = e.features(x, subjects, block, seed)
    assert f.shape == (18, 600) and np.isfinite(f).all()
    np.testing.assert_array_equal(f[:, :150], x)
    altered = x.copy()
    altered[subjects == "B"] += 1000
    np.testing.assert_array_equal(f[subjects == "A"], e.features(altered, subjects, block, seed)[subjects == "A"])
    np.testing.assert_array_equal(f, e.features(x, subjects, block, seed))
    np.testing.assert_allclose(f[:, :150] - f[:, 150:300], f[:, 450:600])


def test_unlabeled_interface_and_explicit_permutation():
    assert list(inspect.signature(e.features).parameters) == ["X", "subjects", "block_size", "permutation_seed", "stream"]
    assert list(inspect.signature(e.base.consistency_features).parameters) == ["X", "subjects", "block_size"]
    x = np.arange(24.).reshape(8, 3)
    subjects = np.array(["A"] * 8)
    order = np.random.default_rng(np.random.SeedSequence([4101, 1])).permutation(np.arange(8))
    actual = e.features(x, subjects, 5, 4101, 1)
    expected = e.base.consistency_features(x[order], subjects[order], 5)
    np.testing.assert_array_equal(actual[order], expected)
    assert not np.array_equal(actual[:, 3:], e.features(x, subjects, 5)[:, 3:])


def test_real_dimensions_normalization_and_rows(data):
    cache, sx, sp, ref = data
    assert sx.shape == (2760, 150) and sp.shape == (80, 150)
    assert ref.subject_id.nunique() == 69 and "HC43" not in ref.subject_id.values
    assert len(np.unique(cache["public_user_ids"])) == 10
    for x, subjects in ((sx, ref.subject_id.to_numpy()), (sp, cache["public_user_ids"])):
        for s in np.unique(subjects):
            np.testing.assert_allclose(x[subjects == s].mean(axis=0), 0, atol=1e-5)
    for c in e.candidates(CONFIG):
        f = e.features(sx, ref.subject_id.to_numpy(), c["train_block"], c["perturbation_seed"])
        assert f.shape == (2760, 150 if c["train_block"] is None else 600)


@pytest.mark.parametrize("seed", CONFIG["random_seeds"])
@pytest.mark.parametrize("candidate", e.candidates(CONFIG), ids=lambda c: c["name"])
def test_all_oof_public_probabilities_and_metadata(data, seed, candidate):
    cache, _, _, ref = data
    name = f"{candidate['name']}_seed{seed}"
    oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
    public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
    fold_public = pd.read_csv(HERE / "predictions" / f"{name}_public_fold_probabilities.csv")
    assert len(oof) == 2760 and len(public) == len(fold_public) == 80
    assert oof.row_id.is_unique and (oof.fold_seed == seed).all()
    for column in ref.columns:
        np.testing.assert_array_equal(oof[column], ref[column])
    folds = e.base.make_subject_folds(ref.subject_id.to_numpy(), ref.health_group.to_numpy(), seed)
    np.testing.assert_array_equal(oof.fold, folds)
    e.base.assert_subject_isolation(oof.subject_id.to_numpy(), folds)
    assert set(folds) == set(range(5))
    assert list(public.columns) == ["user_id", "trial_id", "prob. of positive emotion"]
    for frame in (public, fold_public):
        np.testing.assert_array_equal(frame.user_id, cache["public_user_ids"])
        np.testing.assert_array_equal(frame.trial_id, cache["public_trial_ids"])
    for p in (oof.prob_positive, public["prob. of positive emotion"], fold_public[[f"fold_{i}" for i in range(5)]].to_numpy()):
        assert np.isfinite(p).all() and ((np.asarray(p) >= 0) & (np.asarray(p) <= 1)).all()
    e.base.validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    if candidate["name"] == "consistency_b5_b5":
        original = pd.read_csv(HERE / "predictions" / f"consistency_spectral_seed{seed}_oof.csv")
        np.testing.assert_array_equal(oof.prob_positive, original.prob_positive)


def test_metrics_complete_and_match_saved_predictions(data):
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(HERE / "summary.csv")
    assert metrics["completed_runs"] == len(summary) == 35
    assert len(metrics["runs"]) == 7
    assert not summary.duplicated(["run_name", "fold_seed"]).any()
    for name, run in metrics["runs"].items():
        assert [r["fold_seed"] for r in run["repeats"]] == CONFIG["random_seeds"]
        for repeat in run["repeats"]:
            oof = pd.read_csv(HERE / "predictions" / f"{name}_seed{repeat['fold_seed']}_oof.csv")
            calculated = e.base.summarize(oof.y_true.to_numpy(), oof.prob_positive.to_numpy(), oof, oof.fold.to_numpy())
            assert calculated["overall"]["auc"] == pytest.approx(repeat["result"]["overall"]["auc"])
            assert repeat["result"]["overall"]["threshold"] == .5
            assert len(repeat["result"]["subjects"]) == 69
        for key in e.KEYS:
            values = summary.loc[summary.run_name == name, key]
            assert run["aggregate"][key]["mean"] == pytest.approx(values.mean())
            assert run["aggregate"][key]["std"] == pytest.approx(values.std(ddof=1))
