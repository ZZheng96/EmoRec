import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("micro045", HERE / "run_experiment.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def inputs():
    rng = np.random.default_rng(45)
    w = np.stack([m.normalize_window(x) for x in rng.normal(size=(12, 30, 100))])
    subjects = np.repeat(["A", "B", "C"], 4)
    config = json.loads((HERE / "config.json").read_text())
    return w, subjects, subjects != "C", config


def fit(w, s, train, config):
    with threadpool_limits(limits=1):
        return m.fit_templates(w, s, train, config)


def test_templates_training_only(inputs):
    w, s, train, c = inputs
    a, provenance = fit(w, s, train, c)
    changed = w.copy(); changed[~train] *= 100
    b, other = fit(changed, s, train, c)
    np.testing.assert_array_equal(a, b)
    assert provenance == other
    assert set(provenance["train_subjects"]) == {"A", "B"}
    assert set(provenance["train_window_indices"]) == set(np.flatnonzero(train))
    assert provenance["n_sampled_maps"] == train.sum()*32


def test_validation_labels_do_not_affect_features(inputs):
    w, s, train, c = inputs
    y = np.arange(len(w)) % 2
    a, _ = fit(w, s, train, c)
    before = m.sequence_features(w, a)
    y[~train] = 1-y[~train]
    b, _ = fit(w, s, train, c)
    np.testing.assert_array_equal(before, m.sequence_features(w, b))
    assert "y" not in m.fit_templates.__code__.co_varnames
    assert "y" not in m.sequence_features.__code__.co_varnames


def test_subject_disjoint():
    s = np.repeat([f"S{i}" for i in range(20)], 2)
    g = np.repeat(["HC"]*10 + ["DEP"]*10, 2)
    for seed in range(2026, 2032):
        f = m.base.make_subject_folds(s, g, seed)
        m.base.assert_subject_isolation(s, f)
        assert set(f) == set(range(5))


def test_finite_dimension_and_coverage(inputs):
    w, s, train, c = inputs
    t, _ = fit(w, s, train, c)
    x = m.sequence_features(np.concatenate([w, np.zeros_like(w[:1])]), t)
    assert x.shape == (13, 34) and np.isfinite(x).all()
    np.testing.assert_allclose(x[:, :4].sum(axis=1), 1)


def test_window_independence_and_permutation(inputs):
    w, s, train, c = inputs
    t, _ = fit(w, s, train, c)
    a = m.sequence_features(w, t)
    np.testing.assert_array_equal(a[::-1], m.sequence_features(w[::-1], t))
    changed = w.copy(); changed[0] = 0
    np.testing.assert_array_equal(a[1:], m.sequence_features(changed, t)[1:])
    np.testing.assert_array_equal(a[0], m.sequence_features(w[:1], t)[0])
    raw = np.random.default_rng(1).normal(size=(2, 30, 2500))
    first = m.normalize_window(raw[0]); raw[1] *= 50
    np.testing.assert_array_equal(first, m.normalize_window(raw[0]))


def test_oof_coverage():
    m.check_oof(np.full(20, .5), np.ones(20))
    for p, coverage in [(np.full(20, np.nan), np.ones(20)), (np.full(20, .5), np.zeros(20)), (np.full(20, .5), np.full(20, 2))]:
        with pytest.raises(AssertionError):
            m.check_oof(p, coverage)


def test_saved_artifacts_when_available():
    if not (HERE / "metrics.json").exists():
        return
    metrics = json.loads((HERE / "metrics.json").read_text())
    assert set(metrics["repeat_status"].values()) == {"completed"}
    files = list((HERE / "predictions").glob("*_oof.csv"))
    assert len(files) == 18
    for path in files:
        frame = pd.read_csv(path)
        assert len(frame) == 2760 and frame.window_id.nunique() == 2760
        assert frame.subject_id.nunique() == 69 and "HC43" not in set(frame.subject_id)
        m.base.assert_subject_isolation(frame.subject_id.to_numpy(), frame.fold.to_numpy())
        m.check_oof(frame.prob_positive.to_numpy(), np.ones(len(frame)))
    for path in (HERE / "predictions").glob("*_public.csv"):
        m.validate_submission(pd.read_csv(path), [f"P_test{i}" for i in range(1, 11)])
    for path in (HERE / "input_metadata").glob("*_fold*_templates.json"):
        provenance = json.loads(path.read_text())
        run, fold = path.stem.rsplit("_fold", 1)
        fold = int(fold.split("_")[0])
        assignment = pd.read_csv(HERE / f"input_metadata/{run}_folds.csv")
        assert set(provenance["train_subjects"]) == set(assignment.loc[assignment.fold != fold, "subject_id"])
        assert set(provenance["train_window_indices"]) == set(assignment.index[assignment.fold != fold])


def test_public_format():
    frame = pd.DataFrame({"user_id": np.repeat([f"P_test{i}" for i in range(1, 11)], 8), "trial_id": np.tile(np.arange(1, 9), 10), "prob. of positive emotion": np.full(80, .5)})
    m.validate_submission(frame, [f"P_test{i}" for i in range(1, 11)])
    assert len(frame) == 80
    with pytest.raises(ValueError):
        m.validate_submission(frame.iloc[:-1], [f"P_test{i}" for i in range(1, 11)])
