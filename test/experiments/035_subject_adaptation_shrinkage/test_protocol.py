import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("shrinkage_035", HERE / "run_experiment.py")
run = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run)


def test_unlabeled_fold_isolation_and_subject_independence():
    rng = np.random.default_rng(42)
    train = rng.normal(size=(80, 4))
    valid = rng.normal(size=(20, 4))
    subjects = np.repeat(["a", "b"], 10)
    population = run.fit_population(train)
    assert list(inspect.signature(run.transform).parameters) == ["X", "subjects", "population", "shrinkage"]
    assert list(inspect.signature(run.fit_population).parameters) == ["X"]
    before = run.transform(valid, subjects, population, 0.5)
    changed = valid.copy()
    changed[10:] += 1000
    after = run.transform(changed, subjects, population, 0.5)
    np.testing.assert_array_equal(before[:10], after[:10])
    np.testing.assert_array_equal(population[0], train.mean(axis=0))
    np.testing.assert_array_equal(population[1], train.std(axis=0))
    own = valid[:10]
    expected = (own - (own.mean(0) + train.mean(0)) / 2) / ((own.std(0) + train.std(0)) / 2 + 1e-12)
    np.testing.assert_allclose(before[:10], expected)


@pytest.mark.parametrize("shrinkage", [0, 0.25, 0.5, 0.75, 1])
def test_boundaries_and_determinism(shrinkage):
    rng = np.random.default_rng(2026)
    X = rng.normal(size=(60, 5))
    X[:, 0] = 3
    subjects = np.repeat(["a", "b", "c"], 20)
    population = run.fit_population(X[:40])
    transformed = run.transform(X, subjects, population, shrinkage)
    assert np.isfinite(transformed).all()
    np.testing.assert_array_equal(transformed, run.transform(X, subjects, population, shrinkage))
    if shrinkage == 1:
        np.testing.assert_allclose(transformed, (X - population[0]) / (population[1] + 1e-12))
    if shrinkage == 0:
        for subject in np.unique(subjects):
            own = X[subjects == subject]
            np.testing.assert_allclose(transformed[subjects == subject], (own - own.mean(0)) / (own.std(0) + 1e-12))


def test_model_determinism_probability_range():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(42)
    X, y = rng.normal(size=(100, 8)), np.tile([0, 1], 50)
    with threadpool_limits(limits=2):
        a = HistGradientBoostingClassifier(**config["model"], random_state=2026).fit(X, y).predict_proba(X)[:, 1]
        b = HistGradientBoostingClassifier(**config["model"], random_state=2026).fit(X, y).predict_proba(X)[:, 1]
    np.testing.assert_array_equal(a, b)
    assert np.isfinite(a).all() and ((a >= 0) & (a <= 1)).all()


def test_real_protocol_outputs_and_fold_population():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    data, X, public, folds = run.load_data(config)
    assert "HC43" not in data["subject_ids"]
    for fold in range(5):
        train, valid = folds != fold, folds == fold
        assert not set(data["subject_ids"][train]) & set(data["subject_ids"][valid])
        population = run.fit_population(X[train])
        changed = X.copy()
        changed[valid] += 10000
        for a, b in zip(population, run.fit_population(changed[train])):
            np.testing.assert_array_equal(a, b)
    reference = pd.read_csv(HERE.parent / "002_subject_normalization" / "predictions" / "subject_zscore_exclude_hc43_oof.csv")
    np.testing.assert_array_equal(folds, reference["fold"].to_numpy())
    for name in config["methods"]:
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        details = pd.read_csv(HERE / "predictions" / f"{name}_public_fold_probabilities.csv")
        assert len(oof) == 2760 and len(public) == 80
        np.testing.assert_array_equal(oof["original_trial_id"], data["original_trial_ids"])
        assert oof.groupby("subject_id").size().eq(40).all()
        np.testing.assert_array_equal(oof["fold"], folds)
        np.testing.assert_array_equal(oof["subject_id"], data["subject_ids"])
        np.testing.assert_allclose(public["prob. of positive emotion"], details[[f"fold_{i}_probability" for i in range(5)]].mean(axis=1))
        run.base.validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
        for p in (oof["prob_positive"], public["prob. of positive emotion"]):
            assert np.isfinite(p).all() and p.between(0, 1).all()
