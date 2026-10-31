import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment050", HERE / "run_experiment.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_pairing_is_within_training_subject_and_fixed_count():
    rng = np.random.default_rng(4); X = rng.normal(size=(40, 4)); y = np.tile([0, 1], 20); subjects = np.repeat(["a", "b", "c", "d"], 10)
    px, py, left, right = module.build_pairs(X, y, subjects, np.ones(40, bool), 12, 7)
    assert len(px) == len(py) == 56
    assert np.all(subjects[left] == subjects[right])
    assert np.isfinite(px).all()


def test_subject_folds_are_disjoint_and_deterministic():
    subjects = np.repeat(np.array([f"s{i}" for i in range(20)]), 2); groups = np.repeat(np.arange(20) % 2, 2)
    a = module.make_subject_folds(subjects, groups, 2027); b = module.make_subject_folds(subjects, groups, 2027)
    np.testing.assert_array_equal(a, b)
    for fold in range(5): assert not set(subjects[a == fold]) & set(subjects[a != fold])


def test_training_label_perturbation_changes_pairs_but_validation_labels_do_not():
    rng = np.random.default_rng(2); X = rng.normal(size=(40, 3)); y = np.tile([0, 1], 20); subjects = np.repeat(["a", "b", "c", "d"], 10); mask = subjects != "d"
    _, _, left1, right1 = module.build_pairs(X, y, subjects, mask, 3, 10)
    y2 = y.copy(); y2[mask] = 1 - y2[mask]
    _, _, left2, right2 = module.build_pairs(X, y2, subjects, mask, 3, 10)
    assert not (np.array_equal(left1, left2) and np.array_equal(right1, right2))
    _, _, lv1, rv1 = module.build_pairs(X, y, subjects, mask, 3, 10)
    y3 = y.copy(); y3[~mask] = 1 - y3[~mask]
    _, _, lv2, rv2 = module.build_pairs(X, y3, subjects, mask, 3, 10)
    np.testing.assert_array_equal(lv1, lv2); np.testing.assert_array_equal(rv1, rv2)


def test_finite_rows_and_public_submission_if_present():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8")); cache = module.load_cache((HERE / config["spectral_cache"]).resolve())
    assert cache["X_train"].shape[1] >= 150 and np.isfinite(cache["X_train"][:, :150]).all()
    if (HERE / "metrics.json").exists():
        result = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
        assert set(result["runs"]) == {"pairwise_ranking", "logistic_regression", "spectral_hgb"}
        for name in result["runs"]:
            for repeat in range(1, 6):
                frame = pd.read_csv(HERE / "predictions" / f"{name}_repeat{repeat:02d}_oof.csv")
                assert len(frame) == 2760 and frame.subject_id.nunique() == 69 and "HC43" not in set(frame.subject_id)
                assert frame.prob_positive.between(0, 1).all()
            public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
            assert len(public) == 80 and public["prob. of positive emotion"].between(0, 1).all()
