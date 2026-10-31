import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("experiment021", HERE / "run_experiment.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


@pytest.fixture
def cache():
    return module.load_cache(HERE / CONFIG["spectral_cache"])


def test_columns_and_public_correspondence(cache):
    spectral, combined = module.validate_columns(cache)
    assert len(spectral) == 150 and len(combined) == 210
    for key in ("X_train", "X_public"):
        np.testing.assert_array_equal(cache[key][:, spectral], cache[key][:, combined][:, :150])


@pytest.mark.parametrize("key", ["X_train", "X_public"])
def test_rejects_misaligned_asymmetry(cache, key):
    cache[key][0, 150] += 1
    with pytest.raises(RuntimeError, match="asymmetry column alignment"):
        module.validate_columns(cache)


def test_rejects_wrong_column_names(cache):
    cache["feature_names"][[0, 1]] = cache["feature_names"][[1, 0]]
    with pytest.raises(RuntimeError, match="feature column alignment"):
        module.validate_columns(cache)


@pytest.mark.parametrize("key", list(module.META_COLUMNS) + ["public_user_ids", "public_trial_ids"])
def test_metadata_alignment_rejects_each_mismatch(cache, key):
    train = pd.read_csv(HERE / CONFIG["metadata_reference"])
    public = pd.read_csv(HERE / CONFIG["public_reference"])
    module.validate_metadata(cache, train, public)
    cache[key] = cache[key][::-1].copy()
    with pytest.raises(RuntimeError, match="metadata alignment"):
        module.validate_metadata(cache, train, public)


def test_fixed_fold_subject_isolation(cache):
    keep = cache["subject_ids"] != "HC43"
    reference, folds = module.load_outer_reference(HERE / CONFIG["outer_reference"], cache, keep)
    assert len(reference) == 2760 and reference.subject_id.nunique() == 69
    broken = folds.copy()
    broken[0] = (broken[0] + 1) % 5
    with pytest.raises(RuntimeError, match="subject leakage"):
        module.assert_subject_isolation(cache["subject_ids"][keep], broken)


def test_label_free_normalization_is_subject_local(cache):
    keep = cache["subject_ids"] != "HC43"
    subjects = cache["subject_ids"][keep]
    X = cache["X_train"][keep]
    _, folds = module.load_outer_reference(HERE / CONFIG["outer_reference"], cache, keep)
    normalized = module.normalize_by_subject(X, subjects, "zscore")
    for fold in range(5):
        valid = folds == fold
        np.testing.assert_array_equal(normalized[valid], module.normalize_by_subject(X[valid], subjects[valid], "zscore"))
    first = subjects == subjects[0]
    changed = X.copy()
    changed[~first] += 1000
    np.testing.assert_array_equal(normalized[first], module.normalize_by_subject(changed, subjects, "zscore")[first])
    for key, ids, count in (("X_train", "subject_ids", 40), ("X_public", "public_user_ids", 8)):
        values = module.normalize_by_subject(cache[key], cache[ids], "zscore")
        for subject in np.unique(cache[ids]):
            group = values[cache[ids] == subject]
            assert len(group) == count
            np.testing.assert_allclose(group.mean(axis=0), 0, atol=1e-10)
            raw_std = cache[key][cache[ids] == subject].std(axis=0)
            expected_std = np.where(raw_std > 1e-12, 1.0, raw_std)
            np.testing.assert_allclose(group.std(axis=0), expected_std, atol=1e-10)


def test_fixed_model_parameters():
    expected = {"max_iter": 200, "learning_rate": .03, "max_leaf_nodes": 15,
                "min_samples_leaf": 20, "l2_regularization": 1, "early_stopping": False}
    assert CONFIG["hgb"] == expected
    for fold in range(5):
        pipeline = module.model(CONFIG, 2026 + fold)
        assert isinstance(pipeline[0], StandardScaler)
        params = pipeline[-1].get_params()
        assert params["random_state"] == 2026 + fold
        assert all(params[key] == value for key, value in expected.items())


def test_output_completeness_and_exact_alignment(cache):
    reference = pd.read_csv(HERE / CONFIG["outer_reference"])
    for name in CONFIG["conditions"]:
        oof = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        assert len(oof) == 2760 and oof.subject_id.nunique() == 69
        assert "HC43" not in set(oof.subject_id)
        pd.testing.assert_frame_equal(oof.drop(columns="prob_positive"), reference.drop(columns="prob_positive"))
        assert oof.prob_positive.between(0, 1).all()
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(public) == 80 and not public.duplicated(["user_id", "trial_id"]).any()
        np.testing.assert_array_equal(public.user_id, cache["public_user_ids"])
        np.testing.assert_array_equal(public.trial_id, cache["public_trial_ids"])
        assert public["prob. of positive emotion"].between(0, 1).all()
        module.validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
