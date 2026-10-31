import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PRED = ROOT / "predictions"


def test_required_files_exist():
    for name in ["config.json", "run_experiment.py", "test_protocol.py", "实验报告.md", "metrics.json", "summary.csv"]:
        assert (ROOT / name).exists()
    assert PRED.is_dir()


def test_source_metadata_is_exactly_aligned():
    paths = ["nested_labram_raw_oof.csv", "nested_labram_zscore_oof.csv", "nested_labram_mean_oof.csv"]
    frames = [pd.read_csv(PRED / p) for p in paths]
    assert all(list(f.columns) == ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"] for f in frames)
    assert all(len(f) == 2760 and f.subject_id.nunique() == 69 for f in frames)
    for f in frames[1:]:
        assert f.iloc[:, :5].equals(frames[0].iloc[:, :5])


def test_hc43_excluded():
    f = pd.read_csv(PRED / "nested_labram_raw_oof.csv")
    assert "HC43" not in set(f.subject_id)


def test_outer_fold_count():
    m = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    assert m["protocol"]["outer_folds_from_experiment_002"]
    assert len(m["folds"]) == 15


def test_all_three_conditions_and_controls():
    files = {p.name for p in PRED.glob("*_oof.csv")}
    for c in ["labram_raw", "labram_zscore", "labram_mean"]:
        assert f"nested_{c}_oof.csv" in files
        assert f"fixed_09_01_{c}_oof.csv" in files


def test_candidate_grid_step_is_point_zero_five():
    grid = json.loads((ROOT / "config.json").read_text())[
        "candidate_hgb_weights"]
    assert len(grid) == 21 and np.allclose(np.diff(grid), 0.05)


def test_outer_labels_not_used_for_selection():
    m = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    assert m["protocol"]["outer_validation_labels_used_for_selection"] is False


def test_public_outputs_have_80_rows_and_valid_keys():
    for path in PRED.glob("*_public.csv"):
        f = pd.read_csv(path)
        assert len(f) == 80
        assert list(f.columns) == ["user_id", "trial_id", "prob. of positive emotion"]
        assert f[["user_id", "trial_id"]].duplicated().sum() == 0
        assert f.iloc[:, 2].between(0, 1).all()


def test_strict_nested_protocol_and_refits_are_recorded():
    m = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    assert m["protocol"]["strict_nested_refit"] is True
    assert m["protocol"]["inner_models_refit_per_fold"] is True
    assert m["protocol"]["outer_models_refit_on_complete_outer_train"] is True
    assert "否定" in m["protocol"]["old_protocol_status"]
    for row in m["folds"]:
        assert row["inner_model_fit_count"] >= 6
        assert row["outer_model_fit_count"] >= 2
        assert not set(row["outer_train_subjects"]) & set(row["outer_validation_subjects"])
        for split in row["inner_splits"]:
            assert not set(split["train_subjects"]) & set(split["validation_subjects"])


def test_cache_metadata_and_public_label_protocol():
    m = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    assert m["source_validation"]["metadata_alignment_checked"] is True
    assert m["source_validation"]["outer_folds_checked_against_002"] is True
    assert m["source_validation"]["public_labels_read"] is False
    assert m["protocol"]["hgb_features"].startswith("001")
    assert "004" in m["protocol"]["labram_features"]


def test_metrics_include_required_breakdowns_and_bias_limit():
    m = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))
    assert "model_selection_bias_limitations" in m
    for condition in ["labram_raw", "labram_zscore", "labram_mean"]:
        assert "nested" in m["runs"][condition]
        assert "fixed_09_01" in m["runs"][condition]
        assert "HC" in m["runs"][condition]["nested"]
        assert "DEP" in m["runs"][condition]["nested"]
        assert "subject_macro_auc" in m["runs"][condition]["nested"]
