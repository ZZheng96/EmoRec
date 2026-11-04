import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("exp166", HERE / "run_experiment.py")
e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e)


def test_protocol_and_candidates():
    config = json.loads((HERE / "config.json").read_text(encoding="utf8"))
    assert config["subject"] == "HC31"
    assert config["threshold"] == 0.5
    assert config["outer_validation_labels_used_for_selection"] is False
    assert config["posthoc_threshold_or_quota_search"] is False
    assert tuple(config["candidates"]) == e.CANDIDATES
    assert e.CANDIDATES == ("linear_svc:raw", "linear_svc:reverse", "logistic_regression:raw", "logistic_regression:reverse", "hgb:raw", "hgb:reverse")


def test_data_shape_and_trial_counts():
    config = json.loads((HERE / "config.json").read_text(encoding="utf8"))
    base_config = json.loads((e.BASE_DIR / "config.json").read_text(encoding="utf8"))
    x, ref = e.load_data(base_config)
    assert x.shape == (40, 720)
    assert ref.subject.nunique() == 1 and ref.subject.iloc[0] == "HC31"
    keys = e.trial_keys(ref)
    assert len(np.unique(keys)) == 8
    assert set(np.unique(np.bincount(pd.factorize(keys)[0]))) == {5}


def test_outputs_if_present():
    path = HERE / "predictions" / "trial_oof.csv"
    if not path.exists():
        return
    out = pd.read_csv(path)
    assert len(out) == 8 and out.prob_positive.between(0, 1).all()
    assert (out.window_count == 5).all()
    assert set(out.prediction_fixed05.unique()).issubset({0, 1})
    folds = pd.read_csv(HERE / "predictions" / "window_oof.csv")
    assert len(folds) == 40 and folds.row_index.is_unique
    assert folds.groupby("fold").size().eq(5).all()
