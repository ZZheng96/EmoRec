import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def test_saved_trial_oof_and_metrics():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    out = pd.read_csv(HERE / "predictions/DEP01_trial_mean_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8
    assert out.prediction_count.eq(4).all()
    assert len(pairs) == 32
    assert pairs.groupby("trial_index").size().eq(4).all()
    assert np.isfinite(out.prob_positive).all()
    assert np.isclose(metrics["pooled_oof"]["auc"], 0.25)
    assert np.isclose(metrics["pooled_oof"]["f1"], 0.4)


def test_trial_structure():
    trials = json.loads((HERE / "input_metadata/trials.json").read_text(encoding="utf8"))
    assert len(trials) == 8
    assert sorted(item["y"] for item in trials) == [0, 0, 0, 0, 1, 1, 1, 1]
    assert all(len(item["window_rows"]) == 5 for item in trials)
