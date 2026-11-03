import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def test_saved_oof():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    out = pd.read_csv(HERE / "predictions/DEP01_window_csp_pairs_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8
    assert out.prediction_count.eq(4).all()
    assert len(pairs) == 32
    assert pairs.groupby("trial_index").size().eq(4).all()
    assert np.isclose(metrics["pooled_oof"]["auc"], out.assign(pred=out.prob_positive).pipe(lambda d: __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(d.y, d.pred)))
    assert metrics["n_windows"] == 40
    assert metrics["top_k"] == 5
    assert metrics["candidate_scores_source"] == "inner_oof_only"
    assert metrics["prediction_count"] == 4


def test_selection_has_window_protocol():
    selections = json.loads((HERE / "input_metadata/selections.json").read_text(encoding="utf8"))
    assert len(selections) == 16
    assert all(item["train_window_count"] == 30 and item["valid_window_count"] == 10 for item in selections)
    assert all(len(item["inner_scores"]) == 240 for item in selections)
    assert all(item["top_k"] == 5 and len(item["selected_topk"]) == 5 for item in selections)
    assert all(all(member["candidate_index"] < 120 for member in item["selected_topk"]) for item in selections)
    assert {member["direction"] for item in selections for member in item["selected_topk"]} <= {"raw", "reverse"}


def test_frozen_threshold():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    assert metrics["threshold"] == 0.5
    assert metrics["threshold_search"] is False
