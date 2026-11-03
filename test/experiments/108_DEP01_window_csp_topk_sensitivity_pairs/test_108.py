import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
KS = [1, 3, 5, 10, 20, 240]


def test_saved_outputs_and_metrics():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    assert metrics["top_ks"] == KS
    assert metrics["k_selection"].startswith("none")
    for k in KS:
        out = pd.read_csv(HERE / "predictions" / f"DEP01_window_csp_topk_{k}_pooled.csv")
        pairs = pd.read_csv(HERE / "predictions" / f"DEP01_window_csp_topk_{k}_pairs.csv")
        assert len(out) == 8 and out.prediction_count.eq(4).all()
        assert len(pairs) == 32 and pairs.groupby("trial_index").size().eq(4).all()
        assert np.isclose(metrics["results"][str(k)]["pooled_oof"]["auc"], roc_auc_score(out.y, out.prob_positive))


def test_selection_protocol():
    selections = json.loads((HERE / "input_metadata/selections.json").read_text(encoding="utf8"))
    assert len(selections) == 16
    assert all(item["train_window_count"] == 30 and item["valid_window_count"] == 10 for item in selections)
    assert all(len(item["inner_scores"]) == 240 and item["top_ks"] == KS for item in selections)
    assert all(len(item["ranked_candidates"]) == 240 for item in selections)


def test_fixed_threshold_and_no_outer_selection():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    provenance = json.loads((HERE / "input_metadata/provenance.json").read_text(encoding="utf8"))
    assert metrics["threshold"] == 0.5 and metrics["threshold_search"] is False
    assert provenance["outer_validation_labels_used_for_selection"] is False
