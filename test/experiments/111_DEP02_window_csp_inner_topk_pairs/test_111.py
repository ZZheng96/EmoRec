import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment111", HERE / "run_experiment.py")
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_saved_outputs_and_metrics():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    out = pd.read_csv(HERE / "predictions/DEP02_window_csp_inner_topk_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert metrics["top_ks"] == [1, 3, 5, 10, 20]
    assert len(out) == 8 and out.prediction_count.eq(4).all()
    assert len(pairs) == 32 and pairs.groupby("trial_index").size().eq(4).all()
    assert np.isclose(metrics["pooled_oof"]["auc"], roc_auc_score(out.y, out.prob_positive))
    assert np.isclose(metrics["pooled_oof"]["f1"], f1_score(out.y, out.prediction_fixed05))


def test_selection_protocol_and_isolation():
    selections = json.loads((HERE / "input_metadata/selections.json").read_text(encoding="utf8"))
    assert len(selections) == 16
    for item in selections:
        assert item["train_window_count"] == 30 and item["valid_window_count"] == 10
        assert item["top_ks"] == [1, 3, 5, 10, 20]
        assert len(item["inner_scores"]) == 240 and len(item["ranked_candidates"]) == 240
        assert item["selected_K"] in [1, 3, 5, 10, 20]
        assert len(item["inner_k_scores"]) == 5
        best = max(item["inner_k_scores"], key=lambda r: (-r["auc"], -r["f1"], -r["K"]))
        assert item["selected_K"] == best["K"]
        assert all(r["candidate_index"] < 120 for r in item["inner_scores"])


def test_fixed_threshold_and_no_outer_selection():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    provenance = json.loads((HERE / "input_metadata/provenance.json").read_text(encoding="utf8"))
    assert metrics["threshold"] == 0.5 and metrics["threshold_search"] is False
    assert metrics["outer_validation_labels_used_for_selection"] is False
    assert provenance["outer_validation_labels_used_for_selection"] is False
    assert experiment.TOP_KS == (1, 3, 5, 10, 20)
