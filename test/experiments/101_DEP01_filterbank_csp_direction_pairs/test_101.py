import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def test_saved_oof():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    out = pd.read_csv(HERE / "predictions/DEP01_filterbank_csp_pairs_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8
    assert out.prediction_count.eq(4).all()
    assert len(pairs) == 32
    assert pairs.groupby("trial_index").size().eq(4).all()
    assert np.isclose(metrics["pooled_oof"]["auc"], 0.875)
    assert np.isclose(metrics["pooled_oof"]["f1"], 6 / 7)


def test_selection_has_direction():
    selections = json.loads((HERE / "input_metadata/selections.json").read_text(encoding="utf8"))
    assert len(selections) == 16
    assert {item["selected_direction"] for item in selections} <= {"raw", "reverse"}
