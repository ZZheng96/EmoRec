import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def test_saved_oof_and_protocol_invariants():
    metrics = json.loads((HERE / "metrics.json").read_text(encoding="utf8"))
    out = pd.read_csv(HERE / "predictions/DEP01_filterbank_csp_pairs_oof.csv")
    pairs = pd.read_csv(HERE / "predictions/pair_predictions.csv")
    assert len(out) == 8
    assert out.prediction_count.eq(4).all()
    assert len(pairs) == 32
    assert pairs.groupby("trial_index").size().eq(4).all()
    assert np.isfinite(out.prob_positive).all()
    assert np.isclose(metrics["pooled_oof"]["auc"], 0.0)
    assert np.isclose(metrics["pooled_oof"]["f1"], 2 / 9)


def test_trial_balance():
    out = pd.read_csv(HERE / "predictions/DEP01_filterbank_csp_pairs_oof.csv")
    assert sorted(out.y.tolist()) == [0, 0, 0, 0, 1, 1, 1, 1]
