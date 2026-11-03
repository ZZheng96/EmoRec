import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent

def test_saved_outputs():
    metrics=json.loads((HERE/"metrics.json").read_text(encoding="utf8")); out=pd.read_csv(HERE/"predictions/DEP02_trial_shrinkage_lda_oof.csv"); pairs=pd.read_csv(HERE/"predictions/pair_predictions.csv")
    assert len(out)==8 and len(pairs)==32 and out.prediction_count.eq(4).all() and pairs.groupby("trial_index").size().eq(4).all()
    assert np.isfinite(out.prob_positive).all() and out.prediction_quota4.sum()==4
    assert np.isclose(metrics["pooled_oof"]["auc"],roc_auc_score(out.y,out.prob_positive)); assert np.isclose(metrics["pooled_oof"]["f1"],f1_score(out.y,out.prediction_quota4)); assert np.isclose(metrics["fixed_0.5"]["f1"],f1_score(out.y,out.prediction_fixed05))

def test_protocol():
    config=json.loads((HERE/"config.json").read_text(encoding="utf8")); assert config["target_subject"]=="DEP02"; assert config["outer_validation_labels_used_for_selection"] is False; assert config["positive_quota"]==4; assert config["blocks"]=={"all":720,"spectral":150,"temporal":450,"PE":120}

def test_metadata():
    out=pd.read_csv(HERE/"predictions/DEP02_trial_shrinkage_lda_oof.csv"); assert out.y.tolist()==[0,0,0,0,1,1,1,1]; assert all(len(json.loads(v))==5 for v in out.window_rows)
    p=json.loads((HERE/"input_metadata/provenance.json").read_text(encoding="utf8")); assert p["metadata_rows"]==40 and all(v==5 for v in p["trial_window_counts"].values())
