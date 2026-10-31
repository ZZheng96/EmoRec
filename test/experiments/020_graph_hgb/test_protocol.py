import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("experiment020", Path(__file__).with_name("run_experiment.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_fixed_hgb_parameters():
    config = {"hgb": {"max_iter": 200, "learning_rate": .03, "max_leaf_nodes": 15, "min_samples_leaf": 20, "l2_regularization": 1, "early_stopping": False}}
    estimator = module.model(config, 2026)[-1]
    params = estimator.get_params()
    assert params["max_iter"] == 200 and params["learning_rate"] == .03
    assert params["max_leaf_nodes"] == 15 and params["min_samples_leaf"] == 20
    assert params["l2_regularization"] == 1 and params["early_stopping"] is False


def test_cache_metadata_alignment_rejects_mismatch():
    base = {key: np.asarray([1, 2]) for key in module.META_KEYS}
    graph = {key: value.copy() for key, value in base.items()}
    graph["y_train"][1] = 3
    base.update({"public_user_ids": np.asarray(["P_test1"]), "public_trial_ids": np.asarray([1])})
    graph.update({"public_user_ids": np.asarray(["P_test1"]), "public_trial_ids": np.asarray([1])})
    with pytest.raises(RuntimeError, match="cache alignment"):
        module.assert_cache_alignment(base, graph)


def test_saved_outputs_complete_if_present():
    here = Path(__file__).parent
    if not (here / "metrics.json").exists():
        pytest.skip("Run experiment first")
    for name in ("spectral150", "graph150", "concat300"):
        oof = pd.read_csv(here / "predictions" / f"{name}_oof.csv")
        assert len(oof) == 2760 and oof.subject_id.nunique() == 69 and "HC43" not in set(oof.subject_id)
        assert oof.prob_positive.between(0, 1).all()
        public = pd.read_csv(here / "predictions" / f"{name}_public.csv")
        assert len(public) == 80 and not public.duplicated(["user_id", "trial_id"]).any()
        assert public["prob. of positive emotion"].between(0, 1).all()
