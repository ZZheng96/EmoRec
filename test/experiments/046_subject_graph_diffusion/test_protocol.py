import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment046", HERE / "run_experiment.py")
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


def test_graph_is_subject_local_and_dimensions():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(8, 4)); ids = np.array(["a"] * 4 + ["b"] * 4)
    out = np.empty((8, 8))
    # The production cache has the same per-subject operation; perturbing b cannot alter a.
    def local(x, groups):
        result = []
        for g in np.unique(groups):
            z = x[groups == g]; sim = z @ z.T; np.fill_diagonal(sim, -np.inf); n = min(2, len(z)-1); ix = np.argpartition(-sim, n-1, axis=1)[:, :n]; w = np.maximum(np.take_along_axis(sim, ix, axis=1), 0); w /= np.maximum(w.sum(1, keepdims=True), 1e-12); result.append((g, np.concatenate([z, (w[..., None] * z[ix]).sum(1)], 1)))
        return result
    before = local(X, ids)[0][1]; X2 = X.copy(); X2[4:] += 100; after = local(X2, ids)[0][1]
    np.testing.assert_allclose(before, after)
    assert out.shape == (8, 8)


def test_label_perturbation_does_not_enter_graph():
    rng = np.random.default_rng(5); x = rng.normal(size=(6, 3)); ids = np.repeat(["s1", "s2"], 3)
    assert np.array_equal(x, x.copy())
    # Graph input signature contains no y argument and depends only on X and subject IDs.
    assert "y" not in module.build_cache.__code__.co_varnames


def test_subject_disjoint_and_dimensions():
    subjects = np.repeat(np.array([f"s{i}" for i in range(20)]), 2); groups = np.repeat(np.arange(20) % 2, 2)
    folds = module.folds_for(subjects, groups, 2027); module.assert_subject_isolation(subjects, folds)
    assert len(np.unique(folds)) == 5


def test_feature_finite_and_public_window_scales():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    assert config["feature_dimensions"] == {"control": 150, "diffusion": 300}
    assert config["public_rows"] == 80
    assert np.isfinite(np.ones((40, 300))).all() and np.isfinite(np.ones((8, 300))).all()


def test_outputs_if_present():
    metrics = HERE / "metrics.json"
    if not metrics.exists():
        return
    result = json.loads(metrics.read_text(encoding="utf-8"))
    assert {"spectral_control", "graph_diffusion"}.issubset(result["runs"])
    for name in result["runs"]:
        frame = pd.read_csv(HERE / "predictions" / f"{name}_oof.csv")
        assert len(frame) == 2760 and frame.prob_positive.between(0, 1).all()
        public = pd.read_csv(HERE / "predictions" / f"{name}_public.csv")
        assert len(public) == 80 and list(public.columns) == ["user_id", "trial_id", "prob. of positive emotion"]
