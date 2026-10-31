import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("experiment029", HERE / "run_experiment.py")
protocol = importlib.util.module_from_spec(spec)
spec.loader.exec_module(protocol)


@pytest.fixture(scope="module")
def data():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    return config, *protocol.load_data(config)


def test_selection_label_audit_and_isolation(data):
    config, ref, features, _ = data
    subjects = ref.subject_id.to_numpy()
    for condition in config["conditions"]:
        for fold in range(5):
            mask = (ref.fold != fold).to_numpy()
            blocks = protocol.training_blocks(subjects, mask, condition)
            assert not set(subjects[blocks.ravel()]) & set(subjects[~mask])
            assert set(subjects[blocks.ravel()]) == set(subjects[mask])
            for subject in pd.unique(subjects[mask]):
                selected = blocks[subjects[blocks[:, 0]] == subject]
                rows = np.flatnonzero(subjects == subject)
                if condition == "all_40":
                    np.testing.assert_array_equal(selected[:, 0], rows)
                else:
                    assert len(selected) == 8
                    np.testing.assert_array_equal(selected[:, 0], rows[::5])
                    # Audit only: labels do not enter training_blocks.
                    assert sorted(ref.y_true.to_numpy()[selected[:, 0]]) == [0] * 4 + [1] * 4
                if condition == "one_per_trial_mean5":
                    np.testing.assert_array_equal(selected, rows.reshape(8, 5))
                    assert (np.ptp(ref.y_true.to_numpy()[selected], axis=1) == 0).all()
                    assert (np.ptp(ref.original_trial_id.to_numpy()[selected], axis=1) == 0).all()
    assert features["spectral"][0].shape == (2760, 150)
    assert features["temporal"][0].shape == (2760, 450)


def test_complete_outputs_and_independent_metrics(data):
    config, ref, _, public_ref = data
    result = json.loads((HERE / "metrics.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(HERE / "summary.csv")
    assert len(summary) == 9 and len(result["runs"]) == 9
    for condition in config["conditions"]:
        oofs, publics = {}, {}
        for name in ("spectral", "temporal", "fusion"):
            run = f"{condition}_{name}"
            oof = pd.read_csv(HERE / "predictions" / f"{run}_oof.csv")
            public = pd.read_csv(HERE / "predictions" / f"{run}_public.csv")
            pd.testing.assert_frame_equal(oof[ref.columns], ref)
            assert len(oof) == 2760 and oof.cache_row.is_unique
            assert not oof.duplicated(["subject_id", "subject_window_index"]).any()
            assert oof.groupby("subject_id").fold.nunique().eq(1).all()
            assert len(public) == 80 and public.groupby("user_id").size().eq(8).all()
            pd.testing.assert_frame_equal(public[public_ref.columns], public_ref)
            assert not public.duplicated(["user_id", "trial_id"]).any()
            p = oof.prob_positive.to_numpy()
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            assert public["prob. of positive emotion"].between(0, 1).all()
            actual = result["runs"][run]["overall"]
            assert actual["auc"] == pytest.approx(roc_auc_score(ref.y_true, p))
            assert actual["f1"] == pytest.approx(f1_score(ref.y_true, p >= .5))
            row = summary[(summary.condition == condition) & (summary.model == name)].iloc[0]
            baseline = result["runs"][f"all_40_{name}"]["overall"]
            for metric in ("auc", "f1"):
                assert row[metric] == pytest.approx(actual[metric])
                assert row[f"delta_{metric}_vs_all_40"] == pytest.approx(actual[metric] - baseline[metric])
            oofs[name], publics[name] = p, public["prob. of positive emotion"].to_numpy()
        np.testing.assert_allclose(oofs["fusion"], .5 * oofs["spectral"] + .5 * oofs["temporal"], atol=1e-15)
        np.testing.assert_allclose(publics["fusion"], .5 * publics["spectral"] + .5 * publics["temporal"], atol=1e-15)


@pytest.mark.parametrize("condition", ["all_40", "one_per_trial_8", "one_per_trial_mean5"])
def test_independent_retraining_and_determinism(data, condition):
    config, ref, features, _ = data
    y = ref.y_true.to_numpy()
    # Independent block construction, pipeline fitting and all outer/public prediction.
    with threadpool_limits(limits=1):
        for name, (X, XP) in features.items():
            expected = pd.read_csv(HERE / "predictions" / f"{condition}_{name}_oof.csv").prob_positive.to_numpy()
            for fold in [0, 1, 2, 3, 4, None]:
                train = np.ones(len(ref), dtype=bool) if fold is None else (ref.fold != fold).to_numpy()
                x_train, y_train = [], []
                for subject in pd.unique(ref.loc[train, "subject_id"]):
                    rows = np.flatnonzero(ref.subject_id.to_numpy() == subject)
                    if condition == "all_40":
                        x_train.extend(X[rows]); y_train.extend(y[rows])
                    elif condition == "one_per_trial_8":
                        x_train.extend(X[rows[::5]]); y_train.extend(y[rows[::5]])
                    else:
                        x_train.extend(X[rows].reshape(8, 5, -1).mean(axis=1)); y_train.extend(y[rows[::5]])
                fitted = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=2026 if fold is None else 2026 + fold))
                fitted.fit(np.asarray(x_train), np.asarray(y_train))
                prediction = fitted.predict_proba(XP if fold is None else X[~train])[:, 1]
                target = pd.read_csv(HERE / "predictions" / f"{condition}_{name}_public.csv")["prob. of positive emotion"].to_numpy() if fold is None else expected[~train]
                np.testing.assert_allclose(prediction, target, rtol=0, atol=1e-14)
                if fold == 0:
                    fitted.fit(np.asarray(x_train), np.asarray(y_train))
                    np.testing.assert_array_equal(prediction, fitted.predict_proba(X[~train])[:, 1])
