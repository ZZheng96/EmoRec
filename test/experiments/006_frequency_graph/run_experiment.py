import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.features import CHANNEL_NAMES, FREQUENCY_BANDS
from emo_rec.normalization import normalize_by_subject

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[2]


def load_baseline():
    path = EXPERIMENT_DIR.parent / "002_subject_normalization" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("frequency_graph_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
    return module


def graph_features(segments, sampling_rate=250):
    """Mean absolute off-diagonal band-limited Pearson connectivity per node."""
    segments = np.asarray(segments, dtype=np.float64)
    if segments.ndim != 3 or segments.shape[1:] != (30, 2500):
        raise ValueError("Expected 10-second segments with shape (n, 30, 2500)")
    if not np.isfinite(segments).all():
        raise ValueError("Segments must be finite")
    if sampling_rate != 250:
        raise ValueError("This experiment requires 250 Hz")
    spectrum = np.fft.rfft(segments, axis=-1)
    frequencies = np.fft.rfftfreq(segments.shape[-1], d=1 / sampling_rate)
    strengths = []
    for low, high in FREQUENCY_BANDS.values():
        mask = (frequencies >= low) & (frequencies < high)
        filtered = np.fft.irfft(spectrum * mask, n=segments.shape[-1], axis=-1)
        filtered -= filtered.mean(axis=-1, keepdims=True)
        norms = np.linalg.norm(filtered, axis=-1, keepdims=True)
        unit = np.divide(filtered, norms, out=np.zeros_like(filtered), where=norms > 1e-12)
        correlations = np.clip(unit @ unit.swapaxes(-1, -2), -1, 1)
        diagonal = np.arange(len(CHANNEL_NAMES))
        correlations[:, diagonal, diagonal] = 0
        strengths.append(np.abs(correlations).sum(axis=-1) / (len(CHANNEL_NAMES) - 1))
    names = [f"graph_strength_{channel}_{band}" for channel in CHANNEL_NAMES for band in FREQUENCY_BANDS]
    return np.stack(strengths, axis=-1).reshape(len(segments), -1), np.asarray(names)


def load_graph_cache(data, config):
    path = EXPERIMENT_DIR / config["graph_cache"]
    signature = json.dumps({"version": 1, "sampling_rate": config["sampling_rate"], "bands": FREQUENCY_BANDS, "graph": config["graph"]}, sort_keys=True)
    keys = ("subject_ids", "health_groups", "y_train", "original_trial_ids", "public_user_ids", "public_trial_ids")
    if path.exists():
        with np.load(path) as cache:
            if str(cache["signature"]) != signature or any(not np.array_equal(cache[key], data[key]) for key in keys):
                raise ValueError("Graph cache configuration or sample order mismatch")
            return cache["X_train"], cache["X_public"], cache["feature_names"]
    train, public = [], []
    actual = {key: [] for key in keys}
    for source in sorted((ROOT / "数据" / "训练集").rglob("*.mat")):
        segments, labels, trials = load_train_segments(source)
        features, names = graph_features(segments, config["sampling_rate"])
        train.append(features)
        actual["subject_ids"].extend([source.stem] * len(labels))
        actual["health_groups"].extend(["DEP" if source.parent.name == "抑郁症患者" else "HC"] * len(labels))
        actual["y_train"].extend(labels)
        actual["original_trial_ids"].extend(trials)
        print(f"Graph features: {source.stem}", flush=True)
    for source in sorted((ROOT / "数据" / "公开集" / "公开测试集").glob("*.mat"), key=lambda item: int(item.stem.replace("P_test", ""))):
        segments = load_public_test_segments(source)
        features, names = graph_features(segments, config["sampling_rate"])
        public.append(features)
        actual["public_user_ids"].extend([source.stem] * len(segments))
        actual["public_trial_ids"].extend(range(1, len(segments) + 1))
    for key in keys:
        if not np.array_equal(actual[key], data[key]):
            raise ValueError(f"Raw data order differs from 001 cache: {key}")
    X, X_public = np.concatenate(train), np.concatenate(public)
    path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path, X_train=X, X_public=X_public, feature_names=names, signature=signature, **{key: data[key] for key in keys})
    return X, X_public, names


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    baseline = load_baseline()
    data = baseline.load_feature_cache(EXPERIMENT_DIR / config["feature_cache"])
    folds = baseline.make_subject_folds(data["subject_ids"], data["health_groups"], config["random_seed"])
    power, power_names = baseline.select_log_bandpower(data["X_train"], data["feature_names"])
    public_power, _ = baseline.select_log_bandpower(data["X_public"], data["feature_names"])
    with threadpool_limits(limits=config["cpu_threads"]):
        graph, public_graph, graph_names = load_graph_cache(data, config)
        keep = data["subject_ids"] != config["excluded_subject"]
        selected = {key: data[key][keep] for key in baseline.TRAIN_DATA_KEYS}
        selected.update({key: data[key] for key in ("public_user_ids", "public_trial_ids")})
        if len(np.unique(selected["subject_ids"])) != config["expected_subjects"]:
            raise ValueError("Expected 69 training subjects")
        X = np.concatenate([power, graph], axis=1)[keep]
        X_public = np.concatenate([public_power, public_graph], axis=1)
        names = np.concatenate([power_names, graph_names])
        if X.shape != (2760, config["n_features"]) or X_public.shape != (80, config["n_features"]):
            raise ValueError("Unexpected feature dimensions")
        results = {"experiment_config": config, "sample_order_verified_against_001_cache": True, "runs": {}}
        rows = []
        for method in config["conditions"]:
            name = f"frequency_graph_{method}"
            normalized = normalize_by_subject(X, selected["subject_ids"], method, config["normalization"]["near_zero_std"])
            normalized_public = normalize_by_subject(X_public, selected["public_user_ids"], method, config["normalization"]["near_zero_std"])
            result = baseline.evaluate_run(name, normalized, normalized_public, selected, folds[keep], config, names)
            results["runs"][name] = result
            rows.append({"run_name": name, "n_subjects": result["n_subjects"], "n_samples": result["n_samples"], "n_features": result["n_features"], "oof_auc": result["overall"]["roc_auc"], "oof_binary_f1": result["overall"]["binary_f1"], "oof_balanced_accuracy": result["overall"]["balanced_accuracy"], "fold_auc_mean": result["fold_auc_mean"], "fold_auc_std": result["fold_auc_std"], "subject_auc_mean": result["subject_auc_mean"], "hc_pooled_auc": result["health_groups"]["HC"]["pooled"]["roc_auc"], "dep_pooled_auc": result["health_groups"]["DEP"]["pooled"]["roc_auc"]})
        (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(rows).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
