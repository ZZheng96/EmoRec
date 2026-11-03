import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BANDS = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45))
CHANNELS = ("FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2")
MIRROR_PAIRS = (("FP1", "FP2"), ("F7", "F8"), ("F3", "F4"), ("FT7", "FT8"), ("FC3", "FC4"), ("T3", "T4"), ("C3", "C4"), ("TP7", "TP8"), ("CP3", "CP4"), ("T5", "T6"), ("P3", "P4"), ("O1", "O2"))
PAIR_INDICES = tuple((CHANNELS.index(a), CHANNELS.index(b)) for a, b in MIRROR_PAIRS)
REGIONS = {"frontal": ("FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"), "central": ("FC3", "FCZ", "FC4", "C3", "CZ", "C4"), "parietal": ("CP3", "CPZ", "CP4", "P3", "PZ", "P4"), "occipital": ("O1", "OZ", "O2"), "temporal": ("FT7", "FT8", "T3", "T4", "TP7", "TP8", "T5", "T6")}
REGION_INDICES = {k: tuple(CHANNELS.index(c) for c in v) for k, v in REGIONS.items()}
BLOCKS = ("A", "R", "G_A", "G_R")
STANDARDIZATIONS = ("raw", "scale")
CANDIDATES = tuple((block, standardization, model, value) for block in BLOCKS for standardization in STANDARDIZATIONS for model, value in (("LR", 0.01), ("LR", 0.1), ("LR", 1), ("LR", 10), ("LDA", "auto"), ("LDA", 0.1)))
FIXED_THRESHOLD = 0.5
POSITIVE_QUOTA = 4


def metric(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, np.asarray(p) >= FIXED_THRESHOLD, zero_division=0))}


def quota_predictions(p):
    out = np.zeros(8, dtype=int)
    out[np.argsort(-np.asarray(p), kind="stable")[:POSITIVE_QUOTA]] = 1
    return out


def load_subject(subject):
    paths = [p for p in (ROOT / "数据" / "训练集").glob("*/*.mat") if p.stem == subject]
    assert len(paths) == 1
    with h5py.File(paths[0], "r") as mat:
        pieces = []
        for key in ("EEG_data_neu", "EEG_data_pos"):
            data = np.asarray(mat[key][()], dtype=np.float64).T
            assert data.shape == (30, 50000)
            pieces.append(data.reshape(30, 20, 2500).transpose(1, 0, 2))
    return np.concatenate(pieces), np.r_[np.zeros(20, dtype=int), np.ones(20, dtype=int)], np.r_[np.repeat(np.arange(4), 5), 4 + np.repeat(np.arange(4), 5)], paths[0]


def window_features(x):
    powers = []
    for low, high in BANDS:
        z = sosfiltfilt(butter(4, [low, high], btype="bandpass", fs=250, output="sos"), x, axis=-1)
        powers.append(np.log(np.maximum(np.var(z, axis=-1), 1e-30)))
    power = np.stack(powers, axis=1)
    alpha = power[:, 2, :]
    relative = alpha - power.mean(axis=1)
    alpha_diff = np.stack([alpha[:, a] - alpha[:, b] for a, b in PAIR_INDICES], axis=1)
    relative_diff = np.stack([relative[:, a] - relative[:, b] for a, b in PAIR_INDICES], axis=1)
    regional_a = np.stack([alpha[:, idx].mean(axis=1) for idx in REGION_INDICES.values()], axis=1)
    regional_r = np.stack([relative[:, idx].mean(axis=1) for idx in REGION_INDICES.values()], axis=1)
    regional_a_diff = np.stack([alpha_diff[:, [i for i, (a, b) in enumerate(PAIR_INDICES) if a in idx or b in idx]].mean(axis=1) for idx in REGION_INDICES.values()], axis=1)
    regional_r_diff = np.stack([relative_diff[:, [i for i, (a, b) in enumerate(PAIR_INDICES) if a in idx or b in idx]].mean(axis=1) for idx in REGION_INDICES.values()], axis=1)
    features = {"A": np.c_[alpha, alpha_diff], "R": np.c_[relative, relative_diff], "G_A": np.c_[regional_a, regional_a_diff], "G_R": np.c_[regional_r, regional_r_diff]}
    return features, {"power": power, "alpha": alpha, "relative": relative, "alpha_diff": alpha_diff, "relative_diff": relative_diff}


def aggregate_trials(features, y, trial):
    return {k: np.asarray([v[trial == t].mean(axis=0) for t in range(8)]) for k, v in features.items()}, np.asarray([y[trial == t][0] for t in range(8)])


def trial_pairs(trial):
    trial = np.asarray(trial)
    for p in np.flatnonzero(trial >= 4):
        for n in np.flatnonzero(trial < 4):
            va = np.sort([p, n])
            yield np.setdiff1d(np.arange(len(trial)), va), va


def predict(train, y_train, valid, candidate):
    block, standardization, model_name, value = candidate
    a, b = train[block], valid[block]
    if standardization == "scale":
        scaler = StandardScaler().fit(a)
        a, b = scaler.transform(a), scaler.transform(b)
    if model_name == "LR":
        estimator = LogisticRegression(C=float(value), solver="liblinear", max_iter=3000, random_state=2026)
    else:
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=value)
    estimator.fit(a, y_train)
    return estimator.predict_proba(b)[:, 1]


def select_inner(x, y, trial_ids):
    scores = []
    for index, candidate in enumerate(CANDIDATES):
        folds = [(predict({k: v[tr] for k, v in x.items()}, y[tr], {k: v[va] for k, v in x.items()}, candidate), va) for tr, va in trial_pairs(trial_ids)]
        for direction_index, direction in enumerate(("raw", "reverse")):
            p, yy = [], []
            for value, va in folds:
                p.extend((value if direction == "raw" else 1 - value).tolist())
                yy.extend(y[va].tolist())
            scores.append({"candidate_index": index, "candidate": candidate, "direction": direction, "direction_index": direction_index, **metric(np.asarray(yy), np.asarray(p))})
    return sorted(scores, key=lambda r: (-r["auc"], -r["f1"], r["candidate_index"], r["direction_index"]))


def frozen_hashes():
    return {n: hashlib.sha256((HERE / n).read_bytes()).hexdigest() for n in ("run_experiment.py", "config.json", "冻结协议.md")}


def main():
    started = time.perf_counter()
    x, y_window, window_trial, source = load_subject("DEP02")
    window, audit = window_features(x)
    features, y = aggregate_trials(window, y_window, window_trial)
    assert y.tolist() == [0, 0, 0, 0, 1, 1, 1, 1] and all(v.shape[0] == 8 for v in features.values())
    values, rows, selections = [], [], []
    with threadpool_limits(limits=1):
        for pair_id, (tr, va) in enumerate(trial_pairs(np.arange(8))):
            train = {k: v[tr] for k, v in features.items()}; valid = {k: v[va] for k, v in features.items()}
            ranked = select_inner(train, y[tr], tr); chosen = ranked[0]
            p = predict(train, y[tr], valid, tuple(chosen["candidate"]))
            if chosen["direction"] == "reverse": p = 1 - p
            values.append((va, p))
            selections.append({"pair_id": pair_id, "train_trial_indices": tr.tolist(), "valid_trial_indices": va.tolist(), "train_trial_count": 6, "valid_trial_count": 2, "raw_window_count": 40, "selected_candidate": chosen["candidate"], "selected_direction": chosen["direction"], "inner_scores": ranked, "selection_source": "inner 3x3 trial OOF only"})
            for t, probability in zip(va, p):
                rows.append({"pair_id": pair_id, "trial_index": int(t), "trial_key": f"{int(y[t])}::{int(t % 4) + 1}", "y": int(y[t]), "prob_positive": float(probability), "window_count": 5})
    sums, counts = np.zeros(8), np.zeros(8, dtype=int)
    for va, p in values: sums[va] += p; counts[va] += 1
    p = sums / counts
    out = pd.DataFrame({"trial_index": np.arange(8), "trial_key": [f"{y[t]}::{t % 4 + 1}" for t in range(8)], "y": y, "prob_positive": p, "prediction_count": counts})
    out["prediction_quota4"] = quota_predictions(p); out["prediction_fixed05"] = (p >= FIXED_THRESHOLD).astype(int)
    out.to_csv(HERE / "predictions/DEP02_alpha_oof.csv", index=False); pd.DataFrame(rows).to_csv(HERE / "predictions/pair_predictions.csv", index=False)
    (HERE / "input_metadata/selections.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2), encoding="utf8")
    frozen = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "hashes": frozen_hashes()}
    (HERE / "input_metadata/freeze.json").write_text(json.dumps(frozen, ensure_ascii=False, indent=2), encoding="utf8")
    provenance = {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "script_sha256": hashlib.sha256((HERE / "run_experiment.py").read_bytes()).hexdigest(), "python": platform.python_version(), "threads": 1, "protocol": "prespecified_subject_specific_alpha_A_R_region_nested_pairs", "audit_hypothesis": "DEP02四个物理trial的alpha(8-13Hz) pos-neu均为负", "outer_validation_labels_used_for_selection": False, "feature_audit_shapes": {k: list(v.shape) for k, v in audit.items()}, "freeze": frozen, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "data_description_sha256": hashlib.sha256((ROOT / "数据/数据集说明文档.md").read_bytes()).hexdigest()}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    pair_metrics = [metric(y[va], pv) for va, pv in values]
    result = {"experiment_id": "125_DEP02_prespecified_alpha_pairs", "subject": "DEP02", "pooled_oof": {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, out.prediction_quota4, zero_division=0))}, "fixed_0.5": metric(y, p), "pair_mean": {k: float(np.mean([m[k] for m in pair_metrics])) for k in ("auc", "f1")}, "pair_metrics": pair_metrics, "n_trials": 8, "n_windows": 40, "windows_per_trial": 5, "feature_dimensions": {k: int(v.shape[1]) for k, v in features.items()}, "candidate_model_count": len(CANDIDATES), "candidate_count": len(CANDIDATES) * 2, "blocks": BLOCKS, "standardizations": STANDARDIZATIONS, "classifiers": {"LR_C": [0.01, 0.1, 1, 10], "LDA_shrinkage": ["auto", 0.1]}, "positive_quota": 4, "quota_search": False, "threshold": 0.5, "threshold_search": False, "outer_validation_labels_used_for_selection": False, "elapsed_seconds": time.perf_counter() - started}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    report = f"# 实验125：DEP02审计后预设alpha被试特异探索\n\n- 研究假设：数据审计观察到四个物理trial的alpha(8–13Hz) pos-neu均为负；本轮固定验证，不在outer/inner选择频带。\n- pooled OOF AUC：**{result['pooled_oof']['auc']:.6f}**；quota4 F1：**{result['pooled_oof']['f1']:.6f}**。\n- fixed0.5 AUC/F1：**{result['fixed_0.5']['auc']:.6f} / {result['fixed_0.5']['f1']:.6f}**；pair均值 AUC/F1：**{result['pair_mean']['auc']:.6f} / {result['pair_mean']['f1']:.6f}**。\n- 候选：4块×2标准化×6分类器=48模型，raw/reverse共96项；outer4×4、inner3×3，单线程CPU。耗时：**{result['elapsed_seconds']:.3f}秒**。\n- 结论边界：这是审计后提出的DEP02被试特异探索，仅有8个物理trial的开发OOF证据，不能宣称跨被试泛化或0.95达标。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
