import hashlib
import json
import platform
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.covariance import EuclideanTangentSpace, regularized_covariances
from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRED = HERE / "predictions"
META = HERE / "input_metadata"
KEYS = ["subject_id", "health_group", "physical_trial_index", "window_index", "y_true", "fold"]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def train_paths():
    return sorted((ROOT / "数据" / "训练集").rglob("*.mat"))


def gpu_covariances(segments, shrinkage, batch_size, device):
    if device.type != "cuda":
        raise RuntimeError("Experiment 214 requires CUDA for covariance extraction")
    out = []
    eye = torch.eye(segments.shape[1], dtype=torch.float64, device=device)
    for start in range(0, len(segments), batch_size):
        x = torch.as_tensor(segments[start:start + batch_size], dtype=torch.float64, device=device)
        x = x - x.mean(dim=-1, keepdim=True)
        cov = x @ x.transpose(-1, -2) / (x.shape[-1] - 1)
        scale = cov.diagonal(dim1=-2, dim2=-1).sum(-1) / cov.shape[-1]
        cov = (1 - shrinkage) * cov + shrinkage * scale[:, None, None] * eye
        out.append(cov.cpu().numpy())
        print(f"214 covariance GPU batch={min(start + batch_size, len(segments))}/{len(segments)}", flush=True)
    return np.concatenate(out, axis=0)


def build_reference(config):
    paths = train_paths()
    segments_all, labels, subjects, groups, trials, windows = [], [], [], [], [], []
    raw_files = []
    for path in paths:
        seg, y, trial_ids = load_train_segments(path)
        group = "DEP" if path.parent.name == "抑郁症患者" else "HC"
        for i, trial in enumerate(trial_ids):
            segments_all.append(seg[i]); labels.append(int(y[i])); subjects.append(path.stem); groups.append(group)
            trials.append(int((i // 5) + 1)); windows.append(int(i % 5)); raw_files.append(str(path))
        print(f"214 loaded {path.stem} windows={len(seg)}", flush=True)
    segments = np.asarray(segments_all, dtype=np.float64)
    keep = np.asarray(subjects) != config["excluded_subject"]
    segments = segments[keep]
    y = np.asarray(labels, dtype=np.int64)[keep]
    subjects = np.asarray(subjects)[keep]
    groups = np.asarray(groups)[keep]
    trials = np.asarray(trials, dtype=np.int64)[keep]
    windows = np.asarray(windows, dtype=np.int64)[keep]
    if len(y) != config["expected_train_rows"] or len(np.unique(subjects)) != config["expected_subjects"]:
        raise RuntimeError("unexpected filtered training data")
    ref = pd.DataFrame({"subject_id": subjects, "health_group": groups, "physical_trial_index": trials, "window_index": windows, "y_true": y})
    return segments, ref, raw_files


def make_folds(subjects, groups, seed, n_splits):
    unique = np.unique(subjects)
    unique_groups = np.asarray([groups[np.flatnonzero(subjects == s)[0]] for s in unique])
    split = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assignment = {}
    for fold, (_, valid) in enumerate(split.split(unique, unique_groups)):
        for subject in unique[valid]:
            assignment[subject] = fold
    return np.asarray([assignment[s] for s in subjects], dtype=np.int64)


def assert_protocol(ref, folds, n_splits):
    if ref.duplicated(["subject_id", "physical_trial_index", "window_index"]).any():
        raise RuntimeError("duplicate canonical EEG window key")
    counts = ref.groupby(["subject_id", "physical_trial_index"]).size()
    if not (counts == 5).all() or len(counts) != 552:
        raise RuntimeError("physical trial structure is invalid")
    for fold in range(n_splits):
        train_subjects = set(ref.loc[folds != fold, "subject_id"])
        valid_subjects = set(ref.loc[folds == fold, "subject_id"])
        if train_subjects & valid_subjects:
            raise RuntimeError("subject leakage")
        train_trials = set(zip(ref.loc[folds != fold, "subject_id"], ref.loc[folds != fold, "physical_trial_index"]))
        valid_trials = set(zip(ref.loc[folds == fold, "subject_id"], ref.loc[folds == fold, "physical_trial_index"]))
        if train_trials & valid_trials:
            raise RuntimeError("physical trial leakage")


def metrics(y, p):
    pred = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def summarize(ref, p, folds):
    y = ref.y_true.to_numpy()
    subject_rows = []
    for subject in ref.subject_id.unique():
        m = ref.subject_id.to_numpy() == subject
        subject_rows.append(metrics(y[m], p[m]))
    groups = {g: metrics(y[ref.health_group.to_numpy() == g], p[ref.health_group.to_numpy() == g]) for g in ("HC", "DEP")}
    trial_ref = ref.copy(); trial_ref["p"] = p
    trial = trial_ref.groupby(["subject_id", "health_group", "physical_trial_index"], as_index=False).agg(y_true=("y_true", "first"), p=("p", "mean"))
    fold_rows = [metrics(y[folds == f], p[folds == f]) for f in range(5)]
    return {"overall": metrics(y, p), "health_groups": groups, "trial": metrics(trial.y_true.to_numpy(), trial.p.to_numpy()), "subject_auc_mean": float(np.mean([r["auc"] for r in subject_rows])), "fold_auc_mean": float(np.mean([r["auc"] for r in fold_rows])), "fold_auc_min": float(np.min([r["auc"] for r in fold_rows])), "folds": fold_rows}


def subject_norm(x, ids, condition):
    return x if condition == "raw" else normalize_by_subject(x, ids, "zscore")


def run_condition(name, X, ref, folds, config):
    y = ref.y_true.to_numpy(); ids = ref.subject_id.to_numpy(); p = np.full(len(y), np.nan)
    for fold in range(config["n_splits"]):
        train, valid = folds != fold, folds == fold
        model = make_pipeline(StandardScaler(), LogisticRegression(**config["model"], random_state=2026 + fold))
        model.fit(subject_norm(X[train], ids[train], name.split("_")[-1]), y[train])
        p[valid] = model.predict_proba(subject_norm(X[valid], ids[valid], name.split("_")[-1]))[:, 1]
        print(f"214 condition={name} fold={fold + 1}/{config['n_splits']}", flush=True)
    if not np.isfinite(p).all(): raise RuntimeError(f"incomplete predictions {name}")
    out = ref.copy(); out["prob_positive"] = p; out.to_csv(PRED / f"{name}_oof.csv", index=False)
    return p, summarize(ref, p, folds)


def load_baseline(path):
    base = pd.read_csv(path)
    if len(base) != 2760: raise RuntimeError("057 baseline row count mismatch")
    base["physical_trial_index"] = base["original_trial_id"].astype(int) + 4 * base["y_true"].astype(int)
    base["window_index"] = base.groupby(["subject_id", "physical_trial_index"]).cumcount()
    if base[["subject_id", "physical_trial_index", "window_index"]].duplicated().any():
        raise RuntimeError("057 baseline canonical key is duplicated")
    return base


def key_merge(ref, p, base):
    left = ref.copy(); left["prob_covariance"] = p
    merge_keys = ["subject_id", "health_group", "physical_trial_index", "window_index", "y_true"]
    b = base[merge_keys + ["prob_positive"]].rename(columns={"prob_positive": "prob_baseline"})
    left = left.merge(b, on=merge_keys, how="left", validate="one_to_one")
    if left.prob_baseline.isna().any() or len(left) != len(ref): raise RuntimeError("057 key merge failed")
    left["prob_positive"] = 0.8 * left.prob_baseline + 0.2 * left.prob_covariance
    return left


def main():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8")); PRED.mkdir(exist_ok=True); META.mkdir(exist_ok=True)
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for experiment 214")
    device = torch.device("cuda"); print(f"214 device={torch.cuda.get_device_name(0)}", flush=True)
    segments, ref, raw_files = build_reference(config)
    cov = gpu_covariances(segments, config["shrinkage"], config["gpu_batch_size"], device)
    cpu_check = regularized_covariances(segments[:8], config["shrinkage"])
    if not np.allclose(cov[:8], cpu_check, rtol=1e-7, atol=1e-8): raise RuntimeError("GPU/CPU covariance mismatch")
    np.save(META / "covariances.npy", cov)
    all_runs = {}; summary = []
    baseline = load_baseline(ROOT / "test" / "experiments" / "057_multiscale_permutation_entropy" / "predictions" / "plus_pe_fixed002_oof.csv")
    for condition in config["conditions"]:
        repeat_results = []
        for seed in config["repeat_seeds"]:
            folds = make_folds(ref.subject_id.to_numpy(), ref.health_group.to_numpy(), seed, config["n_splits"]); assert_protocol(ref, folds, config["n_splits"])
            p = np.full(len(ref), np.nan)
            for fold in range(config["n_splits"]):
                train, valid = folds != fold, folds == fold
                space = EuclideanTangentSpace().fit(cov[train])
                Xtr = space.transform(cov[train]); Xva = space.transform(cov[valid])
                model = make_pipeline(StandardScaler(), LogisticRegression(**config["model"], random_state=2026 + fold))
                ids = ref.subject_id.to_numpy()
                model.fit(subject_norm(Xtr, ids[train], condition), ref.y_true.to_numpy()[train])
                p[valid] = model.predict_proba(subject_norm(Xva, ids[valid], condition))[:, 1]
                print(f"214 condition={condition} repeat={seed} fold={fold + 1}/5", flush=True)
            name = f"covariance_tangent_{condition}_repeat{seed}"
            out = ref.copy(); out["fold"] = folds; out["prob_positive"] = p; out.to_csv(PRED / f"{name}_oof.csv", index=False)
            result = summarize(ref, p, folds); result["repeat_seed"] = seed; repeat_results.append(result)
            fused = key_merge(out, p, baseline); fused.to_csv(PRED / f"covariance_plus_057_{condition}_repeat{seed}_oof.csv", index=False)
            fresult = summarize(ref, fused.prob_positive.to_numpy(), folds); fresult["repeat_seed"] = seed
            all_runs[f"fusion_{condition}_{seed}"] = fresult
        all_runs[f"covariance_{condition}"] = repeat_results
        for label, results in ((f"covariance_tangent_{condition}", repeat_results), (f"fusion_{condition}", [all_runs[f"fusion_{condition}_{s}"] for s in config["repeat_seeds"]])):
            aucs = [r["overall"]["auc"] for r in results]; f1s = [r["overall"]["f1"] for r in results]
            dep = [r["health_groups"]["DEP"]["auc"] for r in results]
            summary.append({"run_name": label, "auc_mean": np.mean(aucs), "auc_std": np.std(aucs, ddof=1), "f1_mean": np.mean(f1s), "dep_auc_mean": np.mean(dep), "trial_auc_mean": np.mean([r["trial"]["auc"] for r in results])})
    base_ref = baseline[["subject_id", "health_group", "physical_trial_index", "window_index", "y_true", "fold"]].copy()
    base_metrics = summarize(base_ref, baseline.prob_positive.to_numpy(), baseline.fold.to_numpy())
    all_runs["baseline_057_plus_pe"] = base_metrics
    (HERE / "metrics.json").write_text(json.dumps({"config": config, "device": str(device), "protocol": {"public_labels_used": False, "gpu_cpu_covariance_check": True, "physical_trial_isolation": True, "key_merge": True}, "runs": all_runs}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    report = "# 实验214：严格跨被试Riemannian协方差分支\n\n使用69名训练被试、2760个窗口和552个物理trial；每个外层折只用训练折拟合协方差参考矩阵、切空间、标准化器和分类器。协方差在CUDA上以float64计算，并完成CPU数值抽样核验。公开标签未读取。057融合按规范键一对一合并，禁止按数组位置合并。\n\n## 结果\n\n" + pd.DataFrame(summary).to_markdown(index=False) + "\n\n057冻结基线和所有方法的窗口AUC、F1、HC/DEP、trial指标详见metrics.json。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8")
    pd.DataFrame({"subject_id": ref.subject_id, "health_group": ref.health_group, "physical_trial_index": ref.physical_trial_index, "window_index": ref.window_index, "y_true": ref.y_true}).to_csv(META / "canonical_keys.csv", index=False)
    (META / "provenance.json").write_text(json.dumps({"train_rows": len(ref), "subjects": int(ref.subject_id.nunique()), "excluded_subject": config["excluded_subject"], "public_labels_used": False, "device": str(device), "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__, "raw_files": sorted(set(raw_files))}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1): main()
