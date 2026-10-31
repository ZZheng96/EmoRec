import hashlib
import json
import platform
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy
import torch
from scipy.io import loadmat
from scipy.signal import stft
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
PRED = HERE / "predictions"
CACHE = HERE / "cache" / "stft_windows.npz"


def load_train_subject(path):
    blocks = []
    with h5py.File(path, "r") as mat:
        for key in ("EEG_data_neu", "EEG_data_pos"):
            x = np.asarray(mat[key][()], dtype=np.float64).T
            if x.shape != (30, 50000) or not np.isfinite(x).all():
                raise ValueError(f"invalid training EEG: {path}")
            blocks.append(x.reshape(30, 20, 2500).transpose(1, 0, 2))
    return np.concatenate(blocks), np.r_[np.zeros(20, dtype=np.int64), np.ones(20, dtype=np.int64)]


def load_public_subject(path):
    mat = loadmat(path)
    candidates = [np.asarray(v, dtype=np.float64) for k, v in mat.items() if not k.startswith("__") and isinstance(v, np.ndarray) and v.shape == (30, 20000)]
    if len(candidates) != 1 or not np.isfinite(candidates[0]).all():
        raise ValueError(f"invalid public EEG: {path}")
    return candidates[0].reshape(30, 8, 2500).transpose(1, 0, 2)


def compute_stft_tensor(segments):
    cfg = CONFIG["stft"]
    fs = CONFIG["sampling_rate"]
    rows = []
    frequencies = None
    for segment in np.asarray(segments):
        f, _, z = stft(segment, fs=fs, nperseg=cfg["nperseg"], noverlap=cfg["noverlap"], nfft=cfg["nfft"], boundary=cfg["boundary"], padded=False, axis=-1)
        mask = (f >= cfg["frequency_hz"][0]) & (f <= cfg["frequency_hz"][1])
        frequencies = f[mask]
        rows.append(np.log(np.maximum(np.abs(z[:, mask, :]) ** 2, 1e-12)).astype(np.float32))
    out = np.stack(rows)
    if out.ndim != 4 or out.shape[1] != 30 or not np.isfinite(out).all():
        raise ValueError(f"invalid STFT tensor shape={out.shape}")
    return out, frequencies


def build_cache():
    root = (HERE / "../../../数据").resolve()
    train_root = root / "训练集"
    X, y, subjects, groups, trials = [], [], [], [], []
    for group_name, group in (("正常人", "HC"), ("抑郁症患者", "DEP")):
        for path in sorted((train_root / group_name).glob("*.mat")):
            subject = path.stem
            if subject == CONFIG["excluded_subject"]:
                continue
            segments, labels = load_train_subject(path)
            tensor, frequencies = compute_stft_tensor(segments)
            X.append(tensor); y.append(labels); subjects.extend([subject] * len(labels)); groups.extend([group] * len(labels)); trials.extend(np.tile(np.arange(1, 5), 10))
    public_x, public_users, public_trials = [], [], []
    for path in sorted((root / "公开测试集").glob("P_test*.mat"), key=lambda p: int(p.stem[6:])):
        seg = load_public_subject(path); tensor, _ = compute_stft_tensor(seg); public_x.append(tensor); public_users.extend([path.stem] * 8); public_trials.extend(np.arange(1, 9))
    train = np.concatenate(X); labels = np.concatenate(y)
    np.savez_compressed(CACHE, X_train=train, y_train=np.concatenate(y), subject_ids=np.asarray(subjects), health_groups=np.asarray(groups), original_trial_ids=np.asarray(trials), X_public=np.concatenate(public_x), public_user_ids=np.asarray(public_users), public_trial_ids=np.asarray(public_trials), frequencies=frequencies)


def load_cache(path=CACHE):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def make_subject_folds(subjects, groups, seed):
    unique = np.unique(subjects); ug = np.asarray([groups[np.flatnonzero(subjects == s)[0]] for s in unique])
    assignment = {}
    for fold, (_, valid) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(unique, ug)):
        for s in unique[valid]: assignment[s] = fold
    return np.asarray([assignment[s] for s in subjects], dtype=np.int64)


def assert_subject_isolation(subjects, folds):
    for fold in range(5):
        assert not (set(subjects[folds == fold]) & set(subjects[folds != fold]))


def fit_channel_scaler(train_x):
    mean = train_x.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    std = train_x.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    return mean, np.maximum(std, 1e-6)


def transform_tensor(x, mean, std):
    out = (x - mean) / std
    if not np.isfinite(out).all(): raise ValueError("non-finite normalized STFT")
    return out.astype(np.float32, copy=False)


class TinySpectrogramCNN(nn.Module):
    def __init__(self, n_freq, n_time):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(30, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 16, 3, padding=1, groups=16), nn.Conv2d(16, 32, 1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        return self.head(self.features(x).flatten(1)).squeeze(1)


def seed_all(seed):
    np.random.seed(seed); torch.manual_seed(seed)


def train_cnn(x_train, y_train, x_valid, seed):
    seed_all(seed); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinySpectrogramCNN(x_train.shape[2], x_train.shape[3]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CONFIG["cnn"]["learning_rate"], weight_decay=CONFIG["cnn"]["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed); x_t = torch.from_numpy(x_train).to(device); y_t = torch.from_numpy(y_train.astype(np.float32)).to(device)
    model.train()
    losses = []
    for _ in range(CONFIG["cnn"]["epochs"]):
        order = rng.permutation(len(y_train)); total = 0.0
        for start in range(0, len(order), CONFIG["cnn"]["batch_size"]):
            ix = order[start:start + CONFIG["cnn"]["batch_size"]]; opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x_t[ix]), y_t[ix]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); total += float(loss.item()) * len(ix)
        losses.append(total / len(y_train))
    model.eval()
    with torch.no_grad(): pred = torch.sigmoid(model(torch.from_numpy(x_valid).to(device))).cpu().numpy()
    return pred, {"device": str(device), "epochs": CONFIG["cnn"]["epochs"], "loss": losses}


def classification_metrics(y, p):
    pred = p >= CONFIG["threshold"]; tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, pred, zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)), "threshold": CONFIG["threshold"], "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def summarize(y, p, reference, folds):
    subjects = []
    for subject in np.unique(reference.subject_id):
        mask = reference.subject_id.to_numpy() == subject; subjects.append({"subject_id": subject, "health_group": reference.loc[mask, "health_group"].iloc[0], **classification_metrics(y[mask], p[mask])})
    groups = {}
    for group in ("HC", "DEP"):
        mask = reference.health_group.to_numpy() == group; rows = [r for r in subjects if r["health_group"] == group]
        groups[group] = {"n_subjects": len(rows), "pooled": classification_metrics(y[mask], p[mask]), "subject_auc_mean": float(np.mean([r["auc"] for r in rows]))}
    return {"overall": classification_metrics(y, p), "folds": [{"fold": f, "n_subjects": int(reference.loc[folds == f, "subject_id"].nunique()), "n_samples": int((folds == f).sum()), **classification_metrics(y[folds == f], p[folds == f])} for f in range(5)], "subject_auc_mean": float(np.mean([r["auc"] for r in subjects])), "health_groups": groups, "subjects": subjects}


def save_predictions(name, reference, p, users=None, trials=None, public_p=None):
    PRED.mkdir(parents=True, exist_ok=True); out = reference.copy(); out["prob_positive"] = p; out.to_csv(PRED / f"{name}_oof.csv", index=False)
    if public_p is not None:
        public = pd.DataFrame({"user_id": users, "trial_id": trials, "prob. of positive emotion": public_p}); validate_submission(public, [f"P_test{i}" for i in range(1, 11)]); public.to_csv(PRED / f"{name}_public.csv", index=False)


def evaluate(run_name, data, folds, seed, mode):
    keep = data["subject_ids"] != CONFIG["excluded_subject"]
    x = data["X_train"][keep]; y = data["y_train"][keep]; subjects = data["subject_ids"][keep]; groups = data["health_groups"][keep]
    if mode == "hgb":
        with np.load((HERE / CONFIG["spectral_cache"]).resolve(), allow_pickle=False) as control:
            control_keep = control["subject_ids"] != CONFIG["excluded_subject"]
            data = dict(data); data["X_train"] = control["X_train"][control_keep, :150].astype(np.float32); data["X_public"] = control["X_public"][:, :150].astype(np.float32); x = data["X_train"]
    ref = pd.DataFrame({"subject_id": subjects, "health_group": groups, "original_trial_id": data["original_trial_ids"][keep], "fold": folds, "y_true": y})
    p = np.full(len(y), np.nan); audits = []
    for fold in range(5):
        tr, va = folds != fold, folds == fold
        if mode == "cnn":
            mean, std = fit_channel_scaler(x[tr]); tx = transform_tensor(x[tr], mean, std); vx = transform_tensor(x[va], mean, std); pred, audit = train_cnn(tx, y[tr], vx, seed + fold); audits.append({"fold": fold, "train_subjects": int(np.unique(subjects[tr]).size), "valid_subjects": int(np.unique(subjects[va]).size), "mean": mean.reshape(-1).tolist(), "std": std.reshape(-1).tolist(), **audit})
        else:
            flat = x.reshape(len(x), -1); model = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**CONFIG["hgb"], random_state=seed + fold)); model.fit(flat[tr], y[tr]); pred = model.predict_proba(flat[va])[:, 1]; audits.append({"fold": fold, "train_subjects": int(np.unique(subjects[tr]).size), "valid_subjects": int(np.unique(subjects[va]).size)})
        p[va] = pred
    save_predictions(run_name, ref, p)
    if mode == "cnn":
        mean, std = fit_channel_scaler(x); tx = transform_tensor(x, mean, std); px = transform_tensor(data["X_public"], mean, std); public_p, full_audit = train_cnn(tx, y, px, seed); audits.append({"fold": "full", "train_subjects": int(np.unique(subjects).size), **full_audit})
    else:
        model = make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**CONFIG["hgb"], random_state=seed)); model.fit(x.reshape(len(x), -1), y); public_p = model.predict_proba(data["X_public"].reshape(len(data["X_public"]), -1))[:, 1]
    save_predictions(run_name, ref, p, data["public_user_ids"], data["public_trial_ids"], public_p)
    return summarize(y, p, ref, folds), audits


def main():
    started = time.perf_counter()
    if not CACHE.exists(): build_cache()
    data = load_cache(); keep = data["subject_ids"] != CONFIG["excluded_subject"]
    if len(data["y_train"][keep]) != CONFIG["expected_oof_rows"]: raise RuntimeError("unexpected OOF rows")
    subjects, groups, y = data["subject_ids"][keep], data["health_groups"][keep], data["y_train"][keep]
    fixed_ref = pd.read_csv((HERE / CONFIG["fixed_outer_reference"]).resolve())
    fold_by_subject = fixed_ref.groupby("subject_id").fold.first().to_dict()
    if set(fold_by_subject) != set(subjects): raise RuntimeError("fixed fold subject set mismatch")
    fixed = np.asarray([fold_by_subject[s] for s in subjects], dtype=int)
    assert_subject_isolation(subjects, fixed)
    runs = {}; summary = []; audits = {}
    for mode in ("cnn", "hgb"):
        name = "stft_cnn_fixed" if mode == "cnn" else "spectral_hgb_fixed"
        result, audit = evaluate(name, data, fixed, 2026, mode); runs[name] = result; audits[name] = audit
        summary.append({"run_name": name, "repeat": "fixed", **{k: result["overall"][k] for k in ("auc", "f1", "balanced_accuracy")}, "subject_auc_mean": result["subject_auc_mean"], "hc_auc": result["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": result["health_groups"]["DEP"]["pooled"]["auc"]})
    for seed in CONFIG["repeat_seeds"]:
        folds = make_subject_folds(subjects, groups, seed); assert_subject_isolation(subjects, folds)
        for mode in ("cnn", "hgb"):
            name = f"{'stft_cnn' if mode == 'cnn' else 'spectral_hgb'}_seed{seed}"; result, audit = evaluate(name, data, folds, seed, mode); runs[name] = result; audits[name] = audit
            summary.append({"run_name": name, "repeat": seed, **{k: result["overall"][k] for k in ("auc", "f1", "balanced_accuracy")}, "subject_auc_mean": result["subject_auc_mean"], "hc_auc": result["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": result["health_groups"]["DEP"]["pooled"]["auc"]})
    provenance = {"cache_sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest(), "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "pandas": pd.__version__, "torch": torch.__version__, "elapsed_seconds": time.perf_counter() - started}
    result = {"experiment_config": CONFIG, "provenance": provenance, "protocol": CONFIG["protocol"], "training_audit": audits, "runs": runs}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"); pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False); (HERE / "input_metadata" / "sources.json").write_text(json.dumps({"cache": str(CACHE), "n_train": int(len(data["y_train"])), "stft_shape": list(data["X_train"].shape), "public_shape": list(data["X_public"].shape), "frequency_hz": data["frequencies"].tolist()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


if __name__ == "__main__": main()
