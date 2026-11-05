import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("base032", HERE.parent / "032_repeated_subject_folds" / "run_experiment.py")
base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.features import CHANNEL_NAMES, FREQUENCY_BANDS

CONFIG = json.loads((HERE / "config.json").read_text(encoding="utf8"))
BANDS = list(FREQUENCY_BANDS.items())
KEYS = ["subject_ids", "health_groups", "original_trial_ids", "y_train", "public_user_ids", "public_trial_ids"]


def de_features(segments):
    x = np.asarray(segments, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (30, 2500): raise ValueError("expected (n,30,2500)")
    out = []
    nyq = CONFIG["sampling_rate"] / 2
    for _, (low, high) in BANDS:
        sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
        filtered = sosfiltfilt(sos, x, axis=-1)
        var = np.maximum(np.var(filtered, axis=-1), 1e-20)
        out.append(0.5 * np.log(2 * np.pi * np.e * var))
    return np.stack(out, axis=-1).reshape(len(x), -1)


def cache_data(path):
    if path.exists():
        with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}
    train, public, meta = [], [], {k: [] for k in KEYS}
    files = sorted((ROOT / "数据/训练集").rglob("*.mat"))
    public_files = sorted((ROOT / "数据/公开集/公开测试集").glob("*.mat"), key=lambda p: int(p.stem.replace("P_test", "")))
    for i, f in enumerate(files, 1):
        seg, y, trials = load_train_segments(f); train.append(de_features(seg))
        meta["subject_ids"].extend([f.stem] * len(y)); meta["health_groups"].extend(["DEP" if f.parent.name == "抑郁症患者" else "HC"] * len(y)); meta["original_trial_ids"].extend(trials); meta["y_train"].extend(y)
        print(f"DE {i}/{len(files)} {f.stem}", flush=True)
    for i, f in enumerate(public_files, 1):
        seg = load_public_test_segments(f); public.append(de_features(seg)); meta["public_user_ids"].extend([f.stem] * len(seg)); meta["public_trial_ids"].extend(range(1, len(seg) + 1)); print(f"DE public {i}/{len(public_files)}", flush=True)
    arr = {"X_train": np.concatenate(train), "X_public": np.concatenate(public), "feature_names": np.asarray([f"de_{ch}_{band}" for band, _ in BANDS for ch in CHANNEL_NAMES])}
    arr.update({k: np.asarray(v) for k, v in meta.items()}); np.savez_compressed(path, **arr); return arr


def metrics(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= CONFIG["threshold"], zero_division=0)), "balanced_accuracy": float(balanced_accuracy_score(y, p >= CONFIG["threshold"]))}


def summarize(y, p, ref):
    groups = {g: metrics(y[ref.health_group.to_numpy() == g], p[ref.health_group.to_numpy() == g]) for g in ("HC", "DEP")}
    subs = [metrics(y[ref.subject_id.to_numpy() == s], p[ref.subject_id.to_numpy() == s])["auc"] for s in np.unique(ref.subject_id)]
    return {"overall": metrics(y, p), "health_groups": groups, "subject_auc_mean": float(np.mean(subs)), "n": int(len(y))}


def fit_oof(X, XP, y, subs, groups, seeds):
    pred = {}; publics = {}
    for seed in seeds:
        folds = base.make_subject_folds(subs, groups, seed); base.assert_subject_isolation(subs, folds); p = np.full(len(y), np.nan)
        for fold in range(5):
            train, valid = folds != fold, folds == fold; model = base.make_model(CONFIG, CONFIG["outer_model_seed_base"] + fold); model.fit(X[train], y[train]); p[valid] = model.predict_proba(X[valid])[:, 1]; print(f"OOF seed={seed} fold={fold+1}/5", flush=True)
        pred[seed] = p; model = base.make_model(CONFIG, CONFIG["outer_model_seed_base"]); model.fit(X, y); publics[seed] = model.predict_proba(XP)[:, 1]
    return pred, publics


def main():
    start = time.perf_counter(); (HERE / "predictions").mkdir(exist_ok=True); (HERE / "input_metadata").mkdir(exist_ok=True)
    d = cache_data(HERE / CONFIG["de_cache"])
    def load_npz(path):
        with np.load(path, allow_pickle=False) as z: return {k: z[k] for k in z.files}
    s = load_npz((HERE / CONFIG["spectral_cache"]).resolve()); t = load_npz((HERE / CONFIG["temporal_cache"]).resolve())
    for k in KEYS:
        if not np.array_equal(d[k], s[k]): raise RuntimeError(f"cache alignment mismatch: {k}")
    keep = s["subject_ids"] != CONFIG["excluded_subject"]; subs, groups, y = s["subject_ids"][keep], s["health_groups"][keep], s["y_train"][keep]; ref = pd.DataFrame({"subject_id": subs, "health_group": groups, "original_trial_id": s["original_trial_ids"][keep], "y_true": y})
    z = lambda a, ids: __import__("emo_rec.normalization", fromlist=["normalize_by_subject"]).normalize_by_subject(a, ids, "zscore")
    de, dep = z(d["X_train"][keep], subs), z(d["X_public"], d["public_user_ids"])
    spectral = z(s["X_train"][keep, :150], subs); temporal = z(t["X_train"][keep], subs); spectral_p = z(s["X_public"][:, :150], s["public_user_ids"]); temporal_p = z(t["X_public"], t["public_user_ids"])
    conditions = {"de_only": (de, dep), "de_plus_current": (np.column_stack([spectral, temporal, de]), np.column_stack([spectral_p, temporal_p, dep]))}
    results, allpred, allpub = {}, {}, {}
    for name, (X, XP) in conditions.items():
        pp, pub = fit_oof(X, XP, y, subs, groups, CONFIG["repeat_seeds"]); allpred[name] = pp; allpub[name] = pub
        results[name] = {"repeats": {str(k): summarize(y, v, ref) for k, v in pp.items()}, "aggregate_auc": float(np.mean([summarize(y, v, ref)["overall"]["auc"] for v in pp.values()])), "aggregate_f1": float(np.mean([summarize(y, v, ref)["overall"]["f1"] for v in pp.values()]))}
        for seed, p in pp.items(): ref.assign(fold=base.make_subject_folds(subs, groups, seed), prob_positive=p).to_csv(HERE / "predictions" / f"{name}_repeat{seed}_oof.csv", index=False)
        pd.DataFrame({"user_id": s["public_user_ids"], "trial_id": s["public_trial_ids"], "prob. of positive emotion": np.mean(list(pub.values()), axis=0)}).to_csv(HERE / "predictions" / f"{name}_public.csv", index=False)
    plus = pd.read_csv(HERE / CONFIG["plus_pe_oof"]); graph = pd.read_csv(HERE / CONFIG["graph_oof"]); keys = ["subject_id", "health_group", "original_trial_id", "fold", "y_true"]
    if not plus[keys].equals(graph[keys]): raise RuntimeError("057/006 OOF alignment mismatch")
    basep = .85 * plus.prob_positive.to_numpy() + .15 * graph.prob_positive.to_numpy(); depred = allpred["de_only"][CONFIG["repeat_seeds"][0]]; fixed = {"de_plus_fixed_base": .5 * basep + .5 * depred, "de_expert_fixed_base": CONFIG["fusion"]["de_weight"] * depred + CONFIG["fusion"]["base_weight"] * basep}
    for name, p in fixed.items(): results[name] = {"overall": metrics(y, p), "summary": summarize(y, p, ref)}; ref.assign(prob_positive=p).to_csv(HERE / "predictions" / f"{name}_fixed002_oof.csv", index=False)
    rows = [{"condition": k, "auc": v["aggregate_auc"] if "aggregate_auc" in v else v["overall"]["auc"], "f1": v["aggregate_f1"] if "aggregate_f1" in v else v["overall"]["f1"]} for k, v in results.items()]; pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)
    baseline_auc = results["de_expert_fixed_base"]["overall"]["auc"]; best = max(r["auc"] for r in rows if r["condition"] != "de_expert_fixed_base"); decision = bool(baseline_auc - best >= CONFIG["acceptance"]["delta_auc_min"])
    provenance = {"python": platform.python_version(), "numpy": np.__version__, "strict_subjectwise_oof": True, "validation_labels_used_for_selection": False, "public_labels_used": False, "elapsed_seconds": time.perf_counter() - start}
    (HERE / "metrics.json").write_text(json.dumps({"config": CONFIG, "provenance": provenance, "runs": results, "fixed_fusion": {"plus_pe_weight": .85, "graph_weight": .15}, "stop_line": {"delta_auc": baseline_auc - best, "passed": decision}}, ensure_ascii=False, indent=2), encoding="utf8")
    lines = ["# 实验199：真正频带微分熵（DE）", "", "DE为每窗口、每通道、每频带对带通滤波信号方差作高斯近似微分熵：0.5·ln(2πeσ²)。所有模型使用严格subject-wise OOF；未用验证标签选择特征、权重或阈值。plus_pe/graph采用已存在的固定OOF概率。", "", "|条件|AUC|F1|", "|---|---:|---:|"] + [f"|{r['condition']}|{r['auc']:.6f}|{r['f1']:.6f}|" for r in rows] + ["", f"停止线（+0.005）：相对最佳非固定融合AUC差={baseline_auc-best:+.6f}，通过={decision}。", f"耗时：{provenance['elapsed_seconds']:.2f}秒。详细结果见metrics.json，OOF与公开预测见predictions/。"]
    (HERE / "实验报告.md").write_text("\n".join(lines) + "\n", encoding="utf8"); print(pd.DataFrame(rows).to_string(index=False), flush=True)

if __name__ == "__main__": main()
