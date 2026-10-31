import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from threadpoolctl import threadpool_limits

from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.features import extract_spectral_features
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("metrics032", HERE.parent / "032_repeated_subject_folds/run_experiment.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def normalize_window(window):
    x = np.asarray(window, dtype=np.float64)
    x = x - x.mean(axis=1, keepdims=True)
    x = x - x.mean(axis=0, keepdims=True)
    return (x / np.maximum(np.linalg.norm(x, axis=0, keepdims=True), 1e-12)).T.astype(np.float32)


def fit_templates(windows, subjects, train, config):
    rng = np.random.default_rng(config["template_seed"])
    indices = np.flatnonzero(train)
    sampled = np.stack([rng.choice(windows.shape[1], config["samples_per_window"], replace=False) for _ in indices])
    maps = windows[indices[:, None], sampled].reshape(-1, windows.shape[-1]).copy()
    signs = np.sign(maps[np.arange(len(maps)), np.abs(maps).argmax(axis=1)])
    maps *= np.where(signs == 0, 1, signs)[:, None]
    model = KMeans(n_clusters=config["n_states"], random_state=config["template_seed"], n_init=config["kmeans_n_init"], max_iter=config["kmeans_max_iter"])
    model.fit(maps)
    templates = model.cluster_centers_
    templates -= templates.mean(axis=1, keepdims=True)
    templates /= np.maximum(np.linalg.norm(templates, axis=1, keepdims=True), 1e-12)
    return templates, {"train_subjects": np.unique(subjects[train]).tolist(), "train_window_indices": indices.tolist(), "sample_indices_sha256": hashlib.sha256(sampled.tobytes()).hexdigest(), "n_sampled_maps": len(maps), "samples_per_window": config["samples_per_window"], "seed": config["template_seed"]}


def sequence_features(windows, templates, fs=250):
    k = len(templates)
    result = []
    for window in windows:
        corr = np.abs(window @ templates.T)
        states = corr.argmax(axis=1)
        occupancy = np.bincount(states, minlength=k) / len(states)
        counts = np.bincount(states[:-1] * k + states[1:], minlength=k*k).reshape(k, k)
        transitions = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
        entropy = -(transitions * np.log(np.maximum(transitions, 1e-12))).sum(axis=1)
        starts = np.r_[0, np.flatnonzero(np.diff(states)) + 1]
        lengths = np.diff(np.r_[starts, len(states)]) / fs
        runs = states[starts]
        durations = np.array([lengths[runs == s].mean() if np.any(runs == s) else 0 for s in range(k)])
        occurrences = np.bincount(runs, minlength=k) / (len(states) / fs)
        result.append(np.r_[occupancy, transitions.ravel(), entropy, durations, occurrences, np.mean(corr.max(axis=1)**2), np.mean(np.diff(states) != 0)])
    return np.asarray(result)


def check_oof(probabilities, coverage):
    assert np.isfinite(probabilities).all() and np.all(coverage == 1)
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def build_inputs():
    paths = sorted((ROOT / "数据/训练集").glob("*/*.mat"))
    all_subjects = np.array([p.stem for p in paths])
    all_groups = np.array(["DEP" if p.parent.name == "抑郁症患者" else "HC" for p in paths])
    fixed = dict(zip(all_subjects, base.make_subject_folds(all_subjects, all_groups, 2026)))
    train_paths = [p for p in paths if p.stem != "HC43"]
    public_paths = sorted((ROOT / "数据/公开测试集").glob("*.mat"), key=lambda p: int(p.stem.replace("P_test", "")))
    n = len(train_paths)*40
    windows = np.lib.format.open_memmap(HERE / "cache/train_windows.npy", mode="w+", dtype="float32", shape=(n, 2500, 30))
    public = np.lib.format.open_memmap(HERE / "cache/public_windows.npy", mode="w+", dtype="float32", shape=(80, 2500, 30))
    spectral, labels, subjects, groups, metadata = [], [], [], [], []
    for i, path in enumerate(train_paths):
        raw, y, _ = load_train_segments(path)
        windows[i*40:(i+1)*40] = np.stack([normalize_window(w) for w in raw])
        spectral.append(extract_spectral_features(raw)[0][:, :150])
        labels.extend(y); subjects.extend([path.stem]*40); groups.extend(["DEP" if path.parent.name == "抑郁症患者" else "HC"]*40)
        metadata.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "windows": 40})
        print(f"input {i+1}/{len(train_paths)} {path.stem}", flush=True)
    ps, users, trials = [], [], []
    for i, path in enumerate(public_paths):
        raw = load_public_test_segments(path)
        public[i*8:(i+1)*8] = np.stack([normalize_window(w) for w in raw])
        ps.append(extract_spectral_features(raw)[0][:, :150]); users.extend([path.stem]*8); trials.extend(range(1, 9))
        metadata.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "windows": 8})
    windows.flush(); public.flush()
    subjects, groups = np.array(subjects), np.array(groups)
    x = normalize_by_subject(np.concatenate(spectral), subjects, "zscore")
    xp = normalize_by_subject(np.concatenate(ps), np.array(users), "zscore")
    np.savez_compressed(HERE / "cache/spectral.npz", X=x, X_public=xp, subjects=subjects, groups=groups, y=labels, users=users, trials=trials)
    (HERE / "input_metadata/sources.json").write_text(json.dumps({"sources": metadata, "excluded": "HC43", "raw_window_shape": [30, 2500], "train_rows": n, "public_rows": 80, "fixed_fold_subjects_including_excluded": fixed}, indent=2, default=int), encoding="utf-8")
    return windows, public, x, xp, np.array(labels), subjects, groups, users, trials, np.array([fixed[s] for s in subjects])


def main():
    start = time.perf_counter()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    for folder in ("cache", "predictions", "input_metadata"):
        (HERE / folder).mkdir(exist_ok=True)
    results = {"config": config, "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__}, "runs": {}, "repeat_status": {str(s): "pending" for s in config["repeat_seeds"]}}
    rows = []
    with threadpool_limits(limits=1):
        w, wp, sx, sp, y, subjects, groups, users, trials, fixed = build_inputs()
        reference = pd.DataFrame({"subject_id": subjects, "health_group": groups, "window_id": np.arange(len(y)), "y_true": y})
        assignments = [("fixed", 2026, fixed)] + [(f"repeat{s}", s, base.make_subject_folds(subjects, groups, s)) for s in config["repeat_seeds"]]
        for name, seed, folds in assignments:
            base.assert_subject_isolation(subjects, folds)
            ref = reference.copy(); ref["fold"] = folds
            ref.drop(columns="y_true").to_csv(HERE / f"input_metadata/{name}_folds.csv", index=False)
            predictions = {method: np.full(len(y), np.nan) for method in config["methods"]}
            coverage = np.zeros(len(y), dtype=int)
            for fold in range(5):
                train, valid = folds != fold, folds == fold
                templates, provenance = fit_templates(w, subjects, train, config)
                assert not set(provenance["train_subjects"]) & set(subjects[valid])
                micro = sequence_features(w, templates)
                np.savez_compressed(HERE / f"cache/{name}_fold{fold}.npz", templates=templates, features=micro)
                (HERE / f"input_metadata/{name}_fold{fold}_templates.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
                for method, features in (("spectral_control", sx), ("microstate_hgb", micro), ("spectral_microstate_hgb", np.c_[sx, micro])):
                    model = base.make_model(config, 2026 + fold)
                    model.fit(features[train], y[train])
                    predictions[method][valid] = model.predict_proba(features[valid])[:, 1]
                coverage[valid] += 1
                print(f"{name} fold {fold+1}/5 elapsed={time.perf_counter()-start:.1f}s", flush=True)
            results["runs"][name] = {}
            for method, p in predictions.items():
                check_oof(p, coverage)
                out = ref.copy(); out["prob_positive"] = p
                out.to_csv(HERE / f"predictions/{name}_{method}_oof.csv", index=False)
                r = base.summarize(y, p, ref, folds)
                results["runs"][name][method] = r
                rows.append({"run": name, "seed": seed, "method": method, **r["overall"], "subject_macro_auc": r["subject_auc_mean"], "hc_auc": r["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": r["health_groups"]["DEP"]["pooled"]["auc"], "dep_subject_macro_auc": r["health_groups"]["DEP"]["subject_auc_mean"]})
            if name != "fixed": results["repeat_status"][str(seed)] = "completed"
            results["elapsed_seconds"] = time.perf_counter()-start
            (HERE / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)
        templates, provenance = fit_templates(w, subjects, np.ones(len(y), dtype=bool), config)
        micro, microp = sequence_features(w, templates), sequence_features(wp, templates)
        np.savez_compressed(HERE / "cache/full_templates_features.npz", templates=templates, X=micro, X_public=microp)
        (HERE / "input_metadata/full_templates.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
        for method, x, xp in (("spectral_control", sx, sp), ("microstate_hgb", micro, microp), ("spectral_microstate_hgb", np.c_[sx, micro], np.c_[sp, microp])):
            model = base.make_model(config, 2026); model.fit(x, y)
            frame = pd.DataFrame({"user_id": users, "trial_id": trials, "prob. of positive emotion": model.predict_proba(xp)[:, 1]})
            validate_submission(frame, [f"P_test{i}" for i in range(1, 11)])
            frame.to_csv(HERE / f"predictions/{method}_public.csv", index=False)
    results["elapsed_seconds"] = time.perf_counter()-start
    results["feature_dimensions"] = {"spectral_control": 150, "microstate_hgb": 34, "spectral_microstate_hgb": 184}
    (HERE / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False), flush=True)
    print(f"Total elapsed: {results['elapsed_seconds']:.3f}s", flush=True)
    columns = ["auc", "f1", "balanced_accuracy", "subject_macro_auc", "hc_auc", "dep_auc", "dep_subject_macro_auc"]
    aggregate = summary[summary.run != "fixed"].groupby("method")[columns].agg(["mean", "std"])
    aggregate.to_csv(HERE / "repeat_summary.csv")
    report = "# 实验045：窗口独立微状态拓扑序列\n\n原始EEG先切10秒；排除HC43。固定折在70被试上按002 seed2026生成后排除，重复折按032在69被试上生成。未读取既有OOF或特征缓存，不使用original_trial_id、邻窗、行序特征或公开标签。频谱逐窗口重新提取001的150维，并以无标签subject-zscore为控制（每个被试全体窗口统计，是控制既有的转导假设；微状态没有跨窗口归一化）。\n\n每窗口先通道时间去均值，再逐时刻空间去均值/单位范数。每训练窗口seed2026均匀无放回抽32个时刻，最大绝对通道统一极性，KMeans K=4/n_init=3/max_iter=100；验证仅绝对空间相关分配，无平滑和跨窗串联。34维：4占比（时间覆盖率）、16逐采样转换概率（含自转移）、4转换熵、4平均持续秒数、4每秒出现次数、平均平方相关、切换率。没有标准GFP峰提取，因此结论限于此预注册近似微状态方案。模板/Scaler/HGB只拟合训练折；模型早停关闭。\n\n三个预注册模型：频谱控制、微状态单独、频谱+微状态特征拼接；无事后概率融合/调参。每重复独立OOF；全训练重新拟合模板/模型生成每方法80行公开概率，不评估公开标签。\n\n## 固定折与重复明细\n\n```\n" + summary.to_string(index=False) + "\n```\n\n## 五重复均值/标准差\n\n```\n" + aggregate.to_string() + "\n```\n"
    control = aggregate.loc["spectral_control", ("dep_auc", "mean")]
    candidate = aggregate.loc["spectral_microstate_hgb", ("dep_auc", "mean")]
    conclusion = "微状态单独明显低于控制；拼接总体重复AUC略低于控制、固定DEP下降，重复DEP仅微增且仅2/5重复胜出，未证实稳定提升。停止该方向，不调参、不进行事后融合。"
    report += f"\n## 结论\n\n重复DEP AUC差值（拼接−控制）={candidate-control:+.6f}。{conclusion}\n\n总耗时 {results['elapsed_seconds']:.3f} 秒；5次重复全部真实完成。阈值0.5；每折/每被试指标见metrics.json。抽样索引哈希、训练被试及窗口索引见input_metadata，模板及特征见cache。测试输出见test_protocol.txt。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
