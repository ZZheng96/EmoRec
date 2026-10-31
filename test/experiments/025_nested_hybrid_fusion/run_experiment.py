import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT = EXPERIMENT_DIR.parents[2]
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
OOF_COLUMNS = ["subject_id", "health_group", "original_trial_id", "fold", "y_true", "prob_positive"]
PUBLIC_COLUMNS = ["user_id", "trial_id", "prob. of positive emotion"]


def score(y, p):
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= 0.5, zero_division=0))}


def subject_splits(subject_ids, labels, n_splits, seed):
    subjects = np.asarray(pd.unique(subject_ids))
    subject_labels = np.asarray([labels[np.flatnonzero(subject_ids == s)[0]] for s in subjects])
    by_subject = {}
    for fold, (_, val) in enumerate(StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(subjects, subject_labels)):
        for index in val:
            by_subject[subjects[index]] = fold
    return np.asarray([by_subject[s] for s in subject_ids], dtype=int)


def hgb_model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))


def lr_model(config, seed):
    return make_pipeline(StandardScaler(), LogisticRegression(**config["labram_lr"], random_state=seed))


def choose_weight(y, hgb, aux, candidates):
    rows = []
    for weight in candidates:
        s = score(y, weight * hgb + (1 - weight) * aux)
        rows.append({"hgb_weight": float(weight), "aux_weight": float(1 - weight), "auc": s["auc"], "f1": s["f1"]})
    return max(rows, key=lambda r: (r["auc"], r["f1"], -r["hgb_weight"])), rows


def load_and_align(config):
    spectral_path = ROOT / "test" / "experiments" / "001_spectral_logpower" / "cache" / "features.npz"
    labram_path = ROOT / "test" / "experiments" / "004_labram_frozen" / "cache" / "embeddings.npz"
    with np.load(spectral_path) as z:
        spectral = {k: z[k] for k in z.files}
    with np.load(labram_path) as z:
        labram = {k: z[k] for k in z.files}
    names = np.asarray(spectral["feature_names"])
    mask = np.char.startswith(names.astype(str), "log_bandpower_")
    if mask.sum() != 150:
        raise RuntimeError(f"expected 150 log_bandpower features, got {mask.sum()}")
    keep = spectral["subject_ids"] != config["excluded_subject"]
    expected = ["subject_ids", "y_train", "health_groups", "original_trial_ids"]
    for key in expected:
        if not np.array_equal(spectral[key][keep], labram[key]):
            raise RuntimeError(f"cache metadata mismatch: {key}")
    if not np.array_equal(spectral["public_user_ids"], labram["public_user_ids"]) or not np.array_equal(spectral["public_trial_ids"], labram["public_trial_ids"]):
        raise RuntimeError("public cache metadata mismatch")
    outer_ref = pd.read_csv(EXPERIMENT_DIR.parent / "002_subject_normalization" / "predictions" / "subject_zscore_exclude_hc43_oof.csv")
    if not np.array_equal(outer_ref["subject_id"], labram["subject_ids"]) or not np.array_equal(outer_ref["y_true"], labram["y_train"]):
        raise RuntimeError("002 metadata alignment mismatch")
    if not np.array_equal(outer_ref["fold"], labram["folds"]):
        raise RuntimeError("embedding folds do not match 002 outer folds")
    X_hgb = normalize_by_subject(spectral["X_train"][keep][:, mask], spectral["subject_ids"][keep], "zscore", 1e-12)
    X_hgb_public = normalize_by_subject(spectral["X_public"][:, mask], spectral["public_user_ids"], "zscore", 1e-12)
    X_labram_raw = labram["X_train"]
    X_labram_z = normalize_by_subject(X_labram_raw, labram["subject_ids"], "zscore", 1e-12)
    X_labram_public_raw = labram["X_public"]
    X_labram_public_z = normalize_by_subject(X_labram_public_raw, labram["public_user_ids"], "zscore", 1e-12)
    frame = outer_ref.copy()
    return {
        "X_hgb": X_hgb, "X_hgb_public": X_hgb_public,
        "labram": {"raw": X_labram_raw, "subject_zscore": X_labram_z},
        "labram_public": {"raw": X_labram_public_raw, "subject_zscore": X_labram_public_z},
        "frame": frame, "y": labram["y_train"], "outer": labram["folds"],
        "public_users": labram["public_user_ids"], "public_trials": labram["public_trial_ids"],
        "spectral_metadata_checked": True, "embedding_metadata_checked": True,
    }


def inner_oof(X_hgb, X_aux, y, subject_ids, inner_folds, config, seed):
    hgb_oof = np.full(len(y), np.nan)
    aux_oof = np.full(len(y), np.nan)
    fit_count = 0
    split_rows = []
    for inner in range(config["inner_folds"]):
        train = inner_folds != inner
        val = inner_folds == inner
        train_subjects = set(subject_ids[train])
        val_subjects = set(subject_ids[val])
        if train_subjects & val_subjects:
            raise RuntimeError("inner train/validation subjects overlap")
        hgb = hgb_model(config, seed + inner)
        aux = lr_model(config, seed + inner)
        hgb.fit(X_hgb[train], y[train]); aux.fit(X_aux[train], y[train])
        hgb_oof[val] = hgb.predict_proba(X_hgb[val])[:, 1]
        aux_oof[val] = aux.predict_proba(X_aux[val])[:, 1]
        fit_count += 2
        split_rows.append({"inner_fold": inner, "train_subjects": sorted(train_subjects), "validation_subjects": sorted(val_subjects), "train_rows": int(train.sum()), "validation_rows": int(val.sum())})
    if not np.isfinite(hgb_oof).all() or not np.isfinite(aux_oof).all():
        raise RuntimeError("incomplete strict inner OOF")
    return hgb_oof, aux_oof, fit_count, split_rows


def fit_outer(X_hgb, X_aux, y, train, val, config, seed):
    hgb = hgb_model(config, seed); aux = lr_model(config, seed)
    hgb.fit(X_hgb[train], y[train]); aux.fit(X_aux[train], y[train])
    return hgb.predict_proba(X_hgb[val])[:, 1], aux.predict_proba(X_aux[val])[:, 1], 2


def group_metrics(frame, p):
    y = frame.y_true.to_numpy()
    result = {"overall": score(y, p)}
    for group in ["HC", "DEP"]:
        mask = frame.health_group.to_numpy() == group
        result[group] = score(y[mask], p[mask]) if mask.any() and len(np.unique(y[mask])) == 2 else None
    values = [roc_auc_score(s.y_true, s._p) for _, s in frame.assign(_p=p).groupby("subject_id", sort=False) if s.y_true.nunique() == 2]
    result["subject_macro_auc"] = float(np.mean(values)) if values else None
    return result


def main():
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    PREDICTIONS_DIR.mkdir(exist_ok=True)
    data = load_and_align(config)
    frame, y, outer = data["frame"], data["y"], data["outer"]
    candidates = np.asarray(config["candidate_hgb_weights"], dtype=float)
    conditions = ["labram_raw", "labram_zscore", "labram_mean"]
    all_results, fold_rows, candidate_rows = {}, [], []
    for condition in conditions:
        aux_name = "raw" if condition == "labram_raw" else "subject_zscore"
        nested = np.full(len(y), np.nan); fixed = np.full(len(y), np.nan)
        aux_outer = np.full(len(y), np.nan); hgb_outer = np.full(len(y), np.nan)
        for fold in range(config["outer_folds"]):
            train, val = outer != fold, outer == fold
            inner = subject_splits(frame.subject_id.to_numpy()[train], y[train], config["inner_folds"], config["random_seed"] + fold)
            h_inner, z_inner, fits, split_rows = inner_oof(data["X_hgb"][train], data["labram"][aux_name][train], y[train], frame.subject_id.to_numpy()[train], inner, config, config["random_seed"] + fold * 10)
            if condition == "labram_mean":
                raw_inner = inner_oof(data["X_hgb"][train], data["labram"]["raw"][train], y[train], frame.subject_id.to_numpy()[train], inner, config, config["random_seed"] + fold * 10)[1]
                aux_inner = (raw_inner + z_inner) / 2
                fits += 3
            else:
                aux_inner = z_inner
            best, rows = choose_weight(y[train], h_inner, aux_inner, candidates)
            for row in rows: candidate_rows.append({"condition": condition, "outer_fold": fold, **row})
            h_val, a_val, outer_fits = fit_outer(data["X_hgb"], data["labram"]["raw" if condition == "labram_mean" else aux_name], y, train, val, config, config["random_seed"] + fold)
            if condition == "labram_mean":
                _, a_z, zfits = fit_outer(data["X_hgb"], data["labram"]["subject_zscore"], y, train, val, config, config["random_seed"] + fold)
                a_val = (a_val + a_z) / 2; outer_fits += zfits
            hgb_outer[val], aux_outer[val] = h_val, a_val
            nested[val] = best["hgb_weight"] * h_val + best["aux_weight"] * a_val
            fixed[val] = config["fixed_hgb_weight"] * h_val + (1 - config["fixed_hgb_weight"]) * a_val
            fold_rows.append({"condition": condition, "outer_fold": fold, "n_train_subjects": int(frame.loc[train, "subject_id"].nunique()), "n_validation_subjects": int(frame.loc[val, "subject_id"].nunique()), "selected_hgb_weight": best["hgb_weight"], "selected_aux_weight": best["aux_weight"], "inner_selection_auc": best["auc"], "inner_selection_f1": best["f1"], "inner_model_fit_count": fits, "outer_model_fit_count": outer_fits, "inner_splits": split_rows, "outer_train_subjects": sorted(set(frame.loc[train, "subject_id"])), "outer_validation_subjects": sorted(set(frame.loc[val, "subject_id"]))})
        all_results[condition] = {"nested": nested, "fixed": fixed, "hgb": hgb_outer, "aux": aux_outer}
        for name, prediction in [(f"nested_{condition}", nested), (f"fixed_09_01_{condition}", fixed)]:
            out = frame.copy(); out["prob_positive"] = prediction; out.to_csv(PREDICTIONS_DIR / f"{name}_oof.csv", index=False)
    public_rows = []
    full_inner = subject_splits(frame.subject_id.to_numpy(), y, config["inner_folds"], config["random_seed"] + 100)
    for condition in conditions:
        aux_name = "raw" if condition == "labram_raw" else "subject_zscore"
        h_inner, z_inner, _, _ = inner_oof(data["X_hgb"], data["labram"][aux_name], y, frame.subject_id.to_numpy(), full_inner, config, config["random_seed"] + 100)
        aux_inner = z_inner
        if condition == "labram_mean":
            raw_inner = inner_oof(data["X_hgb"], data["labram"]["raw"], y, frame.subject_id.to_numpy(), full_inner, config, config["random_seed"] + 100)[1]
            aux_inner = (raw_inner + z_inner) / 2
        best, _ = choose_weight(y, h_inner, aux_inner, candidates)
        hgb = hgb_model(config, config["random_seed"]); hgb.fit(data["X_hgb"], y)
        aux = lr_model(config, config["random_seed"]); aux.fit(data["labram"]["raw" if condition == "labram_mean" else aux_name], y)
        public_h = hgb.predict_proba(data["X_hgb_public"])[:, 1]
        public_a = aux.predict_proba(data["labram_public"]["raw" if condition == "labram_mean" else aux_name])[:, 1]
        if condition == "labram_mean":
            z = lr_model(config, config["random_seed"]); z.fit(data["labram"]["subject_zscore"], y)
            public_a = (public_a + z.predict_proba(data["labram_public"]["subject_zscore"])[:, 1]) / 2
        p = best["hgb_weight"] * public_h + best["aux_weight"] * public_a
        fixed = config["fixed_hgb_weight"] * public_h + (1 - config["fixed_hgb_weight"]) * public_a
        for kind, values in [("nested", p), ("fixed_09_01", fixed)]:
            out = pd.DataFrame({"user_id": data["public_users"], "trial_id": data["public_trials"], "prob. of positive emotion": values})
            validate_submission(out, [f"P_test{i}" for i in range(1, 11)]); out.to_csv(PREDICTIONS_DIR / f"{kind}_{condition}_public.csv", index=False)
        public_rows.append({"condition": condition, "selected_hgb_weight": best["hgb_weight"], "selected_aux_weight": best["aux_weight"], "inner_full_train_auc": best["auc"], "inner_full_train_f1": best["f1"]})
    runs = {c: {"nested": group_metrics(frame, v["nested"]), "fixed_09_01": group_metrics(frame, v["fixed"])} for c, v in all_results.items()}
    metrics = {"experiment_config": config, "source_validation": {"spectral_cache": "001_spectral_logpower/cache/features.npz", "embedding_cache": "004_labram_frozen/cache/embeddings.npz", "metadata_alignment_checked": True, "outer_folds_checked_against_002": True, "public_labels_read": False}, "protocol": {"strict_nested_refit": True, "old_protocol_status": "否定：旧版本使用016/004已生成OOF概率，模型可能见过外层验证被试，不纳入任何结果。", "outer_folds_from_experiment_002": True, "outer_validation_labels_used_for_selection": False, "inner_subject_3fold": True, "inner_models_refit_per_fold": True, "outer_models_refit_on_complete_outer_train": True, "candidate_grid": config["candidate_hgb_weights"], "selection": "max AUC, tie max F1", "hgb_features": "001前150维log_bandpower，经002 subject zscore", "labram_features": "004 200维，raw或subject zscore；mean为两套严格重训LR概率均值"}, "folds": fold_rows, "candidate_scores": candidate_rows, "public_selection": public_rows, "runs": runs, "model_selection_bias_limitations": "候选权重选择仍有内层选择乐观偏差；外层OOF只用于最终评估。"}
    (EXPERIMENT_DIR / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    rows = [{"condition": c, "run": k, "overall_auc": v[k]["overall"]["auc"], "overall_f1": v[k]["overall"]["f1"], "hc_auc": v[k]["HC"]["auc"], "dep_auc": v[k]["DEP"]["auc"], "subject_macro_auc": v[k]["subject_macro_auc"]} for c, v in runs.items() for k in v]
    pd.DataFrame(rows).to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    report = "# 实验025：严格嵌套HGB与LaBraM辅助融合\n\n旧版本协议已否定：016/004缓存OOF模型可能包含当前外层验证被试，旧结果不纳入结果。新协议从001与004特征缓存读取并严格核验subject_ids、y、trial、fold及公开键；每个外层fold内按被试3折，仅以内层训练被试重新拟合HGB和LaBraM LogisticRegression，生成真正内层OOF选择0.0至1.0、步长0.05的权重，再在完整外层训练集重训并预测外层验证。raw、subject_zscore及均值条件均不使用旧OOF概率；public模型用全训练集拟合且未读取public标签。\n\n真实新结果见metrics.json、summary.csv和predictions；其中metrics记录每个内层/外层模型拟合次数、训练/验证被试集合及元数据核验。\n"
    (EXPERIMENT_DIR / "实验报告.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
