import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
SOURCE = Path(__file__).resolve()
ROOT = next(p for p in SOURCE.parents if (p / "数据").is_dir())
HERE = ROOT / "test" / "experiments" / "195_HC31_inner_model_selection_audited"
BASE_DIR = ROOT / "test" / "experiments" / "091_within_subject_trial_oof"
_spec = importlib.util.spec_from_file_location("exp091", BASE_DIR / "run_experiment.py")
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

METHODS = ("linear_svc", "logistic_regression", "hgb")
DIRECTIONS = ("raw", "reverse")
CANDIDATES = tuple(f"{m}:{d}" for m in METHODS for d in DIRECTIONS)
FIXED_THRESHOLD = 0.5
RANDOM_STATE = 2026
AUDIT_TOLERANCE = 1e-12
MIN_SPAN = 1e-12
FALLBACK = "linear_svc:raw"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, dtype=float), -60, 60)))


def make_model(method):
    if method == "linear_svc":
        estimator = LinearSVC(C=1.0, dual="auto", max_iter=5000, random_state=RANDOM_STATE)
    elif method == "logistic_regression":
        estimator = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=RANDOM_STATE)
    else:
        estimator = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, random_state=RANDOM_STATE)
    return make_pipeline(StandardScaler(), estimator)


def score(model, x, method, direction):
    p = sigmoid(model.decision_function(x)) if method == "linear_svc" else model.predict_proba(x)[:, 1]
    return p if direction == "raw" else 1.0 - p


def classification(y, p):
    return {"auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None, "f1": float(f1_score(y, p >= FIXED_THRESHOLD, zero_division=0))}


def load_data(config):
    x, ref = base.load_data(config)
    keep = ref.subject.to_numpy() == "HC31"
    return x[keep], ref.loc[keep].reset_index(drop=True)


def trial_keys(ref):
    return np.asarray([f"{y}::{t}" for y, t in zip(ref.y, ref.trial)], dtype=object)


def evaluate_candidate(x, y, keys, train_keys, candidate):
    method, direction = candidate.split(":")
    unique = list(dict.fromkeys(train_keys))
    inner_p, inner_y = [], []
    local_audit = []
    try:
        for held in unique:
            tr = np.isin(keys, unique) & (keys != held)
            va = keys == held
            model = make_model(method)
            model.fit(x[tr], y[tr])
            values = np.asarray(score(model, x[va], method, direction), dtype=float)
            train_prior = float(y[tr].mean())
            prior_target = train_prior if direction == "raw" else 1.0 - train_prior
            finite = bool(np.isfinite(values).all())
            matches_prior = finite and bool(np.all(np.isclose(values, prior_target, atol=AUDIT_TOLERANCE, rtol=0.0)))
            local_audit.append({"held_inner_trial": str(held), "train_rows": int(tr.sum()), "valid_rows": int(va.sum()), "train_positive_prior": train_prior, "score_min": float(np.min(values)) if len(values) else None, "score_max": float(np.max(values)) if len(values) else None, "score_std": float(np.std(values)) if len(values) else None, "score_unique": int(len(np.unique(np.round(values, 12)))) if len(values) else 0, "matches_training_prior": matches_prior, "finite": finite})
            inner_p.extend(values)
            inner_y.extend(y[va])
    except Exception as exc:
        return None, {"candidate": candidate, "error": repr(exc), "valid_for_selection": False, "invalid_reasons": ["fit_or_score_error"], "local_audit": local_audit}
    p = np.asarray(inner_p, dtype=float)
    yy = np.asarray(inner_y, dtype=int)
    reasons = []
    if not np.isfinite(p).all():
        reasons.append("nonfinite_prediction")
    if len(np.unique(yy)) < 2:
        reasons.append("single_class_inner_labels")
    if len(p) == 0 or float(np.ptp(p)) <= MIN_SPAN:
        reasons.append("constant_or_zero_span")
    if local_audit and all(item["matches_training_prior"] for item in local_audit):
        reasons.append("training_prior_degenerate")
    metrics = classification(yy, p) if not reasons or "single_class_inner_labels" not in reasons else {"auc": None, "f1": 0.0}
    audit = {"candidate": candidate, "method": method, "direction": direction, "inner_auc": metrics["auc"], "inner_f1": metrics["f1"], "inner_score_min": float(np.min(p)), "inner_score_max": float(np.max(p)), "inner_score_std": float(np.std(p)), "inner_score_unique": int(len(np.unique(np.round(p, 12)))), "inner_train_positive_prior_mean": float(np.mean([item["train_positive_prior"] for item in local_audit])), "valid_for_selection": not reasons, "invalid_reasons": reasons, "local_audit": local_audit}
    return (metrics, audit)


def choose_candidate(x, y, keys, train_keys, outer_fold, audit_rows):
    results = {}
    for candidate in CANDIDATES:
        result = evaluate_candidate(x, y, keys, train_keys, candidate)
        metrics, audit = result
        audit_rows.append({"outer_fold": outer_fold, **{k: v for k, v in audit.items() if k != "local_audit"}})
        results[candidate] = audit
    valid = [c for c in CANDIDATES if results[c]["valid_for_selection"]]
    if valid:
        selected = max(valid, key=lambda c: (results[c]["inner_auc"] if results[c]["inner_auc"] is not None else -np.inf, results[c]["inner_f1"], -CANDIDATES.index(c)))
        return selected, results, "selected_valid_candidate"
    return FALLBACK, results, "all_candidates_invalid_fixed_fallback"


def subject_run(x, ref):
    y, keys = ref.y.to_numpy(dtype=int), trial_keys(ref)
    unique = list(dict.fromkeys(keys))
    assert len(unique) == 8 and all(np.sum(keys == k) == 5 for k in unique)
    p = np.full(len(y), np.nan)
    folds = np.full(len(y), -1, dtype=int)
    fold_meta, audit_rows, invalid_cases = [], [], []
    for fold, held in enumerate(unique):
        train, valid = keys != held, keys == held
        selected, inner, selection_status = choose_candidate(x, y, keys, keys[train], fold, audit_rows)
        method, direction = selected.split(":")
        model = make_model(method)
        model.fit(x[train], y[train])
        values = np.asarray(score(model, x[valid], method, direction), dtype=float)
        fallback_degenerate = bool((not np.isfinite(values).all()) or float(np.ptp(values)) <= MIN_SPAN)
        p[valid] = values
        folds[valid] = fold
        if selection_status != "selected_valid_candidate" or fallback_degenerate:
            invalid_cases.append({"outer_fold": fold, "held_out_trial_key": str(held), "selection_status": selection_status, "selected_candidate": selected, "fallback_output_degenerate": fallback_degenerate})
        fold_meta.append({"subject": "HC31", "fold": fold, "held_out_trial_key": str(held), "train_rows": int(train.sum()), "valid_rows": int(valid.sum()), "train_positive_prior": float(y[train].mean()), "selected_candidate": selected, "selection_status": selection_status, "fallback_output_degenerate": fallback_degenerate, "valid_candidate_count": int(sum(item["valid_for_selection"] for item in inner.values())), "invalid_candidate_count": int(sum(not item["valid_for_selection"] for item in inner.values())), "outer_validation_labels_used_for_selection": False})
        print(f"[195/8] outer fold {fold + 1}/8 held={held} selected={selected} status={selection_status}", flush=True)
    assert np.isfinite(p).all() and np.array_equal(np.sort(folds), np.repeat(np.arange(8), 5))
    return p, folds, fold_meta, audit_rows, invalid_cases


def main():
    started = time.perf_counter()
    config = json.loads((BASE_DIR / "config.json").read_text(encoding="utf8"))
    config.update({"experiment_id": HERE.name, "audit_tolerance": AUDIT_TOLERANCE, "min_span": MIN_SPAN, "fallback": FALLBACK})
    (HERE / "predictions").mkdir(exist_ok=True)
    (HERE / "input_metadata").mkdir(exist_ok=True)
    (HERE / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf8")
    x, ref = load_data(config)
    y, keys = ref.y.to_numpy(dtype=int), trial_keys(ref)
    with threadpool_limits(limits=1):
        p, folds, fold_meta, audit_rows, invalid_cases = subject_run(x, ref)
    ordered_keys = list(dict.fromkeys(keys))
    trial_p = np.asarray([p[keys == k].mean() for k in ordered_keys])
    trial_y = np.asarray([y[keys == k][0] for k in ordered_keys])
    pred = ref.copy()
    pred["fold"], pred["prob_positive"] = folds, p
    pred["prediction_fixed05"] = (p >= FIXED_THRESHOLD).astype(int)
    pred[["row_index", "subject", "group", "trial", "position", "y", "fold", "prob_positive", "prediction_fixed05"]].to_csv(HERE / "predictions/window_oof.csv", index=False)
    trial_out = pd.DataFrame({"trial_index": range(8), "trial_key": ordered_keys, "y": trial_y, "prob_positive": trial_p, "prediction_fixed05": (trial_p >= FIXED_THRESHOLD).astype(int), "window_count": 5})
    trial_out.to_csv(HERE / "predictions/trial_oof.csv", index=False)
    pd.DataFrame(fold_meta).to_csv(HERE / "input_metadata/folds.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(HERE / "input_metadata/candidate_audit.csv", index=False)
    (HERE / "input_metadata/invalid_cases.json").write_text(json.dumps(invalid_cases, ensure_ascii=False, indent=2), encoding="utf8")
    inputs = [(BASE_DIR / config[k]).resolve() for k in ("spectral_cache", "temporal_cache", "pe_cache")]
    sources = [SOURCE, BASE_DIR / "run_experiment.py", ROOT / "数据" / "数据集说明文档.md"]
    provenance = {"inputs": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}, "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources}, "raw_shape": list(x.shape), "public_arrays_loaded": False, "subjects": ["HC31"], "rows": int(len(ref)), "normalization": config["normalization"], "threads": 1, "python": platform.python_version(), "sklearn": __import__("sklearn").__version__, "outer_validation_labels_used_for_selection": False, "dataset_description_available": True}
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    window_metrics, trial_metrics = classification(y, p), classification(trial_y, trial_p)
    invalid_output = bool(invalid_cases)
    metrics = {"subject": "HC31", "method": "inner-only model:direction selection with candidate prior-degeneracy audit", "window": window_metrics, "trial_mean": trial_metrics, "threshold": FIXED_THRESHOLD, "outer_validation_labels_used_for_selection": False, "candidate_order": list(CANDIDATES), "trial_aggregation": "mean of five window probabilities", "audit_tolerance": AUDIT_TOLERANCE, "min_span": MIN_SPAN, "fallback": FALLBACK, "fallback_or_invalid_fold_count": len(invalid_cases), "valid_result": not invalid_output, "result_interpretation": "invalid_as_effective_model_result_if_any_degeneracy_or_fallback_case", "elapsed_seconds": time.perf_counter() - started, "provenance": provenance}
    (HERE / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
    report = f'''# 实验195：HC31先验退化审计的inner-only模型选择\n\n## 当前结果\n\n| 评估粒度 | AUC | F1（固定阈值0.5） |\n|---|---:|---:|\n| 40窗外层OOF | {window_metrics["auc"]:.6f} | {window_metrics["f1"]:.6f} |\n| 8 trial五窗均值 | {trial_metrics["auc"]:.6f} | {trial_metrics["f1"]:.6f} |\n\n## 有效性结论\n\n- 本实验在候选选择阶段排除非有限、常数/零跨度、以及逐inner折完全等于训练先验（或反向先验）的候选。\n- 退化候选不计入有效模型达标结论。若发生固定fallback或fallback输出退化，以上AUC/F1仍是真实OOF指标，但本实验标记为无效开发结果。\n- 当前无效/fallback折数：**{len(invalid_cases)}**；有效模型结果：**{not invalid_output}**。\n\n## 协议\n\n- HC31，40窗口，8个物理trial，每trial5窗口；外层留一完整物理trial，外层训练35窗口。\n- 外层训练集内7折trial留一，仅使用inner OOF选择候选；外层验证标签不参与选择。\n- 候选为LinearSVC、LogisticRegression、HistGradientBoosting及raw/reverse方向；阈值固定0.5，无quota或事后阈值搜索。\n- 先验审计容差={AUDIT_TOLERANCE}，最小概率跨度={MIN_SPAN}；全候选无效时预先固定fallback为`linear_svc:raw`。\n- trial概率为五个窗口概率的均值。输入沿用091缓存和逐被试无标签z-score；这是同一被试的开发型OOF，不是跨被试泛化成绩。\n- 当前仓库实际存在并读取的数据集说明文件：`数据/数据集说明文档.md`。\n\n## 产物\n\n- `predictions/window_oof.csv`、`predictions/trial_oof.csv`\n- `input_metadata/folds.csv`、`candidate_audit.csv`、`invalid_cases.json`、`provenance.json`\n- `metrics.json`\n'''
    (HERE / "实验报告.md").write_text(report, encoding="utf8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
