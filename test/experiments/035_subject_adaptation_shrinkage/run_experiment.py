import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "normalization_002", HERE.parent / "002_subject_normalization" / "run_experiment.py"
)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def fit_population(X):
    return X.mean(axis=0), X.std(axis=0, ddof=0)


def transform(X, subjects, population, shrinkage):
    if not 0 <= shrinkage <= 1:
        raise ValueError("lambda must be in [0, 1]")
    population_mean, population_std = population
    result = np.empty_like(X, dtype=np.float64)
    for subject in np.unique(subjects):
        mask = subjects == subject
        own = X[mask]
        mean = (1 - shrinkage) * own.mean(axis=0) + shrinkage * population_mean
        std = (1 - shrinkage) * own.std(axis=0, ddof=0) + shrinkage * population_std
        result[mask] = (own - mean) / (std + 1e-12)
    return result


def load_data(config):
    data = base.load_feature_cache((HERE / config["feature_cache"]).resolve())
    folds = base.make_subject_folds(data["subject_ids"], data["health_groups"], 2026)
    X, _ = base.select_log_bandpower(data["X_train"], data["feature_names"])
    public, _ = base.select_log_bandpower(data["X_public"], data["feature_names"])
    keep = data["subject_ids"] != "HC43"
    for key in base.TRAIN_DATA_KEYS:
        data[key] = data[key][keep]
    assert X[keep].shape == (2760, 150)
    assert public.shape == (80, 150)
    return data, X[keep], public, folds[keep]


def metrics(y, probability, subjects, health):
    rows = []
    for subject in np.unique(subjects):
        mask = subjects == subject
        rows.append({"subject_id": str(subject), "health_group": str(health[mask][0]),
                     **base.classification_metrics(y[mask], probability[mask])})
    groups = {}
    for group in ("HC", "DEP"):
        mask = health == group
        groups[group] = {
            **base.classification_metrics(y[mask], probability[mask]),
            "subject_macro_auc": float(np.mean([r["roc_auc"] for r in rows if r["health_group"] == group])),
        }
    return {"overall": base.classification_metrics(y, probability),
            "subject_macro_auc": float(np.mean([r["roc_auc"] for r in rows])),
            "health_groups": groups, "subjects": rows}


def write_report(results, summary):
    baseline = results["subject_zscore"]["overall"]["roc_auc"]
    better = [name for name in ("shrink_025", "shrink_050", "shrink_075")
              if results[name]["overall"]["roc_auc"] > baseline]
    conclusion = ("三个固定收缩系数均未提升整体OOF AUC，不支持收缩改善当前基线。" if not better else
                  "整体OOF AUC高于被试内z-score的方法：" + "、".join(better) + "；仅是单次固定五折结果，不能据此选lambda或声称稳定泛化收益。")
    lines = ["# 035 被试统计固定收缩实验", "", "## 协议", "",
             "复用001的150维log-bandpower缓存及002固定五折（先对70被试分层分折，再排除HC43）。69被试、2760条训练记录；公开10被试、80条记录。",
             "每折仅用训练样本拟合population mean/std（ddof=0）；每个训练/验证/公开被试独立用自己的无标签样本计算mean/std。",
             "固定混合 mean=(1−λ)subject_mean+λpopulation_mean，std=(1−λ)subject_std+λpopulation_std；输出(x−mean)/(std+1e−12)。变换不接收标签。",
             "HGB固定200轮、learning_rate=.03、15叶、min_samples_leaf=20、l2=1、early_stopping=False；seed=2026+fold；阈值0.5。无调参或lambda选择。",
             "公开预测为五个外层训练模型概率的算术平均，每个模型使用各自训练折population；无公开标签，不能计算公开AUC/F1。",
             "验证被试的完整无标签样本用于自身统计，属于transductive协议，并非单样本在线推理。", "", "## 全部预注册结果", "",
             "|方法|λ|OOF AUC|F1|被试宏AUC|折AUC均值±标准差|HC AUC/F1|DEP AUC/F1|", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        lines.append(f"|{row['method']}|{row['lambda']:.2f}|{row['oof_auc']:.6f}|{row['oof_f1']:.6f}|{row['subject_macro_auc']:.6f}|{row['fold_auc_mean']:.6f} ± {row['fold_auc_std']:.6f}|{row['hc_auc']:.6f}/{row['hc_f1']:.6f}|{row['dep_auc']:.6f}/{row['dep_f1']:.6f}|")
    lines += ["", "## 每折结果", "", "|方法|折(0起)|AUC|F1|被试宏AUC|", "|---|---:|---:|---:|---:|"]
    for name, result in results.items():
        for fold in result["folds"]:
            lines.append(f"|{name}|{fold['fold']}|{fold['overall']['roc_auc']:.6f}|{fold['overall']['binary_f1']:.6f}|{fold['subject_macro_auc']:.6f}|")
    lines += ["", "## 结论与限制", "", conclusion,
              "不把不同lambda平均成新方法；每个lambda报告其五折AUC均值与样本标准差，公开概率另按五折均值输出。",
              "单次五折的差异并非显著性检验；HC/DEP是诊断分组，分类目标始终为正负情绪。逐被试与逐折分组指标见metrics.json。",
              "每种方法保存2760行OOF及80行公开预测，公开文件另保留五个折概率便于核对均值。", ""]
    (HERE / "实验报告.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    data, X, public_X, folds = load_data(config)
    subjects, health, y = data["subject_ids"], data["health_groups"], data["y_train"]
    output = HERE / "predictions"
    output.mkdir(exist_ok=True)
    results, summary = {}, []
    with threadpool_limits(limits=2):
        for name, shrinkage in config["methods"].items():
            oof = np.full(len(y), np.nan)
            public_folds = []
            for fold in range(5):
                train, valid = folds != fold, folds == fold
                population = fit_population(X[train])
                model = HistGradientBoostingClassifier(**config["model"], random_state=2026 + fold)
                model.fit(transform(X[train], subjects[train], population, shrinkage), y[train])
                oof[valid] = model.predict_proba(transform(X[valid], subjects[valid], population, shrinkage))[:, 1]
                public_folds.append(model.predict_proba(transform(public_X, data["public_user_ids"], population, shrinkage))[:, 1])
                print(f"{name}: fold {fold + 1}/5", flush=True)
            public_folds = np.asarray(public_folds)
            assert np.isfinite(oof).all() and ((oof >= 0) & (oof <= 1)).all()
            assert np.isfinite(public_folds).all() and ((public_folds >= 0) & (public_folds <= 1)).all()
            pd.DataFrame({"subject_id": subjects, "health_group": health,
                          "original_trial_id": data["original_trial_ids"], "fold": folds,
                          "y_true": y, "prob_positive": oof}).to_csv(output / f"{name}_oof.csv", index=False)
            public = pd.DataFrame({"user_id": data["public_user_ids"], "trial_id": data["public_trial_ids"],
                                   "prob. of positive emotion": public_folds.mean(axis=0)})
            base.validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
            public.to_csv(output / f"{name}_public.csv", index=False)
            for fold in range(5):
                public[f"fold_{fold}_probability"] = public_folds[fold]
            public.to_csv(output / f"{name}_public_fold_probabilities.csv", index=False)
            result = metrics(y, oof, subjects, health)
            result["lambda"] = shrinkage
            result["folds"] = []
            for fold in range(5):
                mask = folds == fold
                result["folds"].append({"fold": fold, "n_samples": int(mask.sum()),
                                        **metrics(y[mask], oof[mask], subjects[mask], health[mask])})
            aucs = [f["overall"]["roc_auc"] for f in result["folds"]]
            result["fold_auc_mean"], result["fold_auc_std"] = float(np.mean(aucs)), float(np.std(aucs, ddof=1))
            results[name] = result
            summary.append({"method": name, "lambda": shrinkage, "oof_auc": result["overall"]["roc_auc"],
                            "oof_f1": result["overall"]["binary_f1"], "subject_macro_auc": result["subject_macro_auc"],
                            "fold_auc_mean": result["fold_auc_mean"], "fold_auc_std": result["fold_auc_std"],
                            "hc_auc": result["health_groups"]["HC"]["roc_auc"], "hc_f1": result["health_groups"]["HC"]["binary_f1"],
                            "dep_auc": result["health_groups"]["DEP"]["roc_auc"], "dep_f1": result["health_groups"]["DEP"]["binary_f1"]})
    (HERE / "metrics.json").write_text(json.dumps({"config": config, "runs": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    write_report(results, summary)
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
