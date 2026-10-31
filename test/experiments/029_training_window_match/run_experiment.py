import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
OUT = HERE / "predictions"


def load_data(config):
    caches = []
    for key in ("spectral_cache", "temporal_cache"):
        with np.load(HERE / config[key], allow_pickle=False) as cache:
            caches.append({k: cache[k] for k in cache.files})
    spectral, temporal = caches
    for key in ("subject_ids", "health_groups", "original_trial_ids", "y_train", "public_user_ids", "public_trial_ids"):
        np.testing.assert_array_equal(spectral[key], temporal[key])
    keep = spectral["subject_ids"] != config["excluded_subject"]
    subjects = spectral["subject_ids"][keep]
    ref = pd.DataFrame({"subject_id": subjects, "health_group": spectral["health_groups"][keep],
                        "original_trial_id": spectral["original_trial_ids"][keep], "y_true": spectral["y_train"][keep]})
    # Read only metadata/folds, never existing OOF probabilities.
    outer = pd.read_csv(HERE / config["outer_reference"], usecols=list(ref.columns) + ["fold"])
    for key in ref.columns:
        np.testing.assert_array_equal(ref[key], outer[key])
    ref["fold"] = outer["fold"]
    ref["cache_row"] = np.flatnonzero(keep)
    ref["subject_window_index"] = ref.groupby("subject_id", sort=False).cumcount()
    assert len(ref) == 2760 and ref.subject_id.nunique() == 69
    assert set(ref.fold) == set(range(5))
    assert ref.groupby("subject_id").fold.nunique().eq(1).all()
    assert ref.groupby("subject_id").size().eq(40).all()
    assert temporal["X_train"].shape[1] == 450
    features = {}
    for name, cache, width in (("spectral", spectral, 150), ("temporal", temporal, 450)):
        features[name] = (normalize_by_subject(cache["X_train"][keep, :width], subjects, "zscore"),
                          normalize_by_subject(cache["X_public"][:, :width], cache["public_user_ids"], "zscore"))
    public = pd.DataFrame({"user_id": spectral["public_user_ids"], "trial_id": spectral["public_trial_ids"]})
    return ref, features, public


def training_blocks(subjects, train_mask, condition):
    if condition not in ("all_40", "one_per_trial_8", "one_per_trial_mean5"):
        raise ValueError(condition)
    blocks = []
    for subject in pd.unique(subjects[train_mask]):
        rows = np.flatnonzero(subjects == subject)
        assert len(rows) == 40 and train_mask[rows].all()
        if condition == "all_40":
            blocks.extend(rows.reshape(-1, 1))
        elif condition == "one_per_trial_8":
            blocks.extend(rows[::5].reshape(-1, 1))
        else:
            blocks.extend(rows.reshape(8, 5))
    return np.asarray(blocks)


def model(config, seed):
    return make_pipeline(StandardScaler(), HistGradientBoostingClassifier(**config["hgb"], random_state=seed))


def predict(X, XP, y, blocks, config, seed):
    fitted = model(config, seed)
    fitted.fit(X[blocks].mean(axis=1), y[blocks[:, 0]])
    return fitted.predict_proba(XP)[:, 1]


def metrics(y, p):
    tn, fp, fn, tp = confusion_matrix(y, p >= .5, labels=[0, 1]).ravel()
    return {"auc": float(roc_auc_score(y, p)), "f1": float(f1_score(y, p >= .5)),
            "balanced_accuracy": float(balanced_accuracy_score(y, p >= .5)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def main():
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    ref, features, public = load_data(config)
    subjects, y, folds = ref.subject_id.to_numpy(), ref.y_true.to_numpy(), ref.fold.to_numpy()
    OUT.mkdir(exist_ok=True)
    runs, summary, audit = {}, [], []
    with threadpool_limits(limits=config["threads"]):
        for condition in config["conditions"]:
            probabilities = {name: np.full(len(y), np.nan) for name in features}
            public_probabilities = {}
            visits = np.zeros(len(y), dtype=int)
            for fold in range(5):
                train, valid = folds != fold, folds == fold
                blocks = training_blocks(subjects, train, condition)
                assert not set(subjects[blocks.ravel()]) & set(subjects[valid])
                for name, (X, _) in features.items():
                    probabilities[name][valid] = predict(X, X[valid], y, blocks, config, config["random_seed"] + fold)
                visits[valid] += 1
                audit.append({"condition": condition, "fold": fold, "train_rows": len(blocks),
                              "train_subjects": len(np.unique(subjects[train])), "validation_rows": int(valid.sum()),
                              "validation_subjects": len(np.unique(subjects[valid]))})
                print(f"{condition}: fold {fold + 1}/5 complete", flush=True)
            assert np.all(visits == 1)
            blocks = training_blocks(subjects, np.ones(len(y), dtype=bool), condition)
            for name, (X, XP) in features.items():
                public_probabilities[name] = predict(X, XP, y, blocks, config, config["random_seed"])
            probabilities["fusion"] = .5 * probabilities["spectral"] + .5 * probabilities["temporal"]
            public_probabilities["fusion"] = .5 * public_probabilities["spectral"] + .5 * public_probabilities["temporal"]
            for name, p in probabilities.items():
                assert np.isfinite(p).all()
                run = f"{condition}_{name}"
                oof = ref.copy(); oof["prob_positive"] = p
                oof.to_csv(OUT / f"{run}_oof.csv", index=False)
                submission = public.copy(); submission["prob. of positive emotion"] = public_probabilities[name]
                validate_submission(submission, [f"P_test{i}" for i in range(1, 11)])
                submission.to_csv(OUT / f"{run}_public.csv", index=False)
                overall = metrics(y, p)
                runs[run] = {"overall": overall,
                             "folds": [{"fold": f, **metrics(y[folds == f], p[folds == f])} for f in range(5)],
                             "health_groups": {g: metrics(y[ref.health_group == g], p[ref.health_group == g]) for g in ("HC", "DEP")},
                             "subjects": [{"subject_id": s, **metrics(y[subjects == s], p[subjects == s])} for s in pd.unique(subjects)]}
                baseline = runs[f"all_40_{name}"]["overall"]
                summary.append({"condition": condition, "model": name, **overall,
                                "delta_auc_vs_all_40": overall["auc"] - baseline["auc"],
                                "delta_f1_vs_all_40": overall["f1"] - baseline["f1"]})
    result = {"config": config, "runs": runs, "training_audit": audit,
              "protocol": {"subject_isolation": True, "validation_visits_per_condition": 1,
                           "existing_oof_probabilities_used": False, "label_based_selection": False,
                           "normalization_before_training_selection": True},
              "provenance": {key + "_sha256": hashlib.sha256((HERE / config[key]).read_bytes()).hexdigest()
                             for key in ("spectral_cache", "temporal_cache", "outer_reference")},
              "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__, "threads": 1}}
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(summary).to_csv(HERE / "summary.csv", index=False)
    report = "# 实验029：训练窗口数量匹配\n\n## 协议\n\n复用001频谱150维、009时序450维、002固定5个外层被试折；排除HC43，共69被试2760行。所有模型本次重训，不读取028概率，002仅读取元数据与fold。HGB与028一致（200轮、learning_rate=.03、15叶、min_samples_leaf=20、L2=1、不早停），单线程，seed=2026+fold；全训练重训seed=2026。频谱/时序融合固定0.5/0.5，分类阈值0.5，不调参或选权重。\n\n训练与验证先各按被试全部40个窗口无标签zscore；公开集按每被试8窗zscore。之后仅在外层训练被试中按缓存稳定行序选择：all_40保留40行；one_per_trial_8取0,5,…,35；one_per_trial_mean5将连续5段特征均值为一行，共8行，特征维度不变。均值条件标签取每组第一段，测试审计组内同标签，不用标签决定选择。均值发生于zscore之后；不重新用8行估计被试归一化，确保只比较训练样本表示。StandardScaler只拟合实际训练行。\n\n每条件每个外层验证10秒窗恰好预测一次，均在相同2760行上评估。公开集80行以全部69个训练被试按对应条件重训生成，不使用公开标签或反馈。mean5是训练50秒平滑表示与验证10秒表示不匹配的试验性对照，不能解释为纯样本量效应。\n\n## 实测结果\n\n|条件|模型|AUC|F1|ΔAUC vs all_40|ΔF1 vs all_40|\n|---|---|---:|---:|---:|---:|\n"
    report += "\n".join(f"|{r['condition']}|{r['model']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['delta_auc_vs_all_40']:+.6f}|{r['delta_f1_vs_all_40']:+.6f}|" for r in summary)
    report += "\n\n## 解释边界\n\n固定折重复实验仅作探索性比较，不据此声称无偏泛化收益；窗口相关性较强，不能把2760行视为独立被试。首次窗口抽样同时改变样本数量与时间位置，mean5同时引入平滑。结果不能单独证明公开集表现会改善。各折、HC/DEP、每被试指标与输入哈希见metrics.json，逐窗及公开预测见predictions。专项测试覆盖标签审计（每被试8行4neu/4pos，仅审计）、被试隔离、产物完整性、确定性和独立重新训练/指标复算。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
