import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from emo_rec.normalization import normalize_by_subject

HERE = Path(__file__).resolve().parent
OUT = HERE / "predictions"
spec = importlib.util.spec_from_file_location("protocol036", HERE.parent / "036_trial_consistency_features" / "run_experiment.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
# Reuse the frozen estimator/evaluation protocol without writing to 036.
base.OUT = OUT
KEYS = ("auc", "f1", "balanced_accuracy", "subject_auc_mean", "hc_auc", "dep_auc")


def candidates(config):
    return [{**m, "perturbation_seed": None} for m in config["methods"]] + [
        {"name": f"consistency_shuffle{seed}", "train_block": 5, "public_block": 2, "perturbation_seed": seed}
        for seed in config["perturbation_seeds"]]


def features(X, subjects, block_size, permutation_seed=None, stream=0):
    if block_size is None:
        return X.copy()
    if permutation_seed is None:
        return base.consistency_features(X, subjects, block_size)
    order = np.arange(len(X))
    rng = np.random.default_rng(np.random.SeedSequence([permutation_seed, stream]))
    for subject in np.unique(subjects):
        rows = np.flatnonzero(subjects == subject)
        order[rows] = rng.permutation(rows)
    shuffled = base.consistency_features(X[order], subjects[order], block_size)
    restored = np.empty_like(shuffled)
    restored[order] = shuffled
    return restored


def prepare(config):
    cache = base.load_cache((HERE / config["spectral_cache"]).resolve())
    keep = cache["subject_ids"] != config["excluded_subject"]
    subjects = cache["subject_ids"][keep]
    assert keep.sum() == config["expected_train_rows"]
    assert len(np.unique(subjects)) == config["expected_subjects"]
    assert len(cache["X_public"]) == config["expected_public_rows"]
    sx = normalize_by_subject(cache["X_train"][keep, :150], subjects, "zscore")
    sp = normalize_by_subject(cache["X_public"][:, :150], cache["public_user_ids"], "zscore")
    ref = pd.DataFrame({"row_id": np.arange(len(subjects)), "subject_id": subjects,
                        "health_group": cache["health_groups"][keep],
                        "original_trial_id": cache["original_trial_ids"][keep], "y_true": cache["y_train"][keep]})
    ref.attrs["public_user_ids"] = cache["public_user_ids"]
    ref.attrs["public_trial_ids"] = cache["public_trial_ids"]
    return cache, sx, sp, ref


def aggregate(values):
    return {"mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)), "values": list(values)}


def main():
    started = time.perf_counter()
    config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))
    OUT.mkdir(exist_ok=True)
    cache, sx, sp, reference = prepare(config)
    runs, rows = {}, []
    result = {"experiment_config": config, "runs": runs, "provenance": {
        "spectral_cache_sha256": base.sha256((HERE / config["spectral_cache"]).resolve()),
        "protocol036_sha256": base.sha256(Path(base.__file__)),
        "config_sha256": base.sha256(HERE / "config.json"),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}}
    with threadpool_limits(limits=1):
        for candidate in candidates(config):
            name = candidate["name"]
            X = features(sx, reference.subject_id.to_numpy(), candidate["train_block"], candidate["perturbation_seed"], 0)
            XP = features(sp, cache["public_user_ids"], candidate["public_block"], candidate["perturbation_seed"], 1)
            runs[name] = {"candidate": candidate, "repeats": []}
            for seed in config["random_seeds"]:
                ref = reference.copy()
                folds = base.make_subject_folds(ref.subject_id.to_numpy(), ref.health_group.to_numpy(), seed)
                base.assert_subject_isolation(ref.subject_id.to_numpy(), folds)
                ref["fold"] = folds
                ref["fold_seed"] = seed
                run = base.evaluate(f"{name}_seed{seed}", X, XP, ref.y_true.to_numpy(), ref, folds, config)
                run["fold_seed"] = seed
                runs[name]["repeats"].append(run)
                r = run["result"]
                row = {"run_name": name, "fold_seed": seed, "perturbation_seed": candidate["perturbation_seed"],
                       "train_block": candidate["train_block"], "public_block": candidate["public_block"],
                       "n_features": X.shape[1], **r["overall"], "subject_auc_mean": r["subject_auc_mean"],
                       "hc_auc": r["health_groups"]["HC"]["pooled"]["auc"], "dep_auc": r["health_groups"]["DEP"]["pooled"]["auc"]}
                rows.append(row)
                pd.DataFrame(rows).to_csv(HERE / "summary.csv", index=False)
                (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"{name} seed={seed} AUC={row['auc']:.6f} elapsed={time.perf_counter()-started:.1f}s", flush=True)
            own = [r for r in rows if r["run_name"] == name]
            runs[name]["aggregate"] = {k: aggregate([r[k] for r in own]) for k in KEYS}
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index="fold_seed", columns="run_name", values="auc")
    result["paired_auc_vs_control"] = {name: aggregate((pivot[name] - pivot.spectral_control).tolist()) for name in runs if name != "spectral_control"}
    drops = pivot.consistency_spectral - pivot[[f"consistency_shuffle{s}" for s in config["perturbation_seeds"]]].mean(axis=1)
    result["paired_order_drop"] = aggregate(drops.tolist())
    result["completed_runs"] = len(rows)
    result["elapsed_seconds"] = time.perf_counter() - started
    (HERE / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    report = "# 实验037：一致性特征重复折与行序稳健性复核\n\n"
    report += "复用001 spectral前150维；排除HC43后2760行、69被试，公开80行。各被试独立无标签z-score。特征仅接受数值矩阵、subject/user身份和预注册块大小/扰动seed，不使用y、health_group、original_trial_id或公开标签。health_group仅用于被试级折分层和分组评估。\n\n"
    report += "## 固定协议\n\n5个外层折seed为2027–2031，沿用032先排除HC43再分折；这与036的70被试先分折再排除不同，因此并非复现036单次2026折分。HGB完全沿用036，模型seed=2026+fold，全量重拟合seed=2026；不调阈值或权重，阈值恒0.5。每个候选/seed均独立完整训练5折及全量重拟合，不复用拟合结果。\n\n"
    report += "consistency_spectral为036原特征协议（训练块5、公开块2）；b5_b5与b2_b2为块大小敏感性；shuffle4101/4102/4103为原5/2协议的行序扰动，与全部5个外层seed交叉。每个subject/user内部独立打乱，训练/公开用不同随机流；统计后逆映射回原行，标签和subject_ids从不打乱。公开块5按5+3处理，不丢弃尾块。公开数据无标签，仅保存预测，不报告性能。\n\n"
    report += "## 真实结果（跨5个折seed均值±样本标准差，ddof=1）\n\n|方法|AUC|F1|BAcc|subject宏AUC|HC AUC|DEP AUC|\n|---|---:|---:|---:|---:|---:|---:|\n"
    for name, run in runs.items():
        report += "|" + name + "|" + "|".join(f"{run['aggregate'][k]['mean']:.6f}±{run['aggregate'][k]['std']:.6f}" for k in KEYS) + "|\n"
    report += "\n## 配对差值与限制\n\n"
    for name, a in result["paired_auc_vs_control"].items():
        report += f"- {name}相对spectral_control AUC差：{a['mean']:+.6f}±{a['std']:.6f}。\n"
    a = result["paired_order_drop"]
    report += f"\n原协议减去每个外层seed下3个扰动平均AUC：{a['mean']:+.6f}±{a['std']:.6f}。预注册实质性下降界限为0.01，不将相关的重复折当作独立样本做显著性检验。"
    report += "若行序扰动显著下降，则036只能算协议敏感探索，不能宣称稳健创新。"
    report += ("本次达到预注册实质性下降界限，036只能算协议敏感探索。" if a["mean"] >= config["material_auc_drop"] else "本次未达到预注册实质性下降界限，但这不是无行序依赖的证明。")
    report += "\n\n5/2与5/5的训练特征完全一致，其OOF应完全相同；公开块大小只能改变无标签公开预测，不能据此判断哪种公开协议更好。公开全量重拟合使用固定模型seed，因此同候选不同外层seed的公开重拟合预测相同是预期行为；另存各折公开预测供审计。所有重复使用同一69人，标准差只描述折分敏感性，不代表独立数据集泛化不确定性。\n\n"
    report += f"完成{len(rows)}个候选×折seed运行，共{len(rows)*6}次拟合；耗时{result['elapsed_seconds']:.1f}秒。每seed/每扰动的六项指标见summary.csv，逐被试/逐折诊断见metrics.json，完整OOF及公开预测见predictions/。未修改036、总进度或提交git。\n"
    (HERE / "实验报告.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
