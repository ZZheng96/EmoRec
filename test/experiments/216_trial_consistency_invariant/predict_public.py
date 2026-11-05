"""用实验216的窗口模型生成公开预测（公开侧trial统计不可得，退化为零统计）。

公开集每个 trial 只有 1 条 10 秒特征，无法像训练集那样在 trial 内取 5 个窗口，
因此公开侧 trial 均值取自身特征、trial std 与两项一致性置 0。
输出为 216 预声明的 0.5×trial_stats + 0.5×057 plus_pe 融合，仅为探索候选。
"""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("base216", HERE / "run_experiment.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
config = json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def main():
    cache = base.base.load_cache((HERE / config["spectral_cache"]).resolve())
    keep = cache["subject_ids"] != config["excluded_subject"]
    subjects = cache["subject_ids"][keep]
    y = cache["y_train"][keep]
    trial_ids = cache["original_trial_ids"][keep]
    X = base.base.normalize_by_subject(cache["X_train"][keep, :150], subjects, "zscore")
    XP = base.base.normalize_by_subject(cache["X_public"][:, :150], cache["public_user_ids"], "zscore")
    F = base.trial_features(X, subjects, trial_ids, config["train_windows_per_trial"])
    with threadpool_limits(limits=1):
        model = base.model(config, config["model_seed_base"])
        model.fit(F, y)
        stats = np.column_stack([XP, XP, np.zeros_like(XP), np.zeros(len(XP)), np.zeros(len(XP))])
        p_trial = model.predict_proba(stats)[:, 1]
    plus = pd.read_csv(ROOT / "test/experiments/057_multiscale_permutation_entropy/predictions/plus_pe_public.csv")
    keys = pd.DataFrame({"user_id": cache["public_user_ids"], "trial_id": cache["public_trial_ids"]})
    merged = keys.merge(plus, on=["user_id", "trial_id"], how="left", validate="one_to_one")
    assert merged["prob. of positive emotion"].notna().all() and len(merged) == 80
    p = config["fusion"]["trial_stats_weight"] * p_trial + config["fusion"]["plus_pe_weight"] * merged["prob. of positive emotion"].to_numpy()
    out = keys.copy()
    out["prob. of positive emotion"] = np.clip(p, 0.0, 1.0)
    out.to_csv(HERE / "predictions/fixed_trial_stats_plus_pe_public_approx.csv", index=False)
    pd.DataFrame({"user_id": keys.user_id, "trial_id": keys.trial_id,
                  "trial_stats": p_trial, "plus_pe": merged["prob. of positive emotion"].to_numpy()}).to_csv(
        HERE / "predictions/public_components_approx.csv", index=False)
    print(out.head().to_string(index=False), len(out), float(out.iloc[:, -1].min()), float(out.iloc[:, -1].max()))


if __name__ == "__main__":
    main()
