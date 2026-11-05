import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT / "src" / "166_HC31_inner_model_selection" / "run_experiment.py"
spec = importlib.util.spec_from_file_location("exp166", SOURCE)
exp166 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp166)
BASE_DIR = exp166.BASE_DIR

FIXED_THRESHOLD = exp166.FIXED_THRESHOLD
CANDIDATES = exp166.CANDIDATES


def load_data(config):
    x, ref = exp166.base.load_data(config)
    keep = ref.subject.to_numpy() == "HC23"
    x, ref = x[keep], ref.loc[keep].reset_index(drop=True)
    assert x.shape == (40, 720) and ref.subject.nunique() == 1 and ref.subject.iloc[0] == "HC23"
    return x, ref


def run_subject(x, ref):
    y = ref.y.to_numpy(dtype=int)
    keys = exp166.trial_keys(ref)
    unique = list(dict.fromkeys(keys))
    assert len(unique) == 8 and all(np.sum(keys == k) == 5 for k in unique)
    p = np.full(len(y), np.nan)
    folds = np.full(len(y), -1, dtype=int)
    fold_meta = []
    for fold, held in enumerate(unique):
        train, valid = keys != held, keys == held
        selected, inner = exp166.choose_candidate(x, y, keys, keys[train])
        method, direction = selected.split(":")
        model = exp166.make_model(method)
        model.fit(x[train], y[train])
        p[valid] = exp166.score(model, x[valid], method, direction)
        folds[valid] = fold
        fold_meta.append({"subject": "HC23", "fold": fold, "held_out_trial_key": str(held), "train_rows": int(train.sum()), "valid_rows": int(valid.sum()), "selected_candidate": selected, "inner_scores": inner})
        print(f"outer fold {fold + 1}/8: held={held}, selected={selected}", flush=True)
    assert np.isfinite(p).all() and np.array_equal(np.sort(folds), np.repeat(np.arange(8), 5))
    return y, keys, p, folds, fold_meta


def main():
    started = time.perf_counter()
    config = json.loads((HERE / "config.json").read_text(encoding="utf8"))
    base_config = json.loads((BASE_DIR / "config.json").read_text(encoding="utf8"))
    (HERE / "predictions").mkdir(exist_ok=True)
    (HERE / "input_metadata").mkdir(exist_ok=True)
    x, ref = load_data(base_config)
    with exp166.threadpool_limits(limits=1):
        y, keys, p, folds, fold_meta = run_subject(x, ref)
    ordered_keys = list(dict.fromkeys(keys))
    trial_p = np.asarray([p[keys == k].mean() for k in ordered_keys])
    trial_y = np.asarray([y[keys == k][0] for k in ordered_keys])
    pred = ref.copy()
    pred["fold"], pred["prob_positive"] = folds, p
    pred[["row_index", "subject", "group", "trial", "position", "y", "fold", "prob_positive"]].to_csv(HERE / "predictions/window_oof.csv", index=False)
    trial_out = pd.DataFrame({"trial_index": range(8), "trial_key": ordered_keys, "y": trial_y, "prob_positive": trial_p, "prediction_fixed05": (trial_p >= FIXED_THRESHOLD).astype(int), "window_count": 5})
    trial_out.to_csv(HERE / "predictions/trial_oof.csv", index=False)
    window_metrics = exp166.classification(y, p)
    trial_metrics = exp166.classification(trial_y, trial_p)
    metrics = {"subject": "HC23", "method": "inner-only model:direction selection", "window": window_metrics, "trial_mean": trial_metrics, "threshold": FIXED_THRESHOLD, "outer_validation_labels_used_for_selection": False, "candidate_order": list(CANDIDATES), "trial_aggregation": "mean of five window probabilities", "elapsed_seconds": time.perf_counter() - started}
    pd.DataFrame(fold_meta).to_csv(HERE / "input_metadata/folds.csv", index=False)
    inputs = [(BASE_DIR / base_config[k]).resolve() for k in ("spectral_cache", "temporal_cache", "pe_cache")]
    sources = [SOURCE, HERE / "run_experiment.py", HERE / "config.json", BASE_DIR / "run_experiment.py", BASE_DIR / "config.json"]
    provenance = {"inputs": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, "sources": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, "raw_shape": list(x.shape), "public_arrays_loaded": False, "subjects": ["HC23"], "rows": int(len(ref)), "normalization": base_config["normalization"], "threads": 1, "python": platform.python_version(), "sklearn": __import__("sklearn").__version__, "dataset_description": "D:/WORK/BCIC2026/EmoRec/数据/数据集说明文档.md缺失，未伪造哈希"}
    metrics["provenance"] = provenance
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    (HERE / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
    report = f"""# 实验169：HC23 inner-only模型选择\n\n## 真实结果\n\n|评估粒度|AUC|F1（固定阈值0.5）|\n|---|---:|---:|\n|40窗外层OOF|{window_metrics['auc']:.6f}|{window_metrics['f1']:.6f}|\n|8 trial五窗均值|{trial_metrics['auc']:.6f}|{trial_metrics['f1']:.6f}|\n\n## 冻结协议\n\n- 仅运行HC23；复用实验166的720维缓存、8外层trial/7内层trial model:direction选择函数。\n- 每个外层折验证1个完整trial（5窗），训练35窗；候选按内层AUC、内层F1、固定候选顺序选择，不使用外层验证标签。\n- 固定阈值0.5，无事后阈值或quota搜索；8个trial各5窗，fold分组无交叉。\n- 091的逐被试全40窗口z-score为transductive归一化；该结果是单被试内评估，不代表跨被试泛化。\n- 按要求不加载公开集。数据集说明文档路径缺失，已如实记录。\n\n## 产物与命令\n\n- 入口：`{HERE / 'run_experiment.py'}`；配置：`{HERE / 'config.json'}`\n- 预测：`predictions/window_oof.csv`、`predictions/trial_oof.csv`；折信息：`input_metadata/folds.csv`；指标：`metrics.json`。\n- 命令：`$env:OMP_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1; py -3.11 test/experiments/169_HC23_inner_model_selection/run_experiment.py`\n- 运行耗时：{metrics['elapsed_seconds']:.3f}秒；单进程、单线程。\n"""
    (HERE / "实验报告.md").write_text(report, encoding="utf8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    main()
