import hashlib
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / "config.json").read_text(encoding="utf8"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = (HERE / CFG["stft_cache"]).resolve()
BASELINE = (HERE / CFG["baseline_public"]).resolve()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


class TinySpectrogramCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(30, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2), nn.Conv2d(16, 16, 3, padding=1, groups=16),
            nn.Conv2d(16, 32, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(32, 1)

    def forward(self, x):
        return self.head(self.features(x).flatten(1)).squeeze(1)


def fit_scaler(x):
    mean = x.mean(axis=(0, 2, 3), keepdims=True).astype("float32")
    std = np.maximum(x.std(axis=(0, 2, 3), keepdims=True), 1e-6).astype("float32")
    return mean, std


def train_full(x, y, seed):
    seed_all(seed)
    model = TinySpectrogramCNN().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG["learning_rate"], weight_decay=CFG["weight_decay"])
    xt = torch.from_numpy(x).to(DEVICE)
    yt = torch.from_numpy(y.astype("float32")).to(DEVICE)
    rng = np.random.default_rng(seed)
    losses = []
    for epoch in range(CFG["epochs"]):
        order = rng.permutation(len(y))
        total = 0.0
        model.train()
        for start in range(0, len(order), CFG["batch_size"]):
            ix = order[start:start + CFG["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(model(xt[ix]), yt[ix])
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * len(ix)
        value = total / len(y)
        losses.append(value)
        print(f"seed={seed} epoch={epoch + 1}/{CFG['epochs']} loss={value:.6f}", flush=True)
    return model, losses


def predict(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.from_numpy(x).to(DEVICE))).cpu().numpy()


def validate_submission(frame):
    expected = CFG["submission_columns"]
    if list(frame.columns) != expected:
        raise RuntimeError(f"submission columns mismatch: {list(frame.columns)}")
    if len(frame) != CFG["expected_public_rows"]:
        raise RuntimeError(f"expected {CFG['expected_public_rows']} rows, got {len(frame)}")
    users = [f"P_test{i}" for i in range(1, 11)]
    if frame.user_id.tolist() != sum(([u] * 8 for u in users), []):
        raise RuntimeError("public user/trial order mismatch")
    if frame.trial_id.tolist() != list(range(1, 9)) * 10:
        raise RuntimeError("public trial order mismatch")
    p = frame[expected[2]].to_numpy(float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise RuntimeError("non-finite or out-of-range probability")


def main():
    started = time.perf_counter()
    (HERE / "predictions").mkdir(exist_ok=True)
    (HERE / "input_metadata").mkdir(exist_ok=True)
    source_config = json.loads((HERE / CFG['source_experiment'] / 'config.json').read_text(encoding='utf8'))
    for key in ('seeds', 'excluded_subject', 'epochs', 'batch_size', 'learning_rate', 'weight_decay'):
        if CFG[key] != source_config[key]:
            raise RuntimeError(f'experiment212 configuration mismatch: {key}')
    if CFG['residual_weight'] != 0.2:
        raise RuntimeError('residual weight must remain fixed at 0.2')
    input_keys = ('X_train', 'y_train', 'subject_ids', 'X_public', 'public_user_ids', 'public_trial_ids')
    with np.load(CACHE, allow_pickle=False) as z:
        data = {k: z[k] for k in input_keys}
    baseline = pd.read_csv(BASELINE)
    validate_submission(baseline)
    if not (np.array_equal(data['public_user_ids'], baseline.user_id.to_numpy())
            and np.array_equal(data['public_trial_ids'], baseline.trial_id.to_numpy())):
        raise RuntimeError('STFT public keys and baseline row order mismatch')
    keep = data["subject_ids"] != CFG["excluded_subject"]
    x = data["X_train"][keep].astype("float32")
    y = data["y_train"][keep].astype("int64")
    public_x = data["X_public"].astype("float32")
    if len(y) != CFG["expected_train_rows"] or len(public_x) != CFG["expected_public_rows"]:
        raise RuntimeError("unexpected cache row count")
    if not np.isfinite(x).all() or not np.isfinite(public_x).all():
        raise RuntimeError("non-finite STFT tensor")
    print(f"213 starting device={DEVICE} train={x.shape} public={public_x.shape} seeds={CFG['seeds']}", flush=True)

    seed_public = {}
    audits = []
    mean, std = fit_scaler(x)
    tx = ((x - mean) / std).astype("float32")
    px = ((public_x - mean) / std).astype("float32")
    for seed in CFG["seeds"]:
        model, losses = train_full(tx, y, seed)
        pred = predict(model, px)
        if not np.isfinite(pred).all() or ((pred < 0) | (pred > 1)).any():
            raise RuntimeError(f"seed {seed} produced invalid probabilities")
        seed_public[str(seed)] = pred
        audits.append({"seed": seed, "epochs": CFG["epochs"], "losses": losses, "train_rows": len(y), "device": str(DEVICE)})
        print(f"seed={seed} full-training complete", flush=True)

    baseline_p = baseline[CFG["submission_columns"][2]].to_numpy(float)
    avg_cnn = np.mean(np.stack(list(seed_public.values())), axis=0)
    residual_p = (1.0 - CFG["residual_weight"]) * baseline_p + CFG["residual_weight"] * avg_cnn
    residual = baseline.copy()
    residual[CFG["submission_columns"][2]] = residual_p
    validate_submission(residual)
    residual_path = HERE / "predictions" / "repeated_seed_mean_residual_0.2_public.csv"
    baseline_path = HERE / "predictions" / "baseline_057_plus_pe_public.csv"
    single = baseline.copy()
    single[CFG['submission_columns'][2]] = 0.8 * baseline_p + 0.2 * seed_public['210']
    validate_submission(single)
    single_path = HERE / 'predictions' / 'seed_210_residual_0.2_public.csv'
    outputs = {residual_path: residual, baseline_path: baseline, single_path: single}
    output_hashes = {}
    for path, frame in outputs.items():
        text = frame.to_csv(index=False)
        output_hashes[str(path.relative_to(HERE))] = hashlib.sha256(text.encode('utf8')).hexdigest()
        path.write_bytes(text.encode('utf8'))

    sources = [CACHE, BASELINE, HERE / "config.json", HERE / "run_public.py", ROOT / "数据" / "数据集说明文档.md",
               HERE / CFG['source_experiment'] / 'config.json',
               HERE / CFG['source_experiment'] / 'run_experiment.py',
               ROOT / 'test/experiments/047_time_frequency_cnn/run_experiment.py',
               ROOT / 'test/experiments/057_multiscale_permutation_entropy/run_experiment.py']
    provenance = {
        "inputs": {str(p.relative_to(ROOT)): sha(p) for p in sources},
        "device": str(DEVICE),
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "train_rows": int(len(y)), "public_rows": int(len(public_x)),
        "excluded_subject": CFG["excluded_subject"], "seeds": CFG["seeds"],
        "full_training_only": True, "public_labels_used": False,
        "residual_weight": CFG['residual_weight'], "baseline_weight": 0.8,
        "train_subjects": int(np.unique(data['subject_ids'][keep]).size),
        "scaler_fit": "all eligible training rows only; no public rows",
        "cache_arrays_read": list(input_keys), "outputs_sha256": output_hashes,
        "public_cache_baseline_keys_equal": True,
        "model_selection": "fixed experiment212 protocol; no public-label selection",
        "baseline_training": "reused frozen057 plus_pe; source script fits all eligible training rows",
        "validation": "all three candidates: 80 rows, exact columns/order, finite probabilities in [0,1]", 
        "public_order_validated": True, "probability_range": {
            "baseline_min": float(baseline_p.min()), "baseline_max": float(baseline_p.max()),
            "residual_min": float(residual_p.min()), "residual_max": float(residual_p.max())
        }, "baseline_source": str(BASELINE.relative_to(ROOT)),
        "oof_reference": {"experiment": "212_stft_residual_seed_stability", "repeated_seed_mean_auc": 0.805494, "public_auc": None, "public_f1": None}
    }
    (HERE / "input_metadata" / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf8")
    (HERE / "input_metadata" / "training_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2), encoding="utf8")
    metrics = {"public_auc": None, "public_f1": None, "training_oof_reference_auc": 0.805494, "training_oof_reference_f1": 0.723512, "elapsed_seconds": time.perf_counter() - started}
    (HERE / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf8")
    report = [
        "# 实验213：公开集提交候选生成", "", "## 协议", "",
        "严格复用实验212的STFT缓存、TinySpectrogramCNN结构、训练集通道均值/std和三个预声明seed（210/211/212）。使用排除HC43后的69名训练被试全部2760窗口训练，每个seed训练2 epoch；公开标签未读取。A候选为三种seed CNN公开预测均值与实验057 plus_pe公开预测按0.2残差权重融合，B候选为严格复用的057 plus_pe公开预测。", "",
        "## 候选与校验", "", "- A：`predictions/repeated_seed_mean_residual_0.2_public.csv`", "- B：`predictions/baseline_057_plus_pe_public.csv`", "- C：`predictions/seed_210_residual_0.2_public.csv`（固定seed210全训练残差0.2备选，非OOF模型）", "- 三个文件均为80行，列顺序为`user_id,trial_id,prob. of positive emotion`；P_test1–P_test10各8行，trial_id均为1–8；概率有限且位于[0,1]。", "- 写出前已逐行核对STFT缓存公开键与057基线键完全一致；仅白名单加载训练标签、训练/公开特征及身份键。", "- CNN在排除HC43的全训练集重新训练，标准化仅拟合训练集；057基线复用已有全训练预测，本轮未重新训练057。未按公开数据选择模型。", "- 输入/输出SHA256、固定融合权重、训练规模和环境见`input_metadata/provenance.json`；逐seed/epoch训练损失见`input_metadata/training_audit.json`。",  "",
        "## 指标", "", "公开AUC/F1：未知（公开标签未读取）。训练集严格OOF仅作参考：实验212三种seed平均残差0.2窗口AUC=0.805494，窗口F1=0.723512；不得视为公开集指标。", "",
        f"设备：{DEVICE}（{provenance['cuda']}）；耗时：{metrics['elapsed_seconds']:.2f}秒。"
    ]
    (HERE / "实验报告.md").write_text("\n".join(report) + "\n", encoding="utf8")
    print(f"A residual range=[{residual_p.min():.6f}, {residual_p.max():.6f}]", flush=True)
    print(f"B baseline range=[{baseline_p.min():.6f}, {baseline_p.max():.6f}]", flush=True)
    print(f"COMPLETE elapsed={metrics['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
