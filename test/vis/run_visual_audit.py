from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "数据" / "训练集"
OUT = Path(__file__).resolve().parent
FS = 250
CHANNELS = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ", "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ", "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2"]
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}
# Fixed approximate 2-D 10-20-style locations; these are display coordinates, not measured montages.
XY = {"FP1": (-.6, 1.0), "FP2": (.6, 1.0), "F7": (-1.0, .55), "F3": (-.5, .58), "FZ": (0, .62), "F4": (.5, .58), "F8": (1.0, .55), "FT7": (-1.05, .18), "FC3": (-.5, .22), "FCZ": (0, .25), "FC4": (.5, .22), "FT8": (1.05, .18), "T3": (-1.15, 0), "C3": (-.5, 0), "CZ": (0, 0), "C4": (.5, 0), "T4": (1.15, 0), "TP7": (-1.05, -.25), "CP3": (-.5, -.25), "CPZ": (0, -.27), "CP4": (.5, -.25), "TP8": (1.05, -.25), "T5": (-.92, -.62), "P3": (-.45, -.58), "PZ": (0, -.62), "P4": (.45, -.58), "T6": (.92, -.62), "O1": (-.5, -.95), "OZ": (0, -.99), "O2": (.5, -.95)}


def load_subject(path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        with h5py.File(path, "r") as f:
            arrays = [np.asarray(f[k][()], dtype=np.float64).T for k in ("EEG_data_neu", "EEG_data_pos")]
    except (OSError, KeyError):
        m = loadmat(path)
        arrays = [np.asarray(m[k], dtype=np.float64) for k in ("EEG_data_neu", "EEG_data_pos")]
    if any(a.shape != (30, 50000) or not np.isfinite(a).all() for a in arrays):
        raise ValueError(f"invalid training array in {path}")
    segments = np.concatenate([a.reshape(30, 20, 2500).transpose(1, 0, 2) for a in arrays])
    labels = np.array(["neu"] * 20 + ["pos"] * 20)
    return segments, labels


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    paths = sorted((DATA / "正常人").glob("HC*.mat")) + sorted((DATA / "抑郁症患者").glob("DEP*.mat"))
    all_count = len(paths)
    excluded = [p.stem for p in paths if p.stem == "HC43"]
    paths = [p for p in paths if p.stem != "HC43"]
    groups = {p.stem: ("HC" if p.stem.startswith("HC") else "DEP") for p in paths}
    feature_rows, subject_rows, arrays = [], [], {}
    freqs = None
    for i, path in enumerate(paths):
        subject, labels = path.stem, groups[path.stem]
        x, y = load_subject(path)
        arrays[subject] = x
        rms = np.sqrt(np.mean(x * x, axis=2)); abs_mean = np.mean(np.abs(x), axis=2)
        freqs, psd = welch(x, fs=FS, window="hann", nperseg=500, noverlap=250, detrend="constant", scaling="density", axis=-1)
        for emotion in ("neu", "pos"):
            mask = y == emotion
            for ch, name in enumerate(CHANNELS):
                bp = []
                for band, (lo, hi) in BANDS.items():
                    fmask = (freqs >= lo) & (freqs < hi)
                    bp.append(np.trapezoid(psd[mask, ch][:, fmask], freqs[fmask], axis=1))
                bp = np.stack(bp, axis=1)
                row = {"subject": subject, "group": groups[subject], "emotion": emotion, "channel": name,
                       "rms_mean": float(np.mean(rms[mask, ch])), "rms_median": float(np.median(rms[mask, ch])),
                       "abs_mean": float(np.mean(abs_mean[mask, ch]))}
                row.update({f"log_bandpower_{band}": float(np.mean(np.log(np.maximum(bp[:, j], 1e-20)))) for j, band in enumerate(BANDS)})
                feature_rows.append(row)
        subject_rows.append({"subject": subject, "group": groups[subject], "n_windows": 40,
                             "global_rms": float(np.mean(rms)), "global_abs_mean": float(np.mean(abs_mean)),
                             "global_rms_cv": float(np.std(rms) / max(np.mean(rms), 1e-12)),
                             "window_rms_iqr": float(np.percentile(rms, 75) - np.percentile(rms, 25)),
                             "mean_channel_rms_sd": float(np.std(np.mean(rms, axis=0)))})
        print(f"[{i + 1}/{len(paths)}] {subject}")
    write_csv(OUT / "subject_channel_emotion_summary.csv", feature_rows)
    write_csv(OUT / "subject_heterogeneity.csv", subject_rows)

    effects = []
    for emotion in ("neu", "pos"):
        for channel in CHANNELS:
            for band in BANDS:
                vals = {g: np.array([r[f"log_bandpower_{band}"] for r in feature_rows if r["emotion"] == emotion and r["channel"] == channel and r["group"] == g]) for g in ("HC", "DEP")}
                diff = vals["DEP"].mean() - vals["HC"].mean(); pooled = np.sqrt(((len(vals["HC"]) - 1) * vals["HC"].var(ddof=1) + (len(vals["DEP"]) - 1) * vals["DEP"].var(ddof=1)) / (len(vals["HC"]) + len(vals["DEP"]) - 2))
                midpoint = (vals["HC"].mean() + vals["DEP"].mean()) / 2.0
                expected_dep_higher = diff >= 0
                consistency = np.mean(vals["DEP"] >= midpoint) if expected_dep_higher else np.mean(vals["DEP"] <= midpoint)
                consistency += np.mean(vals["HC"] <= midpoint) if expected_dep_higher else np.mean(vals["HC"] >= midpoint)
                consistency /= 2.0
                effects.append({"emotion": emotion, "channel": channel, "band": band, "hc_mean": float(vals["HC"].mean()), "dep_mean": float(vals["DEP"].mean()), "dep_minus_hc": float(diff), "cohens_d": float(diff / pooled) if pooled > 0 else 0.0, "direction_consistency_subjects": float(consistency)})
    write_csv(OUT / "hc_dep_effects.csv", effects)

    # Layout figure
    fig, ax = plt.subplots(figsize=(8, 6)); ax.set_aspect("equal"); ax.add_patch(plt.Circle((0, 0), 1.16, fill=False, color="0.4"))
    for ch in CHANNELS:
        x0, y0 = XY[ch]; ax.scatter(x0, y0, s=35, color="#176b87"); ax.text(x0, y0 + .045, ch, ha="center", va="bottom", fontsize=7)
    ax.set_title("30通道固定二维近似布局（非实测坐标）"); ax.axis("off"); fig.tight_layout(); fig.savefig(OUT / "channel_layout.png", dpi=180); plt.close(fig)

    # Representative robust envelope: median and 10-90% across 20 windows, five channels.
    rep = "HC01" if "HC01" in arrays else paths[0].stem; x = arrays[rep][:20, :5]; t = np.arange(x.shape[-1]) / FS
    fig, axes = plt.subplots(5, 1, figsize=(11, 9), sharex=True)
    for ch, ax in enumerate(axes):
        q10, med, q90 = np.percentile(x[:, ch], [10, 50, 90], axis=0); ax.fill_between(t, q10, q90, color="#e6a33a", alpha=.35); ax.plot(t, med, color="#154c79", lw=.7); ax.set_ylabel(CHANNELS[ch]); ax.grid(alpha=.2)
    axes[-1].set_xlabel("时间 (s)"); fig.suptitle(f"代表HC被试 {rep}：neu 20窗稳健包络（10–90%）"); fig.tight_layout(); fig.savefig(OUT / "representative_waveform_envelope.png", dpi=160); plt.close(fig)

    # Group heatmap of DEP-HC log bandpower by channel and band, averaged over emotions.
    mat = np.zeros((30, 5))
    for i, ch in enumerate(CHANNELS):
        for j, band in enumerate(BANDS):
            mat[i, j] = np.mean([r["dep_minus_hc"] for r in effects if r["channel"] == ch and r["band"] == band])
    fig, ax = plt.subplots(figsize=(8, 10)); im = ax.imshow(mat, aspect="auto", cmap="RdBu_r"); ax.set_yticks(range(30), CHANNELS); ax.set_xticks(range(5), BANDS); ax.set_title("DEP − HC 平均 log-bandpower（描述性）"); fig.colorbar(im, ax=ax, label="log power差"); fig.tight_layout(); fig.savefig(OUT / "hc_dep_effect_heatmap.png", dpi=170); plt.close(fig)

    report = {"total_training_files": all_count, "excluded_subjects": excluded, "analyzed_subjects": len(paths), "groups": {"HC": sum(g == "HC" for g in groups.values()), "DEP": sum(g == "DEP" for g in groups.values())}, "windows_per_subject": 40, "windows_total": len(paths) * 40, "channels": CHANNELS, "emotions": {"neu": 20, "pos": 20}, "sampling_rate_hz": FS, "public_test_accessed": False, "auc": None, "f1": None, "interpretation": "All outputs are descriptive label diagnostics; no model fitting, tuning, public prediction, or public-test access."}
    (OUT / "audit_metadata.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# EEG基础统计与可视化审计", "", f"- 输入训练文件：{all_count}；排除：{', '.join(excluded) or '无'}；主分析：{len(paths)}人（HC {report['groups']['HC']}、DEP {report['groups']['DEP']}）。", f"- 每人40窗（neu/pos各20个10秒窗），共{len(paths)*40}窗；30通道；250 Hz。", "- 标签仅作为描述性诊断分层：neu/pos是刺激情绪标签，HC/DEP是被试组别，不能互换。", "- 通道图使用脚本内固定二维近似坐标，不代表实测电极位置。", "- 效应量为被试汇总层面的 DEP−HC Cohen's d；一致率是被试值相对总体方向的描述性比例，不是显著性或泛化证据。", "- 未访问公开测试集，不训练、不调参、不生成预测；AUC/F1不适用于本统计审计，报告为 N/A。", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
