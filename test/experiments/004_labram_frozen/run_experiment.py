import argparse
import ast
import hashlib
import importlib.metadata
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from emo_rec.data import load_public_test_segments
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parent
CACHE_DIR = EXPERIMENT_DIR / "cache"
CACHE_PATH = CACHE_DIR / "embeddings.npz"
SMOKE_PATH = CACHE_DIR / "smoke_test.json"
PREDICTIONS_DIR = EXPERIMENT_DIR / "predictions"
LABRAM_REPO = ROOT / "数据" / "external" / "LaBraM"
BASELINE_DIR = EXPERIMENT_DIR.parent / "002_subject_normalization"
BASELINE_RUN = "subject_zscore_exclude_hc43"
RUN_NAMES = ("labram_frozen_raw", "labram_frozen_subject_zscore")
COMPETITION_CHANNELS = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "FT7", "FC3", "FCZ",
    "FC4", "FT8", "T3", "C3", "CZ", "C4", "T4", "TP7", "CP3", "CPZ",
    "CP4", "TP8", "T5", "P3", "PZ", "P4", "T6", "O1", "OZ", "O2",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_commit(repo: Path) -> str:
    if not (repo / ".git").exists():
        return (repo / "SOURCE_COMMIT").read_text(encoding="utf-8").strip()
    head = (repo / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if not head.startswith("ref: "):
        return head
    ref = head.removeprefix("ref: ")
    ref_path = repo / ".git" / ref
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()
    packed_refs = (repo / ".git" / "packed-refs").read_text(encoding="utf-8").splitlines()
    for line in packed_refs:
        if not line.startswith(("#", "^")) and line.endswith(f" {ref}"):
            return line.split(" ", 1)[0]
    raise ValueError(f"Cannot resolve {ref} in {repo}")


def load_official_standard_1020(repo: Path = LABRAM_REPO) -> list[str]:
    tree = ast.parse((repo / "utils.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "standard_1020" for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("standard_1020 was not found in official utils.py")


def get_input_chans(channel_names: list[str]) -> list[int]:
    standard_1020 = load_official_standard_1020()
    normalized_names = [name.upper() for name in channel_names]
    return [0] + [standard_1020.index(name) + 1 for name in normalized_names]


def extraction_signature(config: dict) -> str:
    relevant = {
        "official": config["official"],
        "channels": config["channels"],
        "preprocessing": config["preprocessing"],
        "labram_model": config["labram_model"],
        "amp": config["extraction"]["amp"],
    }
    payload = json.dumps(relevant, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_provenance(config: dict) -> dict:
    official = config["official"]
    actual_commit = resolve_repo_commit(LABRAM_REPO)
    if actual_commit != official["repo_commit"]:
        raise RuntimeError(f"LaBraM commit mismatch: {actual_commit} != {official['repo_commit']}")
    checkpoint_path = LABRAM_REPO / official["checkpoint"]
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"LaBraM checkpoint not found: {checkpoint_path}")
    actual_sha256 = sha256_file(checkpoint_path)
    if actual_sha256 != official["checkpoint_sha256"]:
        raise RuntimeError("LaBraM checkpoint SHA256 mismatch")
    license_path = LABRAM_REPO / "LICENSE"
    if not license_path.exists() or "MIT License" not in license_path.read_text(encoding="utf-8"):
        raise RuntimeError("Official MIT LICENSE was not found")
    return {
        "repo_url": official["repo_url"],
        "repo_commit": actual_commit,
        "checkpoint_url": official["checkpoint_url"],
        "checkpoint_sha256": actual_sha256,
        "code_license": "MIT",
        "code_license_sha256": sha256_file(license_path),
    }


def select_device(config: dict) -> torch.device:
    requested = config["extraction"]["device"]
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    return torch.device(requested)


def load_labram(config: dict) -> tuple[torch.nn.Module, list[int], dict, torch.device]:
    provenance = verify_provenance(config)
    device = select_device(config)
    sys.path.insert(0, str(LABRAM_REPO))
    try:
        import modeling_finetune
    finally:
        sys.path.pop(0)

    model = modeling_finetune.labram_base_patch200_200(
        pretrained=False,
        num_classes=2,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        use_mean_pooling=True,
        init_scale=0.001,
        use_rel_pos_bias=False,
        use_abs_pos_emb=True,
        init_values=0.1,
        qkv_bias=False,
    )
    checkpoint_path = LABRAM_REPO / config["official"]["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_model = checkpoint
    if isinstance(checkpoint, dict):
        for key in ("model", "module"):
            if key in checkpoint:
                checkpoint_model = checkpoint[key]
                break
    if not isinstance(checkpoint_model, dict):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    if not any(key.startswith("student.") for key in checkpoint_model):
        raise RuntimeError("Official pretraining checkpoint has no student.* backbone weights")
    discarded_container_keys = [key for key in checkpoint_model if not key.startswith("student.")]
    checkpoint_model = {
        key.removeprefix("student."): value
        for key, value in checkpoint_model.items()
        if key.startswith("student.")
    }

    state_dict = model.state_dict()
    ignored_keys = []
    for key in list(checkpoint_model):
        if key.startswith("head.") and (
            key not in state_dict or checkpoint_model[key].shape != state_dict[key].shape
        ):
            checkpoint_model.pop(key)
            ignored_keys.append(key)
    checkpoint_model = {
        key: value for key, value in checkpoint_model.items() if "relative_position_index" not in key
    }
    load_result = model.load_state_dict(checkpoint_model, strict=False)
    allowed_missing = {"head.weight", "head.bias", "fc_norm.weight", "fc_norm.bias"}
    allowed_unexpected = {"mask_token", "norm.weight", "norm.bias", "lm_head.weight", "lm_head.bias"}
    missing = set(load_result.missing_keys)
    unexpected = set(load_result.unexpected_keys)
    if not missing.issubset(allowed_missing) or not unexpected.issubset(allowed_unexpected):
        raise RuntimeError(f"Backbone weight loading failed: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if not torch.equal(model.fc_norm.weight, torch.ones_like(model.fc_norm.weight)) or not torch.equal(
        model.fc_norm.bias, torch.zeros_like(model.fc_norm.bias)
    ):
        raise RuntimeError("Default-initialized fc_norm is not LayerNorm(1, 0)")

    model.to(device)
    model.eval()
    model.requires_grad_(False)
    input_chans = get_input_chans(COMPETITION_CHANNELS)
    load_info = {
        **provenance,
        "checkpoint_top_level_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else [],
        "student_prefix_stripped": True,
        "discarded_non_student_container_keys": discarded_container_keys,
        "ignored_classifier_keys": ignored_keys,
        "allowed_default_initialized_keys": sorted(missing),
        "allowed_pretraining_only_keys": sorted(unexpected),
        "fc_norm_initialization": "Official default LayerNorm weight=1, bias=0; fc_norm is absent from the pretraining checkpoint",
        "missing_keys": load_result.missing_keys,
        "unexpected_keys": load_result.unexpected_keys,
        "all_backbone_weights_loaded": True,
        "torch_weights_only": False,
        "torch_weights_only_reason": "The SHA256-verified official checkpoint container includes numpy scalar metadata, which weights_only=True rejects",
        "frozen": all(not parameter.requires_grad for parameter in model.parameters()),
        "input_chans": input_chans,
    }
    return model, input_chans, load_info, device


def preprocess_continuous_trial(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != 30:
        raise ValueError(f"Expected (30, time) EEG, got {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("EEG contains non-finite values")
    filtered = mne.filter.filter_data(
        data,
        sfreq=250,
        l_freq=0.1,
        h_freq=75.0,
        method="iir",
        iir_params={"order": 4, "ftype": "butter", "output": "sos"},
        phase="zero",
        verbose=False,
    )
    notched = mne.filter.notch_filter(filtered, Fs=250, freqs=[50.0], verbose=False)
    resampled = mne.filter.resample(notched, up=4, down=5)
    expected_length = data.shape[1] * 4 // 5
    if resampled.shape != (30, expected_length) or not np.isfinite(resampled).all():
        raise RuntimeError(f"Invalid preprocessed EEG shape or values: {resampled.shape}")
    return resampled


def split_train_trial(processed: np.ndarray) -> np.ndarray:
    if processed.shape != (30, 10000):
        raise ValueError(f"Expected (30, 10000), got {processed.shape}")
    return processed.reshape(30, 5, 2000).transpose(1, 0, 2)


def load_train_trials(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    trials_by_emotion = []
    with h5py.File(path, "r") as mat:
        for key in ("EEG_data_neu", "EEG_data_pos"):
            if key not in mat:
                raise ValueError(f"{path} does not contain {key}")
            data = np.asarray(mat[key][()], dtype=np.float64).T
            if data.shape != (30, 50000) or not np.isfinite(data).all():
                raise ValueError(f"{path}: invalid {key} shape or values")
            trials_by_emotion.append(data.reshape(30, 4, 12500).transpose(1, 0, 2))
    trials = np.concatenate(trials_by_emotion, axis=0)
    labels = np.concatenate([np.zeros(4, dtype=np.int64), np.ones(4, dtype=np.int64)])
    trial_ids = np.tile(np.arange(1, 5), 2)
    return trials, labels, trial_ids


def preprocess_train_subject(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    trials, labels, original_trial_ids = load_train_trials(path)
    segments = []
    for trial in trials:
        segments.append(split_train_trial(preprocess_continuous_trial(trial)))
    return (
        np.concatenate(segments, axis=0),
        np.repeat(labels, 5),
        np.repeat(original_trial_ids, 5),
        trials,
    )


def preprocess_public_user(path: Path) -> tuple[np.ndarray, np.ndarray]:
    trials = load_public_test_segments(path)
    segments = [preprocess_continuous_trial(trial) for trial in trials]
    return np.asarray(segments), trials


def prepare_model_input(segments: np.ndarray) -> torch.Tensor:
    segments = np.asarray(segments, dtype=np.float32)
    if segments.ndim != 3 or segments.shape[1:] != (30, 2000):
        raise ValueError(f"Expected (batch, 30, 2000), got {segments.shape}")
    if not np.isfinite(segments).all():
        raise ValueError("Segments contain non-finite values")
    samples = segments.reshape(len(segments), 30, 10, 200) / 100.0
    return torch.from_numpy(np.ascontiguousarray(samples))


@torch.no_grad()
def extract_embeddings(
    model: torch.nn.Module,
    input_chans: list[int],
    segments: np.ndarray,
    device: torch.device,
    batch_size: int,
    use_amp: bool,
) -> np.ndarray:
    embeddings = []
    for start in range(0, len(segments), batch_size):
        samples = prepare_model_input(segments[start : start + batch_size]).to(device)
        if use_amp:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                features = model.forward_features(samples, input_chans=input_chans)
        else:
            features = model.forward_features(samples, input_chans=input_chans)
        if features.ndim != 2 or features.shape[0] != len(samples) or not torch.isfinite(features).all():
            raise RuntimeError(f"Invalid LaBraM embedding batch: {tuple(features.shape)}")
        embeddings.append(features.float().cpu().numpy())
    result = np.concatenate(embeddings, axis=0)
    if result.shape != (len(segments), 200):
        raise RuntimeError(f"Expected embeddings {(len(segments), 200)}, got {result.shape}")
    return result


def make_subject_folds(subject_ids: np.ndarray, health_groups: np.ndarray, seed: int) -> np.ndarray:
    subjects = np.unique(subject_ids)
    subject_health = np.asarray([health_groups[np.flatnonzero(subject_ids == subject)[0]] for subject in subjects])
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_by_subject = {}
    for fold, (_, validation_indices) in enumerate(splitter.split(subjects, subject_health)):
        for subject in subjects[validation_indices]:
            fold_by_subject[subject] = fold
    return np.asarray([fold_by_subject[subject] for subject in subject_ids], dtype=np.int64)


def numeric_test_sort_key(path: Path) -> int:
    return int(path.stem.replace("P_test", ""))


def run_smoke_test(config: dict) -> None:
    model, input_chans, load_info, device = load_labram(config)
    train_path = ROOT / "数据" / "训练集" / "正常人" / "HC01.mat"
    public_path = ROOT / "数据" / "公开测试集" / "P_test1.mat"
    train_segments, labels, original_trial_ids, raw_train_trials = preprocess_train_subject(train_path)
    public_segments, raw_public_trials = preprocess_public_user(public_path)
    train_embeddings = extract_embeddings(
        model, input_chans, train_segments, device, config["extraction"]["batch_size"], config["extraction"]["amp"]
    )
    public_embeddings = extract_embeddings(
        model, input_chans, public_segments, device, config["extraction"]["batch_size"], config["extraction"]["amp"]
    )
    if train_embeddings.shape != (40, 200) or public_embeddings.shape != (8, 200):
        raise RuntimeError("Smoke-test embedding shapes are incorrect")
    if not np.isfinite(train_embeddings).all() or not np.isfinite(public_embeddings).all():
        raise RuntimeError("Smoke-test embeddings contain non-finite values")
    result = {
        "status": "passed",
        "train_subject": "HC01",
        "public_user": "P_test1",
        "train_segments_shape": list(train_segments.shape),
        "public_segments_shape": list(public_segments.shape),
        "train_embeddings_shape": list(train_embeddings.shape),
        "public_embeddings_shape": list(public_embeddings.shape),
        "labels_shape": list(labels.shape),
        "original_trial_ids_shape": list(original_trial_ids.shape),
        "all_values_finite": True,
        "raw_train_trial_channel_std_median": float(np.median(np.std(raw_train_trials, axis=-1))),
        "raw_public_trial_channel_std_median": float(np.median(np.std(raw_public_trials, axis=-1))),
        "input_unit": "microvolts; no 1e6 multiplication; divide by 100 at model input",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "weight_loading": load_info,
    }
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_DIR / "smoke_test.npz",
        X_train=train_embeddings,
        X_public=public_embeddings,
        y_train=labels,
        original_trial_ids=original_trial_ids,
    )
    SMOKE_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def build_feature_cache(config: dict) -> None:
    if not SMOKE_PATH.exists():
        raise RuntimeError("Run --smoke-test successfully before full feature extraction")
    model, input_chans, load_info, device = load_labram(config)
    train_files = sorted((ROOT / "数据" / "训练集").glob("*/*.mat"))
    all_subjects = np.asarray([path.stem for path in train_files])
    all_health = np.asarray(["DEP" if path.parent.name == "抑郁症患者" else "HC" for path in train_files])
    all_subject_folds = make_subject_folds(all_subjects, all_health, config["random_seed"])
    fold_by_subject = dict(zip(all_subjects, all_subject_folds))

    train_embeddings = []
    labels = []
    subjects = []
    health_groups = []
    original_trial_ids = []
    folds = []
    included_files = [path for path in train_files if path.stem != "HC43"]
    if len(included_files) != 69:
        raise RuntimeError(f"Expected 69 subjects after HC43 exclusion, found {len(included_files)}")
    for index, path in enumerate(included_files, start=1):
        segments, subject_labels, trial_ids, _ = preprocess_train_subject(path)
        embeddings = extract_embeddings(
            model, input_chans, segments, device, config["extraction"]["batch_size"], config["extraction"]["amp"]
        )
        train_embeddings.append(embeddings)
        labels.append(subject_labels)
        subjects.extend([path.stem] * len(segments))
        health_groups.extend(["DEP" if path.parent.name == "抑郁症患者" else "HC"] * len(segments))
        original_trial_ids.append(trial_ids)
        folds.extend([fold_by_subject[path.stem]] * len(segments))
        print(f"LaBraM train embeddings: {index}/69 {path.stem}", flush=True)

    public_embeddings = []
    public_user_ids = []
    public_trial_ids = []
    test_files = sorted((ROOT / "数据" / "公开测试集").glob("*.mat"), key=numeric_test_sort_key)
    for path in test_files:
        segments, _ = preprocess_public_user(path)
        embeddings = extract_embeddings(
            model, input_chans, segments, device, config["extraction"]["batch_size"], config["extraction"]["amp"]
        )
        public_embeddings.append(embeddings)
        public_user_ids.extend([path.stem] * len(segments))
        public_trial_ids.extend(np.arange(1, 9))
        print(f"LaBraM public embeddings: {path.stem}", flush=True)

    X_train = np.concatenate(train_embeddings)
    X_public = np.concatenate(public_embeddings)
    if X_train.shape != (2760, 200) or X_public.shape != (80, 200):
        raise RuntimeError(f"Unexpected full cache shapes: {X_train.shape}, {X_public.shape}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        X_train=X_train,
        y_train=np.concatenate(labels),
        subject_ids=np.asarray(subjects),
        health_groups=np.asarray(health_groups),
        original_trial_ids=np.concatenate(original_trial_ids),
        folds=np.asarray(folds, dtype=np.int64),
        X_public=X_public,
        public_user_ids=np.asarray(public_user_ids),
        public_trial_ids=np.asarray(public_trial_ids),
        extraction_signature=np.asarray(extraction_signature(config)),
        extraction_device=np.asarray(str(device)),
        checkpoint_sha256=np.asarray(load_info["checkpoint_sha256"]),
    )


def load_feature_cache(config: dict) -> dict[str, np.ndarray]:
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"Embedding cache not found: {CACHE_PATH}")
    with np.load(CACHE_PATH) as cache:
        data = {key: cache[key] for key in cache.files}
    if str(data["extraction_signature"]) != extraction_signature(config):
        raise RuntimeError("Embedding cache does not match the current extraction config; use --rebuild-features")
    if np.any(data["subject_ids"] == "HC43") or data["X_train"].shape != (2760, 200):
        raise RuntimeError("Cache violates HC43 exclusion or expected shape")
    if not np.isfinite(data["X_train"]).all() or not np.isfinite(data["X_public"]).all():
        raise RuntimeError("Embedding cache contains non-finite values")
    return data


def make_model(model_config: dict, seed: int):
    classifier = LogisticRegression(
        C=model_config["C"],
        penalty="l2",
        max_iter=model_config["max_iter"],
        solver="lbfgs",
        random_state=seed,
    )
    return make_pipeline(StandardScaler(), classifier)


def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "binary_f1": float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def load_baseline() -> tuple[pd.DataFrame, dict]:
    oof = pd.read_csv(BASELINE_DIR / "predictions" / f"{BASELINE_RUN}_oof.csv")
    y_true = oof["y_true"].to_numpy()
    probabilities = oof["prob_positive"].to_numpy()
    overall = classification_metrics(y_true, probabilities, threshold=0.5)
    saved = json.loads((BASELINE_DIR / "metrics.json").read_text(encoding="utf-8"))["runs"][BASELINE_RUN]
    health_groups = {}
    for group in ("HC", "DEP"):
        mask = oof["health_group"].to_numpy() == group
        health_groups[group] = {
            **saved["health_groups"][group],
            "pooled": classification_metrics(y_true[mask], probabilities[mask], threshold=0.5),
        }
    return oof, {
        "overall": overall,
        "health_groups": health_groups,
        "subject_auc_mean": saved["subject_auc_mean"],
    }


def validate_against_baseline(data: dict[str, np.ndarray], baseline_oof: pd.DataFrame) -> None:
    if len(baseline_oof) != len(data["y_train"]):
        raise RuntimeError("004 and 002 OOF sample counts differ")
    comparisons = {
        "subject_id": data["subject_ids"],
        "original_trial_id": data["original_trial_ids"],
        "fold": data["folds"],
        "y_true": data["y_train"],
    }
    for column, values in comparisons.items():
        if not np.array_equal(baseline_oof[column].to_numpy(), values):
            raise RuntimeError(f"004 protocol differs from 002 in column {column}")


def evaluate_run(
    run_name: str,
    X: np.ndarray,
    X_public: np.ndarray,
    data: dict[str, np.ndarray],
    config: dict,
) -> dict:
    y = data["y_train"]
    folds = data["folds"]
    seed = config["random_seed"]
    threshold = config["validation"]["probability_threshold"]
    oof_probabilities = np.full(len(y), np.nan)
    for fold in range(config["validation"]["n_splits"]):
        train_mask = folds != fold
        validation_mask = folds == fold
        model = make_model(config["model"], seed)
        model.fit(X[train_mask], y[train_mask])
        oof_probabilities[validation_mask] = model.predict_proba(X[validation_mask])[:, 1]
        print(f"{run_name}: fold {fold + 1}/5", flush=True)
    if not np.isfinite(oof_probabilities).all():
        raise RuntimeError(f"{run_name}: incomplete OOF predictions")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    oof = pd.DataFrame(
        {
            "subject_id": data["subject_ids"],
            "health_group": data["health_groups"],
            "original_trial_id": data["original_trial_ids"],
            "fold": folds,
            "y_true": y,
            "prob_positive": oof_probabilities,
        }
    )
    oof.to_csv(PREDICTIONS_DIR / f"{run_name}_oof.csv", index=False)

    fold_metrics = []
    for fold in range(config["validation"]["n_splits"]):
        mask = folds == fold
        fold_metrics.append(
            {
                "fold": fold,
                "n_subjects": int(len(np.unique(data["subject_ids"][mask]))),
                "n_samples": int(mask.sum()),
                **classification_metrics(y[mask], oof_probabilities[mask], threshold),
            }
        )

    subject_rows = []
    for subject in np.unique(data["subject_ids"]):
        mask = data["subject_ids"] == subject
        subject_rows.append(
            {
                "run_name": run_name,
                "subject_id": subject,
                "health_group": data["health_groups"][mask][0],
                **classification_metrics(y[mask], oof_probabilities[mask], threshold),
            }
        )
    subject_metrics = pd.DataFrame(subject_rows)
    health_metrics = {}
    for health_group in ("HC", "DEP"):
        mask = data["health_groups"] == health_group
        group_subjects = subject_metrics[subject_metrics["health_group"] == health_group]
        health_metrics[health_group] = {
            "n_subjects": int(len(group_subjects)),
            "n_samples": int(mask.sum()),
            "pooled": classification_metrics(y[mask], oof_probabilities[mask], threshold),
            "subject_auc_mean": float(group_subjects["roc_auc"].mean()),
            "subject_auc_std": float(group_subjects["roc_auc"].std(ddof=1)),
            "subject_macro_f1_mean": float(group_subjects["macro_f1"].mean()),
        }

    final_model = make_model(config["model"], seed)
    final_model.fit(X, y)
    public_probabilities = final_model.predict_proba(X_public)[:, 1]
    public = pd.DataFrame(
        {
            "user_id": data["public_user_ids"],
            "trial_id": data["public_trial_ids"],
            "prob. of positive emotion": public_probabilities,
        }
    )
    validate_submission(public, [f"P_test{i}" for i in range(1, 11)])
    public.to_csv(PREDICTIONS_DIR / f"{run_name}_public.csv", index=False)

    return {
        "n_subjects": int(len(np.unique(data["subject_ids"]))),
        "n_samples": int(len(y)),
        "embedding_dim": int(X.shape[1]),
        "overall": classification_metrics(y, oof_probabilities, threshold),
        "folds": fold_metrics,
        "fold_auc_mean": float(np.mean([fold["roc_auc"] for fold in fold_metrics])),
        "fold_auc_std": float(np.std([fold["roc_auc"] for fold in fold_metrics], ddof=1)),
        "subject_auc_mean": float(subject_metrics["roc_auc"].mean()),
        "subject_auc_std": float(subject_metrics["roc_auc"].std(ddof=1)),
        "health_groups": health_metrics,
        "subject_metrics": subject_rows,
    }


def add_baseline_deltas(result: dict, baseline: dict) -> dict:
    deltas = {
        f"overall_{key}": result["overall"][key] - baseline["overall"][key]
        for key in ("roc_auc", "binary_f1", "macro_f1", "balanced_accuracy")
    }
    deltas["subject_auc_mean"] = result["subject_auc_mean"] - baseline["subject_auc_mean"]
    for group in ("HC", "DEP"):
        for key in ("roc_auc", "binary_f1", "macro_f1", "balanced_accuracy"):
            deltas[f"{group}_pooled_{key}"] = (
                result["health_groups"][group]["pooled"][key]
                - baseline["health_groups"][group]["pooled"][key]
            )
    result["delta_vs_002_best"] = deltas
    return deltas


def runtime_info() -> dict:
    distributions = ["numpy", "scipy", "scikit-learn", "pandas", "torch", "torchvision", "timm", "einops", "mne", "h5py"]
    versions = {}
    for name in distributions:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": sys.version,
        "dependencies": versions,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deepspeed_installed": importlib.util.find_spec("deepspeed") is not None,
    }


def write_summary(all_metrics: dict) -> pd.DataFrame:
    rows = []
    for run_name, result in all_metrics["runs"].items():
        delta = result["delta_vs_002_best"]
        rows.append(
            {
                "run_name": run_name,
                "n_subjects": result["n_subjects"],
                "n_samples": result["n_samples"],
                "embedding_dim": result["embedding_dim"],
                "oof_auc": result["overall"]["roc_auc"],
                "oof_binary_f1": result["overall"]["binary_f1"],
                "oof_macro_f1": result["overall"]["macro_f1"],
                "oof_balanced_accuracy": result["overall"]["balanced_accuracy"],
                "fold_auc_mean": result["fold_auc_mean"],
                "fold_auc_std": result["fold_auc_std"],
                "subject_auc_mean": result["subject_auc_mean"],
                "hc_pooled_auc": result["health_groups"]["HC"]["pooled"]["roc_auc"],
                "dep_pooled_auc": result["health_groups"]["DEP"]["pooled"]["roc_auc"],
                "hc_subject_auc_mean": result["health_groups"]["HC"]["subject_auc_mean"],
                "dep_subject_auc_mean": result["health_groups"]["DEP"]["subject_auc_mean"],
                "delta_oof_auc_vs_002_best": delta["overall_roc_auc"],
                "delta_oof_binary_f1_vs_002_best": delta["overall_binary_f1"],
                "delta_oof_macro_f1_vs_002_best": delta["overall_macro_f1"],
                "delta_oof_balanced_accuracy_vs_002_best": delta["overall_balanced_accuracy"],
                "delta_hc_pooled_auc_vs_002_best": delta["HC_pooled_roc_auc"],
                "delta_dep_pooled_auc_vs_002_best": delta["DEP_pooled_roc_auc"],
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(EXPERIMENT_DIR / "summary.csv", index=False)
    return summary


def evaluate(config: dict) -> None:
    data = load_feature_cache(config)
    baseline_oof, baseline = load_baseline()
    validate_against_baseline(data, baseline_oof)
    smoke = json.loads(SMOKE_PATH.read_text(encoding="utf-8")) if SMOKE_PATH.exists() else None
    all_metrics = {
        "experiment_config": config,
        "baseline": {"experiment": "002_subject_normalization", "run_name": BASELINE_RUN, **baseline},
        "provenance": verify_provenance(config),
        "runtime": runtime_info(),
        "extraction_device": str(data["extraction_device"]),
        "smoke_test": smoke,
        "public_test_labels_used": False,
        "public_predictions_validated": True,
        "runs": {},
    }
    for condition in config["conditions"]:
        run_name = condition["run_name"]
        method = condition["normalization"]
        near_zero_std = config["normalization"]["near_zero_std"]
        X = normalize_by_subject(data["X_train"], data["subject_ids"], method, near_zero_std)
        X_public = normalize_by_subject(data["X_public"], data["public_user_ids"], method, near_zero_std)
        result = evaluate_run(run_name, X, X_public, data, config)
        add_baseline_deltas(result, baseline)
        all_metrics["runs"][run_name] = result
        (EXPERIMENT_DIR / "metrics.json").write_text(
            json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        write_summary(all_metrics)
    print(write_summary(all_metrics).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--rebuild-features", action="store_true")
    args = parser.parse_args()
    config = json.loads((EXPERIMENT_DIR / "config.json").read_text(encoding="utf-8"))
    if tuple(condition["run_name"] for condition in config["conditions"]) != RUN_NAMES:
        raise ValueError(f"Conditions must exactly match: {RUN_NAMES}")
    if args.smoke_test:
        run_smoke_test(config)
        return
    if args.rebuild_features or not CACHE_PATH.exists():
        build_feature_cache(config)
    evaluate(config)


if __name__ == "__main__":
    main()
