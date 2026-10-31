import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "数据"
OUTPUT_DIR = ROOT / "test" / "data_audit"


def array_summary(array: np.ndarray) -> dict:
    channel_std = np.std(array, axis=1)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "nan_count": int(np.isnan(array).sum()),
        "inf_count": int(np.isinf(array).sum()),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "zero_variance_channels": np.flatnonzero(channel_std == 0).tolist(),
    }


def audit_file(path: Path, role: str) -> dict:
    if role == "train":
        with h5py.File(path, "r") as mat:
            arrays = {
                key: np.asarray(value[()]).T
                for key, value in mat.items()
                if isinstance(value, h5py.Dataset) and np.issubdtype(value.dtype, np.number)
            }
        reader = "h5py (MATLAB v7.3; transposed to channels x samples)"
    else:
        mat = loadmat(path)
        arrays = {
            key: np.asarray(value)
            for key, value in mat.items()
            if not key.startswith("__") and isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.number)
        }
        reader = "scipy.io.loadmat"
    record = {
        "path": str(path.relative_to(ROOT)),
        "role": role,
        "subject_id": path.stem,
        "group": path.parent.name,
        "reader": reader,
        "arrays": {key: array_summary(value) for key, value in arrays.items()},
    }
    issues = []
    if role == "train":
        for key in ("EEG_data_neu", "EEG_data_pos"):
            if key not in arrays:
                issues.append(f"missing {key}")
            elif arrays[key].shape != (30, 50000):
                issues.append(f"{key} shape is {arrays[key].shape}, expected (30, 50000)")
    else:
        candidates = [key for key, value in arrays.items() if value.shape == (30, 20000)]
        record["test_data_candidates"] = candidates
        if len(candidates) != 1:
            issues.append(f"expected one 30x20000 test array, found {candidates}")
    for key, summary in record["arrays"].items():
        if summary["nan_count"] or summary["inf_count"]:
            issues.append(f"{key} contains non-finite values")
        if summary["zero_variance_channels"]:
            issues.append(f"{key} contains zero-variance channels")
    record["issues"] = issues
    record["warnings"] = [
        f"{key} has near-zero global std ({summary['std']:.3e})"
        for key, summary in record["arrays"].items()
        if summary["std"] < 1e-6
    ]
    return record


def main() -> None:
    train_files = sorted((DATA_DIR / "训练集").glob("*/*.mat"))
    test_files = sorted((DATA_DIR / "公开测试集").glob("*.mat"))
    records = [audit_file(path, "train") for path in train_files]
    records.extend(audit_file(path, "public_test") for path in test_files)
    issues = [f"{record['path']}: {issue}" for record in records for issue in record["issues"]]
    warnings = [f"{record['path']}: {warning}" for record in records for warning in record["warnings"]]
    if len(train_files) != 70:
        issues.append(f"expected 70 training files, found {len(train_files)}")
    if len(test_files) != 10:
        issues.append(f"expected 10 public test files, found {len(test_files)}")

    report = {
        "train_file_count": len(train_files),
        "public_test_file_count": len(test_files),
        "issue_count": len(issues),
        "issues": issues,
        "warning_count": len(warnings),
        "warnings": warnings,
        "files": records,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "mat_audit.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "files"}, ensure_ascii=False, indent=2))
    print(f"Saved detailed audit to {output_path}")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
