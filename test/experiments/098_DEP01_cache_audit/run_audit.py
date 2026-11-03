import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CONFIG = json.loads((ROOT / "test/experiments/097_DEP01_nonlinear_pairs/config.json").read_text(encoding="utf8"))
CACHE_KEYS = ("subject_ids", "health_groups", "original_trial_ids", "y_train")
FEATURE_KEYS = ("X_train", "feature_names")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def main():
    paths = {name: (HERE / "../097_DEP01_nonlinear_pairs" / CONFIG[name]).resolve()
             for name in ("spectral_cache", "temporal_cache", "pe_cache")}
    caches = {name: load(path) for name, path in paths.items()}
    base_name = "spectral_cache"
    base = caches[base_name]
    rows = []
    for name, cache in caches.items():
        assert cache["X_train"].shape[0] == len(cache["subject_ids"])
        assert cache["X_train"].shape[0] == len(cache["y_train"])
        assert np.isfinite(cache["X_train"]).all()
        rows.append({
            "cache": name,
            "path": str(paths[name]),
            "sha256": sha(paths[name]),
            "rows": int(cache["X_train"].shape[0]),
            "features": int(cache["X_train"].shape[1]),
            "feature_names": int(len(cache["feature_names"])),
        })
        for key in CACHE_KEYS:
            np.testing.assert_array_equal(base[key], cache[key])
    dep = base["subject_ids"] == "DEP01"
    assert int(dep.sum()) == 40
    ref = pd.DataFrame({
        "global_row": np.flatnonzero(dep),
        "subject": base["subject_ids"][dep],
        "group": base["health_groups"][dep],
        "trial": base["original_trial_ids"][dep],
        "y": base["y_train"][dep],
    })
    ref["position"] = ref.groupby(["y", "trial"], sort=False).cumcount()
    assert set(ref.group) == {"DEP"}
    assert set(ref.y) == {0, 1}
    assert (ref.groupby(["y", "trial"]).size() == 5).all()
    assert len(ref.groupby(["y", "trial"])) == 8
    feature_stats = []
    for name, cache in caches.items():
        x = cache["X_train"][dep]
        for row, global_row in enumerate(np.flatnonzero(dep)):
            feature_stats.append({
                "cache": name,
                "local_row": row,
                "global_row": int(global_row),
                "subject": str(cache["subject_ids"][global_row]),
                "trial": int(cache["original_trial_ids"][global_row]),
                "y": int(cache["y_train"][global_row]),
                "row_mean": float(x[row].mean()),
                "row_std": float(x[row].std()),
                "row_l2": float(np.linalg.norm(x[row])),
            })
    for name, cache in caches.items():
        np.testing.assert_array_equal(cache["subject_ids"][dep], ref.subject.to_numpy())
        np.testing.assert_array_equal(cache["original_trial_ids"][dep], ref.trial.to_numpy())
        np.testing.assert_array_equal(cache["y_train"][dep], ref.y.to_numpy())
    for name, cache in caches.items():
        x = cache["X_train"][dep]
        ref[f"{name}_mean"] = x.mean(axis=1)
        ref[f"{name}_std"] = x.std(axis=1)
    ref.to_csv(HERE / "input_metadata/DEP01_rows.csv", index=False)
    pd.DataFrame(feature_stats).to_csv(HERE / "input_metadata/DEP01_feature_row_stats.csv", index=False)
    result = {
        "subject": "DEP01",
        "alignment": {"metadata_equal": True, "row_count": 40, "trial_count": 8, "windows_per_trial": 5},
        "cache_inventory": rows,
        "trial_table": ref.groupby(["y", "trial"], sort=False).size().reset_index(name="rows").to_dict("records"),
        "feature_row_statistics": {
            name: {
                "finite": bool(np.isfinite(cache["X_train"][dep]).all()),
                "global_mean": float(cache["X_train"][dep].mean()),
                "global_std": float(cache["X_train"][dep].std()),
                "trial_means": {f"y{int(y)}_trial{int(t)}": float(cache["X_train"][dep][(base["y_train"][dep] == y) & (base["original_trial_ids"][dep] == t)].mean()) for y, t in sorted(set(zip(base["y_train"][dep], base["original_trial_ids"][dep])))},
            } for name, cache in caches.items()
        },
        "inputs": {name: {"path": str(path), "sha256": sha(path)} for name, path in paths.items()},
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    (HERE / "input_metadata/provenance.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
