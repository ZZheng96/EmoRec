"""从实验215的预声明候选生成官方模板格式的提交文件。

主提交：candidate_1_consistency_spectral（严格OOF AUC 0.833320 / F1 0.749930，当前最佳）。
备选：candidate_2_baseline_plus_pe（严格OOF AUC 0.799903 / F1 0.724910，每个trial独立预测）。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent
SRC = ROOT.parent / "test/experiments/215_final_public_candidates/predictions"
COLUMNS = ["user_id", "trial_id", "prob. of positive emotion"]
USERS = [f"P_test{i}" for i in range(1, 11)]


def load(name):
    frame = pd.read_csv(SRC / name)
    assert list(frame.columns) == COLUMNS, frame.columns
    assert len(frame) == 80
    assert not frame.duplicated(["user_id", "trial_id"]).any()
    assert frame["user_id"].tolist() == [u for u in USERS for _ in range(8)]
    assert frame["trial_id"].tolist() == [t for _ in USERS for t in range(1, 9)]
    probs = frame[COLUMNS[-1]].to_numpy(float)
    assert np.isfinite(probs).all() and (probs >= 0).all() and (probs <= 1).all()
    return frame


def write_xlsx(frame, path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Sheet1"
    sheet.append(COLUMNS)
    for user_id, trial_id, prob in frame[COLUMNS].itertuples(index=False):
        sheet.append([user_id, int(trial_id), float(prob)])
    for column, width in zip("ABC", (12, 10, 26)):
        sheet.column_dimensions[column].width = width
    workbook.save(path)


def digest(frame, path):
    content = frame[COLUMNS].to_csv(index=False, lineterminator="\n").encode()
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "rows": len(frame),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "prob_min": float(frame[COLUMNS[-1]].min()),
        "prob_max": float(frame[COLUMNS[-1]].max()),
        "positives_at_0.5": int((frame[COLUMNS[-1]] >= 0.5).sum()),
    }


def main():
    primary = load("candidate_1_consistency_spectral.csv")
    fallback = load("candidate_2_baseline_plus_pe.csv")
    records = []
    for frame, name in ((primary, "赞美诗_测试结果.xlsx"), (fallback, "赞美诗_测试结果_备选_057基线.xlsx")):
        path = ROOT / name
        write_xlsx(frame, path)
        records.append(digest(frame, path))
    (ROOT / "提交校验.json").write_text(
        json.dumps(
            {
                "source_experiment": "test/experiments/215_final_public_candidates",
                "source_sha256": {
                    name: hashlib.sha256((SRC / name).read_bytes()).hexdigest()
                    for name in ("candidate_1_consistency_spectral.csv", "candidate_2_baseline_plus_pe.csv")
                },
                "submissions": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
