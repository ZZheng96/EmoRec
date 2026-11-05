"""生成官方模板格式（xlsx）的公开测试集提交文件。

v1：实验215 candidate_1（一致性频谱），OOF AUC 0.833320/F1 0.749930，已提交，公开 AUC 0.6875。
v2：实验213 A（0.8×057基线 + 0.2×STFT-CNN），OOF AUC 0.805494/F1 0.723512，每个 trial 独立预测。
v3：实验215 candidate_2（057 plus+PE 基线），OOF AUC 0.799903/F1 0.724910，最保守的对照。
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parent
EXP = ROOT.parent / "test/experiments"
COLUMNS = ["user_id", "trial_id", "prob. of positive emotion"]
USERS = [f"P_test{i}" for i in range(1, 11)]
SOURCES = [
    ("v1_一致性", EXP / "215_final_public_candidates/predictions/candidate_1_consistency_spectral.csv"),
    ("v2_057加STFT", EXP / "213_public_submission_candidates/predictions/repeated_seed_mean_residual_0.2_public.csv"),
    ("v3_057基线", EXP / "215_final_public_candidates/predictions/candidate_2_baseline_plus_pe.csv"),
]


def load(path):
    frame = pd.read_csv(path)
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


def digest(frame, path, source):
    content = frame[COLUMNS].to_csv(index=False, lineterminator="\n").encode()
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "source": source.relative_to(ROOT.parent).as_posix(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "rows": len(frame),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "content_sha256": hashlib.sha256(content).hexdigest(),
        "prob_min": float(frame[COLUMNS[-1]].min()),
        "prob_max": float(frame[COLUMNS[-1]].max()),
        "distinct_probs": int(frame[COLUMNS[-1]].nunique()),
        "positives_at_0.5": int((frame[COLUMNS[-1]] >= 0.5).sum()),
    }


def main():
    records = []
    for label, source in SOURCES:
        frame = load(source)
        path = ROOT / f"赞美诗_测试结果_{label}.xlsx"
        write_xlsx(frame, path)
        records.append(digest(frame, path, source))
    (ROOT / "提交校验.json").write_text(
        json.dumps({"submissions": records}, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for record in records:
        print(record)


if __name__ == "__main__":
    main()
