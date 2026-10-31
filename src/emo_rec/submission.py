from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

SUBMISSION_COLUMNS = ["user_id", "trial_id", "prob. of positive emotion"]


def validate_submission(frame: pd.DataFrame, user_ids: Sequence[str]) -> None:
    if list(frame.columns) != SUBMISSION_COLUMNS:
        raise ValueError(f"Columns must be exactly {SUBMISSION_COLUMNS}")

    expected_pairs = [(user_id, trial_id) for user_id in user_ids for trial_id in range(1, 9)]
    actual_pairs = list(frame[["user_id", "trial_id"]].itertuples(index=False, name=None))
    if actual_pairs != expected_pairs:
        raise ValueError("Rows must follow user_id order and trial_id order 1..8, with no gaps or duplicates")

    probabilities = frame["prob. of positive emotion"].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError("Probabilities must be finite values in [0, 1]")


def save_submission(frame: pd.DataFrame, user_ids: Sequence[str], path: Path) -> None:
    validate_submission(frame, user_ids)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_excel(path, index=False)
