import numpy as np
import pandas as pd
import pytest

from emo_rec.submission import SUBMISSION_COLUMNS, validate_submission

USER_IDS = [f"P_test{i}" for i in range(1, 11)]


def valid_submission() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": np.repeat(USER_IDS, 8),
            "trial_id": np.tile(np.arange(1, 9), 10),
            "prob. of positive emotion": np.linspace(0, 1, 80),
        },
        columns=SUBMISSION_COLUMNS,
    )


def test_valid_submission_has_80_ordered_rows() -> None:
    frame = valid_submission()
    validate_submission(frame, USER_IDS)
    assert len(frame) == 80


def test_submission_rejects_wrong_trial_order() -> None:
    frame = valid_submission()
    frame.loc[0, "trial_id"] = 2
    with pytest.raises(ValueError, match="trial_id order"):
        validate_submission(frame, USER_IDS)


def test_submission_rejects_invalid_probability() -> None:
    frame = valid_submission()
    frame.loc[0, "prob. of positive emotion"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_submission(frame, USER_IDS)
