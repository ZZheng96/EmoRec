import numpy as np
import pandas as pd

from run_experiment import apply_quota, quota_by_group


def test_apply_quota_selects_exactly_top_k_with_stable_ties() -> None:
    probabilities = np.array([0.2, 0.9, 0.9, 0.1])
    np.testing.assert_array_equal(apply_quota(probabilities, 2), [0, 1, 1, 0])


def test_quota_by_group_selects_known_quota_per_subject() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["A", "A", "A", "B", "B", "B"],
            "prob_positive": [0.1, 0.8, 0.3, 0.9, 0.2, 0.4],
        }
    )
    np.testing.assert_array_equal(quota_by_group(frame, "subject_id", 1), [0, 1, 0, 1, 0, 0])
