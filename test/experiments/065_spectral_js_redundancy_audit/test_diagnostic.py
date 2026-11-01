import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).parent

def test_diagnostic_artifacts_and_invariants():
    result = json.loads((HERE / 'results' / 'results.json').read_text(encoding='utf8'))
    assert result['provenance']['shape'] == [2760, 780]
    assert result['provenance']['subjects'] == 69
    assert result['provenance']['rows_per_subject'] == 40
    assert result['alignment']['finite'] is True
    assert result['correlation']['nonfinite_pooled'] == 0
    assert result['correlation']['nonfinite_subject_mean'] == 0
    assert result['correlation']['max_matrix_abs_diff'] < 1e-12
    assert np.isfinite(result['correlation']['pooled_max_abs'])
    assert np.isfinite(result['correlation']['mean_subject_max_abs'])
