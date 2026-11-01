import json
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

def test_outputs_and_counts():
    m = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    assert m['n_subjects'] == 69
    t = pd.read_csv(HERE / 'tables/subject_quality_vs_oof.csv')
    assert len(t) == 69 and t.subject_id.nunique() == 69
    assert set(t.health_group) == {'HC', 'DEP'}
    assert not t.subject_id.eq('HC43').any()

def test_proxy_recalculation():
    audit = json.loads((HERE / '../../data_audit/mat_audit.json').resolve().read_text(encoding='utf8'))
    expected = {}
    for f in audit['files']:
        if f['role'] != 'train': continue
        vals = []
        for key in ('EEG_data_neu', 'EEG_data_pos'):
            a = f['arrays'][key]
            vals.append((a['std'], max(abs(a['min']-a['mean']), abs(a['max']-a['mean'])) / a['std']))
        expected[Path(f['path']).stem] = (np.mean([v[0] for v in vals]), np.mean([v[1] for v in vals]))
    t = pd.read_csv(HERE / 'tables/subject_quality_vs_oof.csv')
    for _, row in t.iterrows():
        assert np.isclose(row.std_proxy, expected[row.subject_id][0])
        assert np.isclose(row.range_sd_proxy, expected[row.subject_id][1])

def test_auc_recalculation():
    t = pd.read_csv(HERE / 'tables/subject_quality_vs_oof.csv')
    for method in ('control_600', 'plus_pe'):
        vals = []
        for split in (2042, 2043, 2044, 2045, 2046):
            p = pd.read_csv(HERE / f'../058_pe_split_replication/predictions/{method}_split{split}_oof.csv')
            vals.append(p.groupby('subject_id').apply(lambda z: __import__('sklearn').metrics.roc_auc_score(z.y_true, z.prob_positive), include_groups=False))
        avg = pd.concat(vals, axis=1).mean(axis=1)
        got = t.set_index('subject_id')[f'{method}_auc_primary']
        assert np.allclose(avg.sort_index(), got.sort_index())

def test_inference_shape():
    m = json.loads((HERE / 'metrics.json').read_text(encoding='utf8'))
    assert len(m['primary']) == 12 and len(m['repeat_direction_only']) == 12
    assert all(r['n_subjects'] == {'all': 69, 'HC': 44, 'DEP': 25}[r['group']] for r in m['primary'])
