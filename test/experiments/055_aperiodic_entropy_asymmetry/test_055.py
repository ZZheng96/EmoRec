import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('exp055', HERE / 'run_experiment.py')
exp = importlib.util.module_from_spec(spec); spec.loader.exec_module(exp)
CONFIG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


def pink_noise(rng, n, exponent=1.0, fs=250):
    spectrum = rng.standard_normal(n // 2 + 1) + 1j * rng.standard_normal(n // 2 + 1)
    f = np.fft.rfftfreq(n, 1 / fs); f[0] = f[1]
    return np.fft.irfft(spectrum / f ** (exponent / 2), n)


def test_shapes_finite_and_names():
    rng = np.random.default_rng(0)
    X, names = exp.new_window_features(rng.standard_normal((3, 30, 2500)), CONFIG)
    assert X.shape == (3, 210) and len(names) == 210 and len(set(names)) == 210 and np.isfinite(X).all()
    assert {exp.family_of(n) for n in names} == set(exp.FAMILIES) - {'D_asymmetry'}
    entropy = X[:, 120:150]
    assert (entropy > 0.9).all() and (entropy <= 1.0).all()  # white noise is near-maximal entropy


def test_aperiodic_exponent_and_alpha_peak_recovered():
    rng = np.random.default_rng(1); t = np.arange(2500) / 250
    seg = np.stack([pink_noise(rng, 2500) for _ in range(30)])
    seg = seg / seg.std() + 0.5 * np.sin(2 * np.pi * 10 * t)
    X, _ = exp.new_window_features(seg[None], CONFIG)
    assert np.all(np.abs(X[0, :30] - 1.0) < 0.3)
    assert np.all(np.abs(X[0, 60:90] - 10.0) <= 0.5) and (X[0, 90:120] > 0).all()


def test_wrong_shape_rejected():
    with pytest.raises(ValueError):
        exp.new_window_features(np.zeros((2, 29, 2500)), CONFIG)


def test_alignment_assertion():
    a = {k: np.arange(4) for k in exp.ALIGN_KEYS}; b = {k: v.copy() for k, v in a.items()}
    exp.assert_alignment(a, b)
    b['original_trial_ids'] = b['original_trial_ids'][::-1]
    with pytest.raises(RuntimeError):
        exp.assert_alignment(a, b)


def test_column_sets_dims():
    dims = {m: len(c) for m, c in exp.column_sets().items()}
    assert dims == {'control_600': 600, 'plus_ABC': 810, 'plus_D': 660, 'plus_all': 870, 'new_only_ABCD': 270}


def test_within_subject_effects_shape():
    rng = np.random.default_rng(2)
    subjects = np.repeat(['HC01', 'DEP01'], 40); groups = np.repeat(['HC', 'DEP'], 40); y = np.tile(np.r_[np.zeros(20), np.ones(20)], 2).astype(int)
    names = ['aperiodic_exponent_FP1', 'asymmetry_FP1_FP2_alpha']
    Xn = rng.standard_normal((80, 2)); Xn[y == 1, 0] += 3
    table, fam = exp.within_subject_effects(Xn, y, subjects, groups, names)
    assert table.loc[0, 'consistency_all'] == 1.0 and table.loc[0, 'abs_d_all'] > 1.5 and len(fam) == len(exp.FAMILIES)
