"""053 hierarchical error audit: pure diagnostic over existing 052/046 OOF predictions.

No model is trained, no public-set metric is used, nothing is tuned. Oracle analyses are
explicitly marked as upper bounds that use validation labels and are NOT valid models.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP = HERE.parent
sys.path.insert(0, str(ROOT / 'src'))
from emo_rec.normalization import normalize_by_subject  # noqa: E402

SPLITS = ['fixed002', 'repeat2027', 'repeat2028', 'repeat2029', 'repeat2030', 'repeat2031']
ST_FILES = {s: EXP / '052_subject_group_dro' / 'predictions' / f'spectral_temporal_hgb_control_{s}_oof.csv' for s in SPLITS}
SP_FILES = {'fixed002': EXP / '046_subject_graph_diffusion' / 'predictions' / 'spectral_control_oof.csv',
            **{f'repeat{x}': EXP / '046_subject_graph_diffusion' / 'predictions' / f'spectral_control_seed{x}_oof.csv' for x in range(2027, 2032)}}
KEY = ['subject_id', 'health_group', 'original_trial_id', 'y_true']
SEED = 2053
N_BOOT = 2000
N_PERM = 200
TOPK = 30
OUT = HERE / 'tables'


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float('nan')


def load_preds(files):
    frames = {s: pd.read_csv(f) for s, f in files.items() if f.exists()}
    if len(frames) != len(files):
        return None
    ref = frames[SPLITS[0]]
    assert len(ref) == 2760, len(ref)
    for s, df in frames.items():
        assert len(df) == 2760 and df[KEY].equals(ref[KEY]), f'row order mismatch in {s}'
        assert np.isfinite(df.prob_positive).all()
        assert df.groupby('fold').subject_id.nunique().sum() == 69, 'subject leaks across folds'
        # original_trial_id is 1..4 within each label; a physical trial is (subject, label, id) -> key 1..8
        df['trial_key'] = df.original_trial_id + 4 * df.y_true
    return frames


def subject_table(df, p):
    rows = []
    d = df.assign(p=p)
    tr = d.groupby(['subject_id', 'trial_key'], sort=False).agg(health_group=('health_group', 'first'), y=('y_true', 'first'), p=('p', 'mean')).reset_index()
    for sid, g in d.groupby('subject_id', sort=False):
        t = tr[tr.subject_id == sid]
        rows.append({'subject_id': sid, 'health_group': g.health_group.iloc[0], 'window_auc': auc(g.y_true, g.p), 'trial_auc': auc(t.y, t.p)})
    return pd.DataFrame(rows), tr


def level_metrics(df, p):
    st, tr = subject_table(df, p)
    out = {'window_auc': auc(df.y_true, p), 'trial_auc': auc(tr.y, tr.p),
           'subject_macro_window_auc': float(st.window_auc.mean()), 'subject_macro_trial_auc': float(st.trial_auc.mean())}
    for g in ('HC', 'DEP'):
        m = df.health_group.to_numpy() == g
        tm = tr.health_group.to_numpy() == g
        out[f'{g}_window_auc'] = auc(df.y_true[m], p[m])
        out[f'{g}_trial_auc'] = auc(tr.y[tm], tr.p[tm])
        out[f'{g}_subject_macro_window_auc'] = float(st.window_auc[st.health_group == g].mean())
    return out, st, tr


def transforms(df, p):
    s = df.subject_id.to_numpy()
    centered = p - pd.Series(p).groupby(s).transform('mean').to_numpy()
    lg = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    logit_z = (lg - pd.Series(lg).groupby(s).transform('mean').to_numpy()) / pd.Series(lg).groupby(s).transform('std').to_numpy()
    rank = pd.Series(p).groupby(s).rank(pct=True).to_numpy()
    return {'raw': p, 'subject_centered': centered, 'subject_logit_zscore': logit_z, 'subject_rank': rank}


def unlabeled_proxies(df, p):
    rows = []
    d = df.assign(p=p)
    for sid, g in d.groupby('subject_id', sort=False):
        x = g.p.to_numpy()
        tm = g.groupby('trial_key').p.mean().to_numpy()
        n = len(x)
        sk, ku = stats.skew(x), stats.kurtosis(x)
        bc = (sk ** 2 + 1) / (ku + 3 * (n - 1) ** 2 / ((n - 2) * (n - 3)))
        rows.append({'subject_id': sid, 'mean_prob': x.mean(), 'abs_mean_minus_half': abs(x.mean() - 0.5), 'std_prob': x.std(),
                     'bimodality_coef': bc, 'between_trial_var_frac': tm.var() / max(x.var(), 1e-12), 'trial_mean_range': tm.max() - tm.min()})
    return pd.DataFrame(rows)


def cohens_d(X, y):
    a, b = X[y == 1], X[y == 0]
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2))
    return (a.mean(0) - b.mean(0)) / np.where(sp > 1e-12, sp, np.nan)


def subject_d_matrix(X, y, subjects, order):
    return np.vstack([cohens_d(X[subjects == s], y[subjects == s]) for s in order])


def sign_stats(D, idx):
    ref = np.sign(np.nanmean(D[:, idx], 0))
    return float(np.mean(np.sign(D[:, idx]) == ref))


def feature_audit(subjects, groups, y, trials, st_mean_auc):
    cfg = json.loads((EXP / '052_subject_group_dro' / 'config.json').read_text(encoding='utf-8'))
    spec = importlib.util.spec_from_file_location('base032', EXP / '032_repeated_subject_folds' / 'run_experiment.py')
    base = importlib.util.module_from_spec(spec); spec.loader.exec_module(base)
    sp = base.load_cache((EXP / '052_subject_group_dro' / cfg['spectral_cache']).resolve())
    tp = base.load_cache((EXP / '052_subject_group_dro' / cfg['temporal_cache']).resolve())
    base.assert_cache_alignment(sp, tp)
    keep = sp['subject_ids'] != cfg['excluded_subject']
    assert np.array_equal(sp['subject_ids'][keep], subjects) and np.array_equal(sp['y_train'][keep], y)
    assert np.array_equal(sp['original_trial_ids'][keep] + 4 * y, trials)
    sm = np.asarray([n.startswith('log_bandpower_') for n in sp['feature_names']])
    names = np.concatenate([np.asarray(sp['feature_names'])[sm], np.asarray(tp['feature_names']) if 'feature_names' in tp else np.asarray([f'temporal_{i}' for i in range(tp['X_train'].shape[1])])])
    X = np.column_stack([normalize_by_subject(sp['X_train'][keep][:, sm], subjects, 'zscore'), normalize_by_subject(tp['X_train'][keep], subjects, 'zscore')])
    assert X.shape == (2760, 600) and np.isfinite(X).all()
    order = pd.unique(subjects)
    grp = np.asarray([groups[subjects == s][0] for s in order])
    D = np.nan_to_num(subject_d_matrix(X, y, subjects, order))
    # trial-level d (window-averaged, 4 vs 4) to avoid pseudo-replication of 5 windows per trial
    tdf = pd.DataFrame(X).assign(s=subjects, t=trials, y=y).groupby(['s', 't'], sort=False).mean().reset_index()
    Xt, st_, yt = tdf.drop(columns=['s', 't', 'y']).to_numpy(), tdf.s.to_numpy(), tdf.y.to_numpy().astype(int)
    Dt = np.nan_to_num(subject_d_matrix(Xt, yt, st_, order))

    res = {}
    for tag, M in (('window_d', D), ('trial_d', Dt)):
        by_abs = np.argsort(-np.mean(np.abs(M), 0))[:TOPK]
        by_signed = np.argsort(-np.abs(np.mean(M, 0)))[:TOPK]
        r = {}
        for sel_name, idx in (('top30_by_mean_abs_d', by_abs), ('top30_by_abs_mean_d', by_signed)):
            r[sel_name] = {'features': names[idx].tolist(), 'sign_agreement_all': sign_stats(M, idx),
                           'sign_agreement_HC_vs_allref': float(np.mean(np.sign(M[grp == 'HC'][:, idx]) == np.sign(np.mean(M[:, idx], 0)))),
                           'sign_agreement_DEP_vs_allref': float(np.mean(np.sign(M[grp == 'DEP'][:, idx]) == np.sign(np.mean(M[:, idx], 0)))),
                           'sign_agreement_HC_own': sign_stats(M[grp == 'HC'], idx), 'sign_agreement_DEP_own': sign_stats(M[grp == 'DEP'], idx),
                           'mean_abs_d_HC': float(np.mean(np.abs(M[grp == 'HC'][:, idx]))), 'mean_abs_d_DEP': float(np.mean(np.abs(M[grp == 'DEP'][:, idx]))),
                           'mean_signed_d_HC': float(np.mean(M[grp == 'HC'][:, idx] * np.sign(np.mean(M[:, idx], 0)))),
                           'mean_signed_d_DEP': float(np.mean(M[grp == 'DEP'][:, idx] * np.sign(np.mean(M[:, idx], 0))))}
        r['mean_abs_d_all600_HC'] = float(np.mean(np.abs(M[grp == 'HC'])))
        r['mean_abs_d_all600_DEP'] = float(np.mean(np.abs(M[grp == 'DEP'])))
        r['HC_vs_DEP_mean_d_vector_corr'] = float(np.corrcoef(M[grp == 'HC'].mean(0), M[grp == 'DEP'].mean(0))[0, 1])
        # alignment: subject d-vector vs leave-subject-out group mean
        tot = M.sum(0)
        align = np.asarray([np.corrcoef(M[i], (tot - M[i]) / (len(M) - 1))[0, 1] for i in range(len(M))])
        hc_tot = M[grp == 'HC'].sum(0); nhc = (grp == 'HC').sum()
        align_hc = np.asarray([np.corrcoef(M[i], (hc_tot - M[i]) / (nhc - 1) if grp[i] == 'HC' else hc_tot / nhc)[0, 1] for i in range(len(M))])
        a = st_mean_auc.reindex(order).to_numpy()
        r['alignment'] = {'mean_HC': float(align[grp == 'HC'].mean()), 'mean_DEP': float(align[grp == 'DEP'].mean()),
                          'n_negative_alignment': int((align < 0).sum()), 'n_negative_alignment_HC': int((align[grp == 'HC'] < 0).sum()), 'n_negative_alignment_DEP': int((align[grp == 'DEP'] < 0).sum()),
                          'mannwhitney_HC_vs_DEP_p': float(stats.mannwhitneyu(align[grp == 'HC'], align[grp == 'DEP']).pvalue)}
        for nm, al in (('loso_all', align), ('loso_HC_template', align_hc)):
            rho, pv = stats.spearmanr(al, a)
            r['alignment'][f'spearman_{nm}_vs_subject_auc'] = {'rho': float(rho), 'p': float(pv)}
            for g in ('HC', 'DEP'):
                rho, pv = stats.spearmanr(al[grp == g], a[grp == g])
                r['alignment'][f'spearman_{nm}_vs_subject_auc_{g}'] = {'rho': float(rho), 'p': float(pv)}
        res[tag] = r
        pd.DataFrame({'subject_id': order, 'health_group': grp, 'alignment_loso_all': align, 'alignment_HC_template': align_hc,
                      'mean_abs_d': np.mean(np.abs(M), 1), 'subject_window_auc_mean6': a}).to_csv(OUT / f'feature_alignment_{tag}.csv', index=False)

    # label-permutation null: permute the 8 trial labels (4/4) within each subject, keep windows tied to trials
    rng = np.random.default_rng(SEED)
    null_abs, null_sign, null_sign_signed = [], [], []
    tri_index = {s: np.unique(trials[subjects == s]) for s in order}
    for _ in range(N_PERM):
        yp = y.copy()
        for s in order:
            m = subjects == s
            tids = tri_index[s]
            lab = np.asarray([y[m][trials[m] == t][0] for t in tids])
            mapping = dict(zip(tids, rng.permutation(lab)))
            yp[m] = [mapping[t] for t in trials[m]]
        Mp = np.nan_to_num(subject_d_matrix(X, yp, subjects, order))
        idx = np.argsort(-np.mean(np.abs(Mp), 0))[:TOPK]
        idx2 = np.argsort(-np.abs(np.mean(Mp, 0)))[:TOPK]
        null_abs.append(np.mean(np.abs(Mp[:, idx]))); null_sign.append(sign_stats(Mp, idx)); null_sign_signed.append(sign_stats(Mp, idx2))
    res['permutation_null_window_d'] = {'n_perm': N_PERM, 'top30_mean_abs_d': [float(np.mean(null_abs)), float(np.percentile(null_abs, 95))],
                                        'top30_by_mean_abs_sign_agreement': [float(np.mean(null_sign)), float(np.percentile(null_sign, 95))],
                                        'top30_by_abs_mean_sign_agreement': [float(np.mean(null_sign_signed)), float(np.percentile(null_sign_signed, 95))],
                                        'observed_top30_mean_abs_d': float(np.mean(np.abs(D[:, np.argsort(-np.mean(np.abs(D), 0))[:TOPK]]))),
                                        'note': '[mean, 95th percentile] under within-subject trial-label permutation'}
    return res


def main():
    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    st = load_preds(ST_FILES)
    assert st is not None, 'spectral_temporal control OOF missing'
    sp = load_preds(SP_FILES)
    ref = st['fixed002']
    if sp is not None:
        assert sp['fixed002'][KEY].equals(ref[KEY]), 'spectral control rows not aligned'
    y = ref.y_true.to_numpy(); subjects = ref.subject_id.to_numpy(); groups = ref.health_group.to_numpy(); trials = ref.trial_key.to_numpy()
    cnt = ref.groupby(['subject_id', 'trial_key']).y_true.agg(['size', 'nunique'])
    assert (cnt['size'] == 5).all() and (cnt['nunique'] == 1).all() and len(cnt) == 69 * 8
    assert ref.subject_id.nunique() == 69 and (ref.groupby('subject_id').y_true.mean() == 0.5).all()
    metrics = {'n_rows': 2760, 'n_subjects': 69, 'n_HC': int(ref.groupby('subject_id').health_group.first().eq('HC').sum()),
               'n_DEP': int(ref.groupby('subject_id').health_group.first().eq('DEP').sum()), 'trial_rows_check': '69x8 (subject,trial) cells, each 5 rows, constant label'}
    if sp is not None:
        metrics['spectral_only_folds_identical_to_052'] = {s: bool(np.array_equal(sp[s].fold.to_numpy(), st[s].fold.to_numpy())) for s in SPLITS}
    print('alignment checks passed; spectral-only control', 'found' if sp is not None else 'not found', flush=True)

    models = {'spectral_temporal': st, **({'spectral_only': sp} if sp is not None else {})}
    # 1+3: levels and transforms
    lvl_rows, tf_rows, subj = [], [], {}
    for mname, frames in models.items():
        subj[mname] = {}
        for s in SPLITS:
            df = frames[s]; p = df.prob_positive.to_numpy()
            lm, stab, _ = level_metrics(df, p)
            lvl_rows.append({'model': mname, 'split': s, **lm})
            subj[mname][s] = stab.set_index('subject_id')
            for tn, q in transforms(df, p).items():
                row = {'model': mname, 'split': s, 'transform': tn, 'pooled_window_auc': auc(y, q)}
                for g in ('HC', 'DEP'):
                    row[f'{g}_pooled_auc'] = auc(y[groups == g], q[groups == g])
                tr = pd.DataFrame({'s': subjects, 't': trials, 'y': y, 'q': q}).groupby(['s', 't']).mean()
                row['pooled_trial_auc'] = auc(tr.y, tr.q)
                tf_rows.append(row)
    lvl = pd.DataFrame(lvl_rows); lvl.to_csv(OUT / 'levels_window_trial_subject.csv', index=False)
    tf = pd.DataFrame(tf_rows); tf.to_csv(OUT / 'subject_offset_transforms.csv', index=False)
    num = lambda d: d.select_dtypes('number')
    metrics['levels'] = {m: {'per_split': lvl[lvl.model == m].set_index('split').drop(columns='model').to_dict('index'),
                             'mean6': num(lvl[lvl.model == m]).mean().to_dict()} for m in models}
    metrics['offset_transforms'] = {m: tf[tf.model == m].groupby('transform')[['pooled_window_auc', 'HC_pooled_auc', 'DEP_pooled_auc', 'pooled_trial_auc']].mean().to_dict('index') for m in models}
    print(lvl.to_string(index=False)); print(tf.groupby(['model', 'transform'])[['pooled_window_auc', 'HC_pooled_auc', 'DEP_pooled_auc', 'pooled_trial_auc']].mean().to_string(), flush=True)

    # 2: per-subject stability
    stab_out = {}
    for mname in models:
        W = pd.DataFrame({s: subj[mname][s].window_auc for s in SPLITS}); T = pd.DataFrame({s: subj[mname][s].trial_auc for s in SPLITS})
        g = subj[mname]['fixed002'].health_group
        tab = pd.DataFrame({'health_group': g, 'window_auc_mean6': W.mean(1), 'window_auc_std6': W.std(1, ddof=1), 'window_auc_min6': W.min(1), 'window_auc_max6': W.max(1),
                            'n_splits_window_auc_lt_0p5': (W < 0.5).sum(1), 'trial_auc_mean6': T.mean(1), 'n_splits_trial_auc_lt_0p5': (T < 0.5).sum(1)})
        tab = pd.concat([tab, W.add_prefix('window_auc_'), T.add_prefix('trial_auc_')], axis=1)
        tab.sort_values('window_auc_mean6').to_csv(OUT / f'per_subject_auc_{mname}.csv')
        cp = W.corr('pearson').to_numpy(); cs = W.corr('spearman').to_numpy(); iu = np.triu_indices(6, 1)
        o = {'between_split_corr_pearson_mean': float(cp[iu].mean()), 'between_split_corr_pearson_min': float(cp[iu].min()), 'between_split_corr_spearman_mean': float(cs[iu].mean()),
             'trial_between_split_corr_spearman_mean': float(T.corr('spearman').to_numpy()[iu].mean()),
             'mean_within_subject_std_across_splits': float(tab.window_auc_std6.mean()), 'between_subject_std_of_mean_auc': float(tab.window_auc_mean6.std(ddof=1)),
             'icc_like_ratio': float(tab.window_auc_mean6.var(ddof=1) / (tab.window_auc_mean6.var(ddof=1) + (tab.window_auc_std6 ** 2).mean()))}
        for gname, gm in (('all', np.ones(len(tab), bool)), ('HC', tab.health_group == 'HC'), ('DEP', tab.health_group == 'DEP')):
            t = tab[gm]
            o[gname] = {'n': int(len(t)), 'mean_subject_auc': float(t.window_auc_mean6.mean()), 'median_subject_auc': float(t.window_auc_mean6.median()),
                        'consistent_inverted_all6': int((t.n_splits_window_auc_lt_0p5 == 6).sum()), 'inverted_ge4of6': int((t.n_splits_window_auc_lt_0p5 >= 4).sum()),
                        'mean6_auc_lt_0p5': int((t.window_auc_mean6 < 0.5).sum()), 'mean6_auc_lt_0p6': int((t.window_auc_mean6 < 0.6).sum()),
                        'trial_consistent_inverted_all6': int((t.n_splits_trial_auc_lt_0p5 == 6).sum()),
                        'consistent_inverted_ids': t.index[t.n_splits_window_auc_lt_0p5 == 6].tolist()}
        stab_out[mname] = o
        subj[mname]['_tab'] = tab
    if sp is not None:
        a, b = subj['spectral_temporal']['_tab'].window_auc_mean6, subj['spectral_only']['_tab'].window_auc_mean6.reindex(subj['spectral_temporal']['_tab'].index)
        stab_out['cross_model_subject_auc_spearman'] = float(stats.spearmanr(a, b)[0])
        stab_out['cross_model_subject_auc_pearson'] = float(np.corrcoef(a, b)[0, 1])
    metrics['subject_stability'] = stab_out
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk not in ('all', 'HC', 'DEP')}) for k, v in stab_out.items()}, indent=1))
    print(json.dumps({m: {g: {k: v for k, v in stab_out[m][g].items()} for g in ('all', 'HC', 'DEP')} for m in models}, indent=1), flush=True)

    # 4: oracle polarity flip (uses validation labels -> upper bound only) + unlabeled proxies
    tabst = subj['spectral_temporal']['_tab']
    oracle_rows = []
    for s in SPLITS:
        df = st[s]; p = df.prob_positive.to_numpy(); sa = subj['spectral_temporal'][s].window_auc
        inv_split = set(sa.index[sa < 0.5]); inv_cons = set(tabst.index[tabst.n_splits_window_auc_lt_0p5 >= 4])
        cen = transforms(df, p)['subject_centered']
        for tag, inv in (('flip_split_inverted', inv_split), ('flip_ge4of6_inverted', inv_cons)):
            m = np.isin(subjects, list(inv))
            pf = np.where(m, 1 - p, p); cf = np.where(m, -cen, cen)
            oracle_rows.append({'split': s, 'oracle': tag, 'n_flipped': len(inv), 'raw_auc': auc(y, p), 'flipped_auc': auc(y, pf), 'flipped_centered_auc': auc(y, cf),
                                'HC_flipped_auc': auc(y[groups == 'HC'], pf[groups == 'HC']), 'DEP_flipped_auc': auc(y[groups == 'DEP'], pf[groups == 'DEP'])})
        # oracle per-subject perfect-within ranking ceiling of offset removal: macro AUC is the ceiling of any per-subject monotone recalibration
    orc = pd.DataFrame(oracle_rows); orc.to_csv(OUT / 'oracle_polarity_flip.csv', index=False)
    metrics['ORACLE_polarity_flip_UPPER_BOUND_NOT_A_MODEL'] = orc.groupby('oracle').mean(numeric_only=True).to_dict('index')
    print(orc.groupby('oracle').mean(numeric_only=True).to_string(), flush=True)

    prox_list = [unlabeled_proxies(st[s], st[s].prob_positive.to_numpy()).set_index('subject_id') for s in SPLITS]
    prox = sum(prox_list) / len(prox_list)
    prox = prox.join(tabst[['health_group', 'window_auc_mean6']])
    prox.to_csv(OUT / 'unlabeled_proxies_vs_subject_auc.csv')
    pr = {}
    for c in [c for c in prox.columns if c not in ('health_group', 'window_auc_mean6')]:
        rho, pv = stats.spearmanr(prox[c], prox.window_auc_mean6)
        pr[c] = {'rho_all': float(rho), 'p_all': float(pv)}
        for g in ('HC', 'DEP'):
            q = prox[prox.health_group == g]; rho, pv = stats.spearmanr(q[c], q.window_auc_mean6)
            pr[c][f'rho_{g}'] = float(rho); pr[c][f'p_{g}'] = float(pv)
    inv_mask = tabst.n_splits_window_auc_lt_0p5 >= 4
    pr['_proxy_auc_for_detecting_ge4of6_inverted'] = {c: auc(inv_mask.reindex(prox.index).astype(int), -prox[c]) for c in pr if not c.startswith('_')}
    metrics['unlabeled_proxy_spearman_vs_subject_auc'] = pr
    print(json.dumps(pr, indent=1), flush=True)

    # 5: feature direction consistency
    metrics['feature_effect_direction'] = feature_audit(subjects, groups, y, trials, tabst.window_auc_mean6)
    fe = metrics['feature_effect_direction']
    print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else {a: b for a, b in vv.items() if a != 'features'}) for kk, vv in v.items()} if isinstance(v, dict) else v for k, v in fe.items()}, indent=1), flush=True)

    # 6: trial-id heterogeneity (stimulus identity uncertain)
    lab = ref.groupby(['trial_key']).y_true.agg(['mean', 'nunique'])
    metrics['trial_id_label_map'] = {int(k): {'mean_label': float(v['mean']), 'label_constant_across_subjects': bool(v['nunique'] == 1)} for k, v in lab.iterrows()}
    pbar = np.mean([st[s].prob_positive.to_numpy() for s in SPLITS], 0)
    tdf = ref[KEY + ['trial_key']].assign(p=pbar, pc=transforms(ref, pbar)['subject_centered']).groupby(['subject_id', 'health_group', 'trial_key'], sort=False).agg(y=('y_true', 'first'), p=('p', 'mean'), pc=('pc', 'mean')).reset_index()
    tdf['correct_0p5'] = ((tdf.p > 0.5).astype(int) == tdf.y).astype(int)
    tdf['correct_centered'] = ((tdf.pc > 0).astype(int) == tdf.y).astype(int)
    tdf.to_csv(OUT / 'trial_level_predictions_mean6.csv', index=False)
    trows = []
    for (tid, g), q in list(tdf.groupby(['trial_key', 'health_group'])) + [((t, 'ALL'), q) for t, q in tdf.groupby('trial_key')]:
        trows.append({'trial_key': int(tid), 'group': g, 'n_subjects': len(q), 'frac_label_positive': float(q.y.mean()), 'acc_0p5': float(q.correct_0p5.mean()),
                      'acc_centered': float(q.correct_centered.mean()), 'mean_prob': float(q.p.mean())})
    tt = pd.DataFrame(trows).sort_values(['group', 'trial_key']); tt.to_csv(OUT / 'trial_id_accuracy.csv', index=False)
    het = {}
    for g in ('ALL', 'HC', 'DEP'):
        q = tdf if g == 'ALL' else tdf[tdf.health_group == g]
        ct = pd.crosstab(q.trial_key, q.correct_centered)
        chi = stats.chi2_contingency(ct)
        het[g] = {'chi2': float(chi[0]), 'dof': int(chi[2]), 'p': float(chi[1])}
        # within-label heterogeneity (trial id confounded with label if label map is constant)
        for lv in (0, 1):
            qq = q[q.y == lv]; ct2 = pd.crosstab(qq.trial_key, qq.correct_centered)
            if ct2.shape[0] > 1 and ct2.shape[1] > 1:
                c2 = stats.chi2_contingency(ct2); het[g][f'within_label{lv}'] = {'chi2': float(c2[0]), 'dof': int(c2[2]), 'p': float(c2[1])}
    # per-subject: number of the 8 trials misclassified (centered) -- concentrated vs diffuse errors
    per_sub_err = tdf.groupby('subject_id').correct_centered.apply(lambda c: int(8 - c.sum()))
    het['n_wrong_trials_per_subject_hist'] = {int(k): int(v) for k, v in per_sub_err.value_counts().sort_index().items()}
    # binomial expectation if each trial independently wrong with overall rate
    pw = 1 - tdf.correct_centered.mean()
    het['binomial_expected_hist'] = {k: float(69 * stats.binom.pmf(k, 8, pw)) for k in range(9)}
    het['overdispersion_var_ratio'] = float(per_sub_err.var(ddof=1) / (8 * pw * (1 - pw)))
    metrics['trial_id_heterogeneity'] = het
    print(tt.to_string(index=False)); print(json.dumps(het, indent=1), flush=True)

    # 7: subject-cluster bootstrap on fixed002 control
    rng = np.random.default_rng(SEED)
    p = st['fixed002'].prob_positive.to_numpy()
    uniq = pd.unique(subjects); idx_of = {s: np.flatnonzero(subjects == s) for s in uniq}
    ug = np.asarray([groups[idx_of[s][0]] for s in uniq])
    hc_s, dep_s = uniq[ug == 'HC'], uniq[ug == 'DEP']
    boots, gaps = [], []
    for _ in range(N_BOOT):
        pick = rng.choice(uniq, len(uniq), replace=True); ii = np.concatenate([idx_of[s] for s in pick]); boots.append(auc(y[ii], p[ii]))
        h = np.concatenate([idx_of[s] for s in rng.choice(hc_s, len(hc_s), replace=True)])
        d = np.concatenate([idx_of[s] for s in rng.choice(dep_s, len(dep_s), replace=True)])
        gaps.append(auc(y[h], p[h]) - auc(y[d], p[d]))
    m = groups == 'HC'
    metrics['bootstrap_fixed002'] = {'n_boot': N_BOOT, 'seed': SEED, 'scheme': 'subject-cluster; gap resampled stratified within group',
                                     'pooled_auc': auc(y, p), 'pooled_auc_ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                                     'hc_minus_dep_auc': auc(y[m], p[m]) - auc(y[~m], p[~m]), 'gap_ci95': [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
                                     'frac_gap_le_0': float(np.mean(np.asarray(gaps) <= 0))}
    print(json.dumps(metrics['bootstrap_fixed002'], indent=1))
    metrics['elapsed_seconds'] = time.perf_counter() - t0
    metrics['inputs'] = {k: str(v) for k, v in {**{f'st_{s}': f for s, f in ST_FILES.items()}, **{f'sp_{s}': f for s, f in SP_FILES.items()}}.items()}
    (HERE / 'metrics.json').write_text(json.dumps(metrics, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
    print(f'done in {metrics["elapsed_seconds"]:.1f}s', flush=True)


if __name__ == '__main__':
    main()
