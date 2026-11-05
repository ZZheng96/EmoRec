import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from scipy.special import expit
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from emo_rec.normalization import normalize_by_subject

CFG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')


def model(kind, fold):
    if kind == 'lr':
        estimator = LogisticRegression(C=CFG['linear_C'], max_iter=CFG['max_iter'], random_state=CFG['seed'] + fold)
    elif kind == 'svc':
        estimator = LinearSVC(C=CFG['linear_C'], dual=False, max_iter=CFG['max_iter'], random_state=CFG['seed'] + fold)
    else:
        estimator = HistGradientBoostingClassifier(**CFG['hgb'], random_state=CFG['seed'] + fold)
    return make_pipeline(StandardScaler(), estimator)


def predict(fitted, X):
    if isinstance(fitted[-1], LinearSVC):
        return expit(fitted.decision_function(X))
    return fitted.predict_proba(X)[:, 1]


def metric(y, p):
    return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, p >= CFG['threshold'], zero_division=0))}


def main():
    start = time.perf_counter()
    for folder in ('predictions', 'input_metadata'):
        (HERE / folder).mkdir(exist_ok=True)
    paths = {k: (HERE / CFG[k]).resolve() for k in ('de_cache', 'spectral_cache', 'temporal_cache', 'fixed_reference', 'baseline_oof')}
    inputs = {}
    for key in ('de_cache', 'spectral_cache', 'temporal_cache'):
        with np.load(paths[key], allow_pickle=False) as z:
            inputs[key] = {k: z[k] for k in ('X_train', 'subject_ids', 'health_groups', 'original_trial_ids', 'y_train')}
    d, s, t = [inputs[k] for k in ('de_cache', 'spectral_cache', 'temporal_cache')]
    for other in (s, t):
        for key in ('subject_ids', 'health_groups', 'original_trial_ids', 'y_train'):
            assert np.array_equal(d[key], other[key]), key
    keep = d['subject_ids'] != CFG['excluded_subject']
    subs, groups, y = [d[k][keep] for k in ('subject_ids', 'health_groups', 'y_train')]
    ref = pd.read_csv(paths['fixed_reference'])
    keys = ['subject_id', 'health_group', 'original_trial_id', 'y_true', 'fold']
    for col, arr in [('subject_id', subs), ('health_group', groups), ('original_trial_id', d['original_trial_ids'][keep]), ('y_true', y)]:
        assert np.array_equal(ref[col].to_numpy(), arr), col
    ref = ref[keys].copy()
    ref['row_index'] = np.arange(len(ref))
    ref['window_in_subject'] = ref.groupby('subject_id', sort=False).cumcount()
    ref['physical_trial_index'] = ref.window_in_subject // 5
    ref['window_in_trial'] = ref.window_in_subject % 5
    assert len(y) == 2760 and len(np.unique(subs)) == 69
    for subject in np.unique(subs):
        ix = subs == subject
        assert ix.sum() == 40 and np.array_equal(y[ix], np.repeat([0, 1], 20))
        assert np.array_equal(ref.original_trial_id.to_numpy()[ix], np.tile(np.repeat(np.arange(1, 5), 5), 2))
        assert ref.loc[ix, 'fold'].nunique() == 1
        assert len(np.unique(groups[ix])) == 1
    folds = ref.fold.to_numpy()
    assert set(folds) == set(range(5))
    zscore = lambda x: normalize_by_subject(x, subs, 'zscore')
    X = {'de': zscore(d['X_train'][keep]), 'spectral': zscore(s['X_train'][keep, :150]), 'temporal': zscore(t['X_train'][keep])}
    assert X['de'].shape[1] == 150 and all(np.isfinite(a).all() for a in X.values())
    subjects = np.unique(subs)
    raw = d['X_train'][keep]
    descriptors = np.stack([np.r_[raw[subs == subject].mean(0), np.log(np.maximum(raw[subs == subject].std(0), 1e-8))] for subject in subjects])
    subject_folds = np.array([folds[subs == subject][0] for subject in subjects])
    names = ['de_lr', 'de_svc', 'de_hgb', 'hcdep_lr', 'hcdep_svc', 'hcdep_hgb', 'prototype_lr', 'prototype_hgb', 'spectral_svc', 'temporal_svc', 'de_spectral_temporal_score_fusion']
    predictions = {name: np.full(len(y), np.nan) for name in names}
    audits, routes, scales = [], [], []
    with threadpool_limits(limits=CFG['threads']):
        for fold in range(5):
            tr, va = folds != fold, folds == fold
            assert not set(subs[tr]) & set(subs[va])
            train_trials = set(zip(subs[tr], ref.physical_trial_index.to_numpy()[tr]))
            valid_trials = set(zip(subs[va], ref.physical_trial_index.to_numpy()[va]))
            assert not train_trials & valid_trials
            print(f'fold {fold + 1}/5: train {len(set(subs[tr]))} subjects / valid {len(set(subs[va]))}', flush=True)
            for kind in ('lr', 'svc', 'hgb'):
                print(f'  DE unified + HC/DEP experts: {kind}', flush=True)
                fitted = model(kind, fold).fit(X['de'][tr], y[tr])
                predictions['de_' + kind][va] = predict(fitted, X['de'][va])
                for group in ('HC', 'DEP'):
                    ti, vi = tr & (groups == group), va & (groups == group)
                    expert = model(kind, fold).fit(X['de'][ti], y[ti])
                    predictions['hcdep_' + kind][vi] = predict(expert, X['de'][vi])
            st = subject_folds != fold
            scaler = StandardScaler().fit(descriptors[st])
            km = KMeans(n_clusters=CFG['clusters'], n_init=20, random_state=CFG['seed'] + fold).fit(scaler.transform(descriptors[st]))
            assignments = km.predict(scaler.transform(descriptors))
            lookup = dict(zip(subjects, assignments))
            route = np.array([lookup[sid] for sid in subs])
            audit = {'fold': fold, 'train_subjects': subjects[st].tolist(), 'valid_subjects': subjects[~st].tolist(), 'descriptor_scaler_mean': scaler.mean_.tolist(), 'descriptor_scaler_scale': scaler.scale_.tolist(), 'cluster_centers': km.cluster_centers_.tolist(), 'physical_trial_intersection': 0, 'subject_intersection': 0}
            audits.append(audit)
            for sid, cluster in zip(subjects, assignments):
                routes.append({'fold': fold, 'subject_id': sid, 'role': 'train' if sid in set(subs[tr]) else 'validation', 'cluster': int(cluster)})
            for kind in ('lr', 'hgb'):
                print(f'  training-fold prototype experts: {kind}', flush=True)
                for cluster in range(CFG['clusters']):
                    ti, vi = tr & (route == cluster), va & (route == cluster)
                    assert ti.sum() > 0 and len(np.unique(y[ti])) == 2
                    expert = model(kind, fold).fit(X['de'][ti], y[ti])
                    if vi.any():
                        predictions['prototype_' + kind][vi] = predict(expert, X['de'][vi])
            fused = np.zeros(va.sum())
            for weight, block in zip(CFG['fusion_weights'], ('de', 'spectral', 'temporal')):
                fitted = model('svc', fold).fit(X[block][tr], y[tr])
                train_score = fitted.decision_function(X[block][tr])
                valid_score = fitted.decision_function(X[block][va])
                center, scale = float(train_score.mean()), max(float(train_score.std()), 1e-8)
                fused += weight * (valid_score - center) / scale
                scales.append({'fold': fold, 'block': block, 'train_score_mean': center, 'train_score_std': scale, 'weight': weight})
                if block != 'de':
                    predictions[block + '_svc'][va] = expit(valid_score)
            predictions['de_spectral_temporal_score_fusion'][va] = expit(fused)
            print(f'fold {fold + 1}/5 complete ({time.perf_counter() - start:.1f}s)', flush=True)
    baseline = pd.read_csv(paths['baseline_oof'])
    assert np.array_equal(ref[keys].to_numpy(), baseline[keys].to_numpy())
    predictions['baseline_057_plus_pe'] = baseline.prob_positive.to_numpy()
    rows, details = [], {}
    for name, p in predictions.items():
        assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
        frame = ref.assign(prob_positive=p)
        trial = frame.groupby(['subject_id', 'health_group', 'physical_trial_index', 'fold'], as_index=False).agg(y_true=('y_true', 'first'), prob_positive=('prob_positive', 'mean'))
        overall = metric(y, p)
        row = {'method': name, **overall}
        for group in ('HC', 'DEP'):
            mask = groups == group
            row.update({group.lower() + '_' + k: v for k, v in metric(y[mask], p[mask]).items()})
        row.update({'trial_' + k: v for k, v in metric(trial.y_true.to_numpy(), trial.prob_positive.to_numpy()).items()})
        subject_metrics = [{'subject_id': sid, **metric(y[subs == sid], p[subs == sid])} for sid in subjects]
        row['subject_macro_auc'] = float(np.mean([m['auc'] for m in subject_metrics]))
        row['subject_macro_f1'] = float(np.mean([m['f1'] for m in subject_metrics]))
        rows.append(row)
        details[name] = {'summary': row, 'subjects': subject_metrics, 'folds': [{'fold': f, **metric(y[folds == f], p[folds == f])} for f in range(5)]}
        frame.to_csv(HERE / f'predictions/{name}_oof.csv', index=False)
        trial.to_csv(HERE / f'predictions/{name}_trials.csv', index=False)
    ref.to_csv(HERE / 'input_metadata/folds.csv', index=False)
    pd.DataFrame(routes).to_csv(HERE / 'input_metadata/prototype_routes.csv', index=False)
    pd.DataFrame(scales).to_csv(HERE / 'input_metadata/fusion_score_scaling.csv', index=False)
    write_json(HERE / 'input_metadata/prototypes.json', audits)
    pd.DataFrame(descriptors, index=subjects).rename_axis('subject_id').to_csv(HERE / 'input_metadata/subject_descriptors.csv')
    source_paths = list(paths.values()) + [Path(__file__), HERE / 'config.json', ROOT / 'src/emo_rec/data.py', ROOT / 'src/emo_rec/normalization.py', ROOT / '数据/数据集说明文档.md', ROOT / '排行.md', HERE.parent / '199_de_plus_pe/run_experiment.py']
    write_json(HERE / 'input_metadata/provenance.json', {'inputs': {str(p.relative_to(ROOT)): sha(p) for p in source_paths}, 'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__}, 'normalization': CFG['normalization'], 'validation_labels_used_for_selection_or_routing': False, 'public_arrays_loaded': False, 'historical_095_results_used': False, 'de_feature_order': 'channel-major, band-minor from 199 reshape; cached feature_names not used', 'physical_trial_identity_source': 'documented loader row order, 5 successive windows per physical trial', 'health_routing': 'observed HC/DEP metadata, not emotion labels; public deployment requires same metadata'})
    elapsed = time.perf_counter() - start
    write_json(HERE / 'metrics.json', {'config': CFG, 'runs': details, 'elapsed_seconds': elapsed, 'strict_subjectwise_oof': True, 'all_physical_trial_splits_disjoint': True, 'public_auc': None, 'public_f1': None})
    pd.DataFrame(rows).to_csv(HERE / 'summary.csv', index=False)
    table = ['|方法|AUC|F1|HC AUC/F1|DEP AUC/F1|trial AUC/F1|', '|---|---:|---:|---|---|---|']
    for r in rows:
        table.append(f"|{r['method']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['hc_auc']:.6f}/{r['hc_f1']:.6f}|{r['dep_auc']:.6f}/{r['dep_f1']:.6f}|{r['trial_auc']:.6f}/{r['trial_f1']:.6f}|")
    by_name = {r['method']: r for r in rows}
    comparisons = [('hcdep_lr', 'de_lr'), ('hcdep_svc', 'de_svc'), ('hcdep_hgb', 'de_hgb'), ('prototype_lr', 'de_lr'), ('prototype_hgb', 'de_hgb'), ('de_spectral_temporal_score_fusion', 'de_svc')]
    lines = ['# 实验207：排行榜方法迁移的跨被试低成本对照', '', '## 协议与边界', '', '排行榜仅提供方法线索，不是可复现架构或成绩证据。本轮固定使用002/057同一subject-wise 5-fold，排除历史预先排除的HC43，共69人（44 HC/25 DEP）、2760个10秒窗口、552个物理trial。所有模型重新拟合，不使用199拼接历史OOF；057仅作逐键对齐的冻结基线。每名被试只出现于一个验证折，物理trial的5窗口不跨折。', 'DE复用199缓存：逐通道5频带滤波方差的高斯微分熵。DE/频谱/时域采用整被试无标签zscore；验证被试仅用自身40窗口归一化，属于transductive covariate访问，不是完全inductive单窗口协议。模型StandardScaler、专家及原型的缩放/KMeans仅训练fold拟合。没有外层调参或权重/阈值选择；C=.1，HGB沿用固定参数，阈值=.5。', '健康专家按已知HC/DEP元数据路由，不根据情绪标签、个体历史表现或验证得分选择；公开部署前必须确认健康元数据可用，不能根据P_test编号猜组。原型用原始DE的被试均值和log标准差（对角方差统计），训练被试StandardScaler+KMeans(k=2)，验证被试以自身无标签统计分配最近中心。未读取095结果。', '多特征SVM是DE、频谱150维、时域450维三个LinearSVC的分数融合；每块分数均值/标准差仅从该外层训练fold拟合后的训练分数计算，固定等权平均再sigmoid；不是真正MKL，sigmoid分数也不是校准概率。各单块对照保留原始margin的sigmoid；跨折汇总AUC可能受分数尺度影响，折级和被试级指标亦提供。', '传统sklearn模型CPU单线程，无GPU适用模块；复用缓存避免重复滤波。公开数组未加载、未生成公开预测、公开真实AUC/F1未知。', '', '## 全量结果', ''] + table
    lines += ['', '## 相对统一模型的预设对照（AUC/F1差）', '']
    for candidate, control in comparisons:
        a, b = by_name[candidate], by_name[control]
        lines.append(f"- {candidate} vs {control}: ΔAUC={a['auc'] - b['auc']:+.6f}, ΔF1={a['f1'] - b['f1']:+.6f}；{'AUC有增益' if a['auc'] > b['auc'] else 'AUC未改善'}。")
    best = max(rows, key=lambda r: r['auc'])
    lines += ['', f"本轮事后最高窗口AUC来自{best['method']}：AUC={best['auc']:.6f}，F1={best['f1']:.6f}，仅是描述性比较，不构成新选择器。", '单次固定5折是开发对照，不提供独立重复稳定性或显著性证据；反复使用同一开发折可能有研究者选择偏差。训练每人40窗口与公开每人8窗口存在长度/分布差异。未达0.95不得宣称达标；205/206为有标签被试内物理trial验证，不能解释为公开跨被试泛化。202历史保守融合OOF 0.803836/0.726616亦不是公开成绩；本轮不更换公开提交候选。', 'input_metadata保存完整折分配、原型及训练缩放参数、每折训练/验证路由、分数缩放和输入SHA256；predictions保存全部窗口和trial均值预测；metrics包含逐折/逐被试结果。原始数据未改写。', f'总耗时：{elapsed:.2f}秒。']
    (HERE / '实验报告.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)
    print(f'COMPLETE 5/5 folds; {elapsed:.1f}s; subject/trial isolation assertions passed', flush=True)


if __name__ == '__main__':
    main()
