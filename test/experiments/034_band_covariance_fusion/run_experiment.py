import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import expm, fractional_matrix_power
from scipy.signal import butter, sosfiltfilt
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from emo_rec.data import load_public_test_segments, load_train_segments
from emo_rec.features import FREQUENCY_BANDS
from emo_rec.normalization import normalize_by_subject
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = HERE / 'cache' / 'band_covariance.npz'
META_KEYS = ('subject_ids', 'health_groups', 'original_trial_ids', 'y_train', 'public_user_ids', 'public_trial_ids')
TRIU = np.triu_indices(30)


def sha256(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def load_cache(path):
    with np.load(path, allow_pickle=False) as d: return {k: d[k] for k in d.files}


def covariance_features(segments, config):
    x = np.asarray(segments, dtype=np.float64)
    if x.ndim != 3 or x.shape[1:] != (30, 2500) or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError('Expected finite nonempty EEG shape (n,30,2500)')
    feats = []
    covs = []
    for low, high in config['bands']:
        sos = butter(config['filter']['order'], [low, high], btype='bandpass', fs=config['sampling_rate'], output='sos')
        filtered = sosfiltfilt(sos, x, axis=-1)
        centered = filtered - filtered.mean(axis=-1, keepdims=True)
        c = np.einsum('nct,ndt->ncd', centered, centered) / (centered.shape[-1] - 1)
        c = c + np.eye(30)[None] * config['covariance']['regularization']
        c /= np.trace(c, axis1=1, axis2=2)[:, None, None]
        covs.append(c)
    return np.asarray(covs).transpose(1, 0, 2, 3)


def matrix_log(c):
    vals, vecs = np.linalg.eigh(c)
    return (vecs * np.log(np.maximum(vals, 1e-15))[None, :]) @ vecs.T


def tangent_features(covs, centers):
    out = []
    for band in range(covs.shape[1]):
        center = centers[band]
        invsqrt = fractional_matrix_power(center, -0.5).real
        vals = []
        for c in covs[:, band]:
            vals.append(matrix_log(invsqrt @ c @ invsqrt)[TRIU])
        out.append(np.asarray(vals))
    result = np.concatenate(out, axis=1)
    if result.shape[1] != 5 * 465 or not np.isfinite(result).all(): raise ValueError('Invalid covariance features')
    return result


def assert_alignment(spectral, cov):
    for key in META_KEYS:
        if not np.array_equal(spectral[key], cov[key]): raise ValueError(f'Cache metadata mismatch: {key}')
    if cov['cov_train'].shape[1:] != (5, 30, 30) or cov['cov_public'].shape[1:] != (5, 30, 30): raise ValueError('Invalid covariance shape')
    if not np.isfinite(cov['cov_train']).all() or not np.isfinite(cov['cov_public']).all(): raise ValueError('Non-finite covariance')


def build_cache(spectral, config):
    train_cov=[]; labels=[]; subjects=[]; groups=[]; trials=[]; public_cov=[]; users=[]; ptrials=[]; sources=[]
    paths=sorted((ROOT/'数据'/'训练集').glob('*/*.mat'))
    for i,path in enumerate(paths):
        seg,y,ids=load_train_segments(path); train_cov.append(covariance_features(seg,config)); labels.extend(y); subjects.extend([path.stem]*len(y)); groups.extend(['DEP' if path.parent.name=='抑郁症患者' else 'HC']*len(y)); trials.extend(ids); sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path)}); print(f'Covariance {i+1}/{len(paths)}: {path.stem}',flush=True)
    paths=sorted((ROOT/'数据'/'公开测试集').glob('*.mat'),key=lambda p:int(p.stem.replace('P_test','')))
    for path in paths:
        seg=load_public_test_segments(path); public_cov.append(covariance_features(seg,config)); users.extend([path.stem]*len(seg)); ptrials.extend(range(1,len(seg)+1)); sources.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path)})
    data={'cov_train':np.concatenate(train_cov),'y_train':np.asarray(labels),'subject_ids':np.asarray(subjects),'health_groups':np.asarray(groups),'original_trial_ids':np.asarray(trials),'cov_public':np.concatenate(public_cov),'public_user_ids':np.asarray(users),'public_trial_ids':np.asarray(ptrials),'metadata_json':np.asarray(json.dumps({'config':config,'sources':sources,'spectral_sha256':sha256(HERE/config['spectral_cache'])},ensure_ascii=False))}
    assert_alignment(spectral,data); CACHE.parent.mkdir(exist_ok=True); np.savez_compressed(CACHE,**data); return data


def load_reference(spectral,config):
    keep=spectral['subject_ids']!='HC43'; frame=pd.read_csv(HERE/config['reference_oof'])
    for col,key in [('subject_id','subject_ids'),('health_group','health_groups'),('original_trial_id','original_trial_ids'),('y_true','y_train')]:
        if not np.array_equal(frame[col].to_numpy(),spectral[key][keep]): raise ValueError(f'OOF alignment mismatch: {col}')
    if len(frame)!=2760 or frame.subject_id.nunique()!=69: raise ValueError('Unexpected reference rows')
    assert_subject_isolation(frame); return keep,frame.drop(columns='prob_positive')

def assert_subject_isolation(frame):
    if set(frame.fold)!=set(range(5)) or frame.groupby('subject_id').fold.nunique().max()!=1: raise ValueError('Invalid subject folds')

def metrics(y,p): return {'auc':float(roc_auc_score(y,p)),'f1':float(f1_score(y,p>=.5)),'balanced_accuracy':float(balanced_accuracy_score(y,p>=.5))}
def summarize(frame,p):
    result={'overall':metrics(frame.y_true,p),'folds':{},'health_groups':{},'subjects':{}}
    for f in range(5): result['folds'][str(f)]=metrics(frame.loc[frame.fold==f,'y_true'],p[frame.fold.to_numpy()==f])
    for g in ('HC','DEP'):
        m=frame.health_group.to_numpy()==g; result['health_groups'][g]=metrics(frame.loc[m,'y_true'],p[m])
    for s in frame.subject_id.unique():
        m=frame.subject_id.to_numpy()==s; result['subjects'][s]=metrics(frame.loc[m,'y_true'],p[m])
    result['subject_auc_mean']=float(np.mean([v['auc'] for v in result['subjects'].values()])); return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--rebuild-features',action='store_true'); args=ap.parse_args(); started=time.perf_counter(); config=json.loads((HERE/'config.json').read_text(encoding='utf-8')); spectral=load_cache(HERE/config['spectral_cache']); cov=build_cache(spectral,config) if args.rebuild_features or not CACHE.exists() else load_cache(CACHE); assert_alignment(spectral,cov); keep,reference=load_reference(spectral,config); output=HERE/'predictions'; output.mkdir(exist_ok=True); y=reference.y_true.to_numpy(); folds=reference.fold.to_numpy(); runs={}; summary=[]
    with threadpool_limits(limits=1):
        for name in ('spectral_control','band_covariance','fixed_05_05'):
            oof=np.full(len(y),np.nan); public_parts=[]
            for fold in range(5):
                train=folds!=fold; valid=~train
                centers=[]
                for b in range(5):
                    logs=np.asarray([matrix_log(v) for v in cov['cov_train'][keep][train,b]]); centers.append(expm(logs.mean(axis=0)).real)
                centers=np.asarray(centers)
                Xc=tangent_features(cov['cov_train'][keep],centers); Xcp=tangent_features(cov['cov_public'],centers)
                Xs=normalize_by_subject(spectral['X_train'][keep,:150],spectral['subject_ids'][keep],'zscore'); Xsp=normalize_by_subject(spectral['X_public'][:,:150],spectral['public_user_ids'],'zscore')
                hgb=make_pipeline(StandardScaler(),HistGradientBoostingClassifier(**config['hgb'],random_state=2026+fold)); hgb.fit(Xs[train],y[train]); ps=hgb.predict_proba(Xs[valid])[:,1]
                lr=make_pipeline(StandardScaler(),LogisticRegression(**config['logistic'],random_state=2026+fold)); lr.fit(Xc[train],y[train]); pc=lr.predict_proba(Xc[valid])[:,1]
                if name=='spectral_control': oof[valid]=ps
                elif name=='band_covariance': oof[valid]=pc
                else: oof[valid]=(ps+pc)/2
                print(f'{name}: fold {fold+1}/5',flush=True)
            centers=[]
            for b in range(5):
                logs=np.asarray([matrix_log(v) for v in cov['cov_train'][keep][:, b]]); centers.append(expm(logs.mean(axis=0)).real)
            centers=np.asarray(centers); Xc=tangent_features(cov['cov_train'][keep],centers); Xcp=tangent_features(cov['cov_public'],centers); Xs=normalize_by_subject(spectral['X_train'][keep,:150],spectral['subject_ids'][keep],'zscore'); Xsp=normalize_by_subject(spectral['X_public'][:,:150],spectral['public_user_ids'],'zscore')
            hgb=make_pipeline(StandardScaler(),HistGradientBoostingClassifier(**config['hgb'],random_state=2026)); hgb.fit(Xs,y); ps=hgb.predict_proba(Xsp)[:,1]; lr=make_pipeline(StandardScaler(),LogisticRegression(**config['logistic'],random_state=2026)); lr.fit(Xc,y); pc=lr.predict_proba(Xcp)[:,1]; pp=ps if name=='spectral_control' else pc if name=='band_covariance' else (ps+pc)/2
            frame=reference.copy(); frame['prob_positive']=oof; frame.to_csv(output/f'{name}_oof.csv',index=False); public=pd.DataFrame({'user_id':cov['public_user_ids'],'trial_id':cov['public_trial_ids'],'prob. of positive emotion':pp}); validate_submission(public,[f'P_test{i}' for i in range(1,11)]); public.to_csv(output/f'{name}_public.csv',index=False); runs[name]=summarize(reference,oof); summary.append({'run_name':name,**runs[name]['overall'],'subject_auc_mean':runs[name]['subject_auc_mean'],**{f'{g.lower()}_{k}':v for g,m in runs[name]['health_groups'].items() for k,v in m.items()}})
    result={'config':config,'runs':runs,'elapsed_seconds':time.perf_counter()-started,'provenance':{'spectral_sha256':sha256(HERE/config['spectral_cache']),'covariance_sha256':sha256(CACHE),'reference_sha256':sha256(HERE/config['reference_oof']),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__},'protocol':{'cache_alignment_checked':True,'subject_isolation_checked':True,'train_rows':2760,'public_rows':80,'excluded_subject':'HC43','fixed_probability_average':True,'train_fold_centers':True,'unlabeled_subject_zscore':True}}
    (HERE/'metrics.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8'); pd.DataFrame(summary).to_csv(HERE/'summary.csv',index=False); base=runs['spectral_control']['overall']['auc']; report='# 实验034：五频带跨通道协方差切空间与频谱HGB固定融合\n\n严格复用001频谱缓存、原始EEG读取和002固定5个被试级外层折，排除HC43。每个10秒窗口对五频带Butterworth SOS零相位滤波，计算30×30迹归一化且加1e-6正则协方差；每折仅用训练折协方差的log-Euclidean几何中心，将各频带log(C)的上三角（465维）拼接为2325维特征。分类器、权重和阈值均预注册固定，subject z-score仅无标签按被试样本计算。\n\n## 实测结果\n\n|方法|AUC|F1|BAcc|HC AUC|HC F1|DEP AUC|DEP F1|\n|---|---:|---:|---:|---:|---:|---:|---:|\n';
    for r in summary: report+=f"|{r['run_name']}|{r['auc']:.6f}|{r['f1']:.6f}|{r['balanced_accuracy']:.6f}|{r['hc_auc']:.6f}|{r['hc_f1']:.6f}|{r['dep_auc']:.6f}|{r['dep_f1']:.6f}|\n"
    report+=f'\n相对spectral_control，band_covariance AUC差为{runs["band_covariance"]["overall"]["auc"]-base:+.6f}，fixed_05_05差为{runs["fixed_05_05"]["overall"]["auc"]-base:+.6f}。这是复用002折的探索性比较；如低于对照，不解释为优势。\n'; (HERE/'实验报告.md').write_text(report,encoding='utf-8'); print(pd.DataFrame(summary).to_string(index=False),flush=True)
if __name__=='__main__': main()
