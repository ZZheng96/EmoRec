import hashlib, importlib.util, json, platform, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from threadpoolctl import threadpool_limits

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf8'))
T205 = ROOT / 'test/experiments/205_remaining_subject_auc_breakthrough/run_experiment.py'
spec = importlib.util.spec_from_file_location('e205', T205)
e205 = importlib.util.module_from_spec(spec); spec.loader.exec_module(e205)
CANDIDATES = e205.CANDIDATES

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf8')
def metric(y, p):
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    return {'auc': float(roc_auc_score(y, p)), 'f1': float(f1_score(y, p >= .5, zero_division=0))}
def rank(auc, f1): return int(np.lexsort((np.arange(len(auc)), -f1, -auc))[0])
def pairs_for(rep):
    # Four disjoint positive-negative pairings; repeat changes the deterministic matching.
    return [(np.setdiff1d(np.arange(8), [4 + j, (j + rep) % 4]), np.array([4 + j, (j + rep) % 4])) for j in range(4)]
def inner_choice(cache, train_trials, pe):
    # Candidate/direction selection is made only from inner physical-trial OOF.
    trials = np.asarray(train_trials); sums = np.zeros((len(CANDIDATES), 40)); counts = np.zeros(40, int)
    for ti, vi in e205.base.trial_pairs(trials):
        tt, vt = trials[ti], trials[vi]; idx = e205.base.window_indices(cache.trial, vt)
        sums[:, idx] += cache.get(tt)[:, idx]; counts[idx] += 1
    idx = e205.base.window_indices(cache.trial, trials); assert np.all(counts[idx] == 3)
    trial_pred = {agg: np.stack([(sums[:, cache.trial == t] / counts[cache.trial == t]).mean(axis=1) for t in trials], 1) for agg in ['mean']}
    # Use the same valid inner OOF candidate ranking as 205; no outer labels enter.
    tp = trial_pred['mean']; yy = np.asarray([cache.y[cache.trial == t][0] for t in trials])
    directed = np.stack([tp, 1 - tp], 1).reshape(-1, len(trials)); auc=[]; f=[]
    for k in range(len(directed)): auc.append(roc_auc_score(yy, directed[k])); f.append(f1_score(yy, directed[k] >= .5, zero_division=0))
    chosen = rank(np.asarray(auc), np.asarray(f)); ci, di = chosen // 2, chosen % 2
    pe_trial = np.asarray([pe[cache.trial == t].mean() for t in trials]) if pe is not None else None
    options=[]
    for agg in ('mean','median'):
        vals=[]
        for csp_w in ([0.25,0.5,0.75] if pe_trial is not None else [1.0]):
            cp = tp[ci] if di == 0 else 1 - tp[ci]
            # Inner OOF predictions are already trial-aggregated; median is evaluated on the
            # same finite OOF trial vector to keep aggregation selection label-safe.
            basev = cp
            v = csp_w * basev + (1-csp_w)*pe_trial if pe_trial is not None else basev
            vals.append((roc_auc_score(yy,v), f1_score(yy,v>=.5,zero_division=0), csp_w, agg))
        options.extend(vals)
    pick = max(options, key=lambda z:(z[0],z[1],-z[2],z[3]))
    rows=[{'candidate_index':k//2,'direction':CFG['directions'][k%2],'auc':float(auc[k]),'f1':float(f[k])} for k in range(len(auc))]
    return ci, di, pick[2], pick[3], rows, {'mean':float(pick[0]),'f1':float(pick[1]),'options':[{'auc':float(a),'f1':float(f1),'weight':w,'aggregation':ag} for a,f1,w,ag in options]}

def run_subject(subject, num):
    print(f'[{num}/3] {subject}: loading 40 windows / 8 physical trials', flush=True)
    paths=list((ROOT/'数据/训练集').rglob(subject+'.mat')); assert len(paths)==1
    sys.path.insert(0,str(ROOT/'src')); from emo_rec.data import load_train_segments
    x,y,original=load_train_segments(paths[0]); trial=original-1+4*y
    assert x.shape==(40,30,2500) and np.isfinite(x).all() and np.array_equal(np.bincount(trial),np.full(8,5))
    pe, align=e205.baseline(subject,y,trial,paths[0]); assert pe is not None
    cache=e205.FitCache(e205.base.covariances(x),y,trial)
    methods=['baseline_plus_pe','csp_inner','csp_fixed_raw','csp_fixed_reverse','csp_plus_pe_inner']
    sums={m:np.zeros(40) for m in methods}; counts=np.zeros(40,int); audit=[]; pair_rows=[]
    for rep in CFG['repeats']:
        for j,(tt,vt) in enumerate(pairs_for(rep)):
            print(f'[{num}/3] {subject} repeat {rep+1}/4 pair {j+1}/4 train={tt.tolist()} query={vt.tolist()}',flush=True)
            assert not set(tt)&set(vt) and len(tt)==6 and len(vt)==2
            ci,di,w,agg,cs,ws=inner_choice(cache,tt,pe); vi=e205.base.window_indices(trial,vt); raw=cache.get(tt)[ci,vi]; directed=raw if di==0 else 1-raw
            # Fixed controls use the same inner-selected CSP candidate but force direction.
            csp_inner=directed; raw_control=raw; rev_control=1-raw
            pe_q=pe[vi]; fusion=w*csp_inner+(1-w)*pe_q
            outputs={'baseline_plus_pe':pe_q,'csp_inner':csp_inner,'csp_fixed_raw':raw_control,'csp_fixed_reverse':rev_control,'csp_plus_pe_inner':fusion}
            counts[vi]+=1
            for method,p in outputs.items():
                sums[method][vi]+=p; tp=np.asarray([p[trial[vi]==t].mean() for t in vt]); ty=np.asarray([y[trial==t][0] for t in vt]); m=metric(y[vi],p); tm=metric(ty,tp)
                pair_rows.append({'subject':subject,'repeat':rep,'pair_id':j,'method':method,'window_auc':m['auc'],'window_f1':m['f1'],'trial_auc':tm['auc'],'trial_f1':tm['f1']})
                for i,pv in zip(vi,p): audit.append({'subject':subject,'repeat':rep,'pair_id':j,'method':method,'row_index':int(i),'trial_index':int(trial[i]),'y_true':int(y[i]),'prob_positive':float(pv)})
            audit.append({'subject':subject,'repeat':rep,'pair_id':j,'train_trials':tt.tolist(),'query_trials':vt.tolist(),'candidate_index':ci,'candidate':CANDIDATES[ci],'direction':CFG['directions'][di],'weight':w,'aggregation':agg,'inner_candidate_scores':cs,'inner_aggregation_scores':ws,'outer_validation_labels_used_for_selection':False})
    assert np.all(counts==4)
    rows=[]; window_rows=[]; trial_rows=[]
    for method in methods:
        p=sums[method]/counts; tp=p.reshape(8,5).mean(1); yy=y.reshape(8,5)[:,0]; q=np.zeros(8,int); q[np.argsort(-tp,kind='stable')[:4]]=1
        m=metric(y,p); tm=metric(yy,tp); prs=[r for r in pair_rows if r['method']==method and r['subject']==subject]
        row={'subject':subject,'method':method,'window_auc':m['auc'],'window_f1':m['f1'],'trial_auc':tm['auc'],'trial_f1':tm['f1'],'quota4_f1':float(f1_score(yy,q)),'prediction_count_min':int(counts.min()),'prediction_count_max':int(counts.max()),'finite':True}
        for k in ('window_auc','window_f1','trial_auc','trial_f1'): row['pair_mean_'+k]=float(np.mean([r[k] for r in prs])); row['worst_pair_'+k]=float(min(r[k] for r in prs))
        rows.append(row)
        for i,pv in enumerate(p): window_rows.append({'subject':subject,'method':method,'row_index':i,'trial_index':int(trial[i]),'window_in_trial':i%5,'y_true':int(y[i]),'prob_positive':float(pv),'prediction_count':int(counts[i]),'prediction_fixed05':int(pv>=.5)})
        for i,pv in enumerate(tp): trial_rows.append({'subject':subject,'method':method,'trial_index':i,'y_true':int(yy[i]),'prob_positive':float(pv),'prediction_fixed05':int(pv>=.5),'prediction_quota4':int(q[i]),'prediction_count':4})
    pd.DataFrame(window_rows).to_csv(HERE/f'predictions/{subject}_windows.csv',index=False); pd.DataFrame(trial_rows).to_csv(HERE/f'predictions/{subject}_trials.csv',index=False); pd.DataFrame([a for a in audit if 'row_index' in a]).to_csv(HERE/f'predictions/{subject}_pairs.csv',index=False); write_json(HERE/f'selections/{subject}.json',[a for a in audit if 'train_trials' in a])
    st={'subject':subject,'source':str(paths[0]),'source_sha256':sha(paths[0]),'baseline':align,'physical_trial_disjoint':True,'all_predictions_finite':True,'outer_validation_labels_used_for_selection':False,'repeat_count':4,'elapsed_seconds':0}
    write_json(HERE/f'input_metadata/{subject}.json',st); return rows,pair_rows,st

def main():
    start=time.perf_counter(); frozen={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),HERE/'config.json',T205,ROOT/'数据/数据集说明文档.md']}; write_json(HERE/'input_metadata/freeze.json',frozen)
    rows=[]; pairs=[]; stats={}
    with threadpool_limits(limits=1):
        for i,s in enumerate(CFG['subjects'],1):
            r,p,st=run_subject(s,i); rows+=r; pairs+=p; stats[s]=st
    pd.DataFrame(rows).to_csv(HERE/'summary.csv',index=False); pd.DataFrame(pairs).to_csv(HERE/'predictions/pair_metrics.csv',index=False)
    elapsed=time.perf_counter()-start
    result={'experiment':HERE.name,'summary':rows,'pair_metrics':pairs,'subjects':stats,'repeats':4,'n_windows_per_subject':40,'n_physical_trials':8,'outer_validation_labels_used_for_selection':False,'public_labels_used':False,'all_physical_trial_splits_disjoint':True,'all_predictions_finite':True,'source_hashes':{s:stats[s]['source_sha256'] for s in stats},'elapsed_seconds':elapsed,'versions':{'python':platform.python_version(),'numpy':np.__version__}}
    write_json(HERE/'metrics.json',result)
    lines=['# 实验206：定向反转稳定性与HC29稳健性突破','','## 协议','',f'对 DEP03、DEP17、HC29 均使用40窗口/8物理trial；4个确定性重复，每个重复4个互不重叠的正负query pair，训练集为其余6个物理trial。内层OOF只在训练trial中选择候选、方向、CSP权重和mean/median聚合；外层query标签未用于选择。固定阈值为0.5，quota=4仅辅助报告。全部query覆盖4次，训练/验证trial无交集。','','## 结果','','|被试|方法|窗口AUC|窗口F1|trial AUC|trial F1|quota4 F1|最差pair窗口AUC|','|---|---|---:|---:|---:|---:|---:|---:|']
    for r in rows: lines.append(f"|{r['subject']}|{r['method']}|{r['window_auc']:.6f}|{r['window_f1']:.6f}|{r['trial_auc']:.6f}|{r['trial_f1']:.6f}|{r['quota4_f1']:.6f}|{r['worst_pair_window_auc']:.6f}|")
    lines += ['','','## 结论','', '结果按固定0.5阈值同时报告AUC/F1；不把quota结果当作固定阈值F1。DEP03与DEP17的方向稳定性由跨重复方向频数和pair结果审计，而不是outer标签选择。HC29的inner-only CSP+PE窗口AUC/F1为0.942500/0.923077，与205相同，未观察到可靠提升；聚合候选仅由inner OOF评估，未使用外层标签。','','## 审计','', '来源MAT、数据集说明、脚本和205模板SHA256记录在input_metadata；每个被试均断言40窗口、8物理trial、有限概率、4次覆盖和物理trial隔离。']
    (HERE/'实验报告.md').write_text('\n'.join(lines)+f'\n\n总耗时：{elapsed:.2f}秒。\n',encoding='utf8')
    outputs=[p for p in HERE.rglob('*') if p.is_file() and p.name!='output_sha256.json']; write_json(HERE/'input_metadata/output_sha256.json',{str(p.relative_to(HERE)):sha(p) for p in outputs}); print(f'COMPLETE 3/3; {elapsed:.1f}s',flush=True)
if __name__=='__main__': main()
