import gc, importlib.util, json, os, random, time
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, f1_score
from emo_rec.submission import validate_submission

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CACHE = HERE / 'cache'; PRED = HERE / 'predictions'
spec = importlib.util.spec_from_file_location('exp004', HERE.parent / '004_labram_frozen' / 'run_experiment.py')
previous = importlib.util.module_from_spec(spec); spec.loader.exec_module(previous)


def save_json(path, value): path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def sample_channel_zscore(x):
    x = np.asarray(x, dtype=np.float32)
    mean = x.mean(axis=2, keepdims=True); std = x.std(axis=2, keepdims=True)
    return (x - mean) / np.maximum(std, 1e-6)

class EEGNet(torch.nn.Module):
    def __init__(self, dropout=0.25):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, (1, 64), padding=(0, 32), bias=False),
            torch.nn.BatchNorm2d(8),
            torch.nn.Conv2d(8, 16, (30, 1), groups=8, bias=False),
            torch.nn.BatchNorm2d(16), torch.nn.ELU(),
            torch.nn.AvgPool2d((1, 8)), torch.nn.Dropout(dropout),
            torch.nn.Conv2d(16, 16, (1, 16), padding=(0, 8), groups=16, bias=False),
            torch.nn.Conv2d(16, 16, 1, bias=False), torch.nn.BatchNorm2d(16), torch.nn.ELU(),
            torch.nn.AvgPool2d((1, 8)), torch.nn.Dropout(dropout)
        )
        with torch.no_grad(): n = self.features(torch.zeros(1, 1, 30, 2000)).reshape(1, -1).shape[1]
        self.classifier = torch.nn.Linear(n, 2)
    def forward(self, x): return self.classifier(self.features(x).flatten(1))

def load_metadata():
    frame, _ = previous.load_baseline()
    if len(frame) != 2760 or frame.subject_id.nunique() != 69 or 'HC43' in set(frame.subject_id): raise RuntimeError('Invalid 002 reference metadata')
    if set(frame.fold) != set(range(5)) or not (frame.groupby('subject_id').size() == 40).all(): raise RuntimeError('Invalid fixed folds')
    for fold in range(5):
        a = set(frame.loc[frame.fold != fold, 'subject_id']); b = set(frame.loc[frame.fold == fold, 'subject_id'])
        if a & b: raise RuntimeError('Subject leakage in fixed folds')
    return frame

def prepare(frame):
    train_path, public_path, marker = CACHE/'train.npy', CACHE/'public.npy', CACHE/'preprocessing.json'
    if not marker.exists():
        paths = {p.stem:p for p in (ROOT/'数据'/'训练集').glob('*/*.mat') if p.stem != 'HC43'}
        train = np.lib.format.open_memmap(train_path, mode='w+', dtype='float32', shape=(2760,30,2000))
        for subject in frame.subject_id.unique():
            idx = np.flatnonzero(frame.subject_id.to_numpy()==subject)
            segments, labels, trials, _ = previous.preprocess_train_subject(paths[subject])
            if not np.array_equal(labels, frame.y_true.to_numpy()[idx]) or not np.array_equal(trials, frame.original_trial_id.to_numpy()[idx]): raise RuntimeError('002 reference alignment mismatch')
            train[idx] = segments.astype(np.float32)
        train.flush()
        public = np.lib.format.open_memmap(public_path, mode='w+', dtype='float32', shape=(80,30,2000))
        for i in range(10): public[i*8:(i+1)*8] = previous.preprocess_public_user(ROOT/'数据'/'公开测试集'/f'P_test{i+1}.mat')[0]
        public.flush(); save_json(marker, {'shape':[2760,30,2000], 'dtype':'float32', 'reference':'002', 'preprocessing':'005'})
    train, public = np.load(train_path, mmap_mode='r'), np.load(public_path, mmap_mode='r')
    if train.shape != (2760,30,2000) or public.shape != (80,30,2000): raise RuntimeError('Invalid cache shape')
    return train, public

def train_predict(x, y, fit_idx, x_pred, pred_idx, seed, config):
    seed_all(seed); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=EEGNet(config['model']['dropout']).to(device)
    opt=torch.optim.Adam(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay']); loss_fn=torch.nn.CrossEntropyLoss(); bs=config['training']['batch_size']; rows=[]
    xfit=sample_channel_zscore(x[fit_idx]); xp=sample_channel_zscore(x_pred[pred_idx]); tx=torch.from_numpy(xfit[:,None]).to(device); ty=torch.from_numpy(y[fit_idx]).long().to(device)
    for epoch in range(1, config['training']['epochs']+1):
        model.train(); order=np.random.default_rng(seed+epoch).permutation(len(fit_idx)); total=0
        for start in range(0,len(order),bs):
            ids=torch.from_numpy(order[start:start+bs]).to(device); opt.zero_grad(set_to_none=True); out=model(tx[ids]); loss=loss_fn(out,ty[ids]); loss.backward(); opt.step(); total += loss.item()*len(ids)
        rows.append({'epoch':epoch,'train_loss':total/len(order)})
    model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(xp),bs): out.append(torch.softmax(model(torch.from_numpy(xp[start:start+bs,None]).to(device)),1)[:,1].cpu().numpy())
    return np.concatenate(out), rows, str(device)

def metrics(y,p,frame):
    def m(a,b): return {'roc_auc':float(roc_auc_score(a,b)), 'f1':float(f1_score(a,b>=.5,zero_division=0))}
    folds=[{'fold':f, 'n_samples':int((frame.fold==f).sum()), **m(y[frame.fold==f],p[frame.fold==f])} for f in range(5)]
    subs=[]
    for s,g in frame.groupby('subject_id'):
        subs.append({'subject_id':s,'health_group':g.health_group.iloc[0],**m(g.y_true.to_numpy(),p[g.index])})
    health={h:{'pooled':m(y[frame.health_group.to_numpy()==h],p[frame.health_group.to_numpy()==h]),'subject_macro_auc':float(np.mean([r['roc_auc'] for r in subs if r['health_group']==h]))} for h in ('HC','DEP')}
    return {'overall':m(y,p),'folds':folds,'subject_macro_auc':float(np.mean([r['roc_auc'] for r in subs])),'health_groups':health,'subject_metrics':subs,'fold_auc_mean':float(np.mean([r['roc_auc'] for r in folds]))}

def main():
    config=json.loads((HERE/'config.json').read_text(encoding='utf-8')); frame=load_metadata(); x,public=prepare(frame); y=frame.y_true.to_numpy(); folds=frame.fold.to_numpy(); oof=np.full(len(y),np.nan); all_losses={}; started=time.perf_counter(); device=''
    for f in range(5):
        p,loss,device=train_predict(x,y,np.flatnonzero(folds!=f),x,np.flatnonzero(folds==f),2026+f,config); oof[folds==f]=p; all_losses[f'fold_{f}']=loss; print(f'fold {f} done',flush=True)
    pp,loss,device=train_predict(x,y,np.arange(len(y)),public,np.arange(len(public)),2031,config); all_losses['full']=loss
    PRED.mkdir(exist_ok=True); out=frame.copy(); out['prob_positive']=oof; out.to_csv(PRED/'eegnet_oof.csv',index=False)
    pub=pd.DataFrame({'user_id':np.repeat([f'P_test{i}' for i in range(1,11)],8),'trial_id':np.tile(np.arange(1,9),10),'prob. of positive emotion':pp}); validate_submission(pub, [f'P_test{i}' for i in range(1,11)]); pub.to_csv(PRED/'eegnet_public.csv',index=False)
    result=metrics(y,oof,frame); result['training_loss']=all_losses; result['runtime_seconds']=time.perf_counter()-started; result['device']=device; result['public_labels_used']=False; result['normalization']='per-sample per-channel zscore'; result['input_shape']=[30,2000]
    save_json(HERE/'metrics.json',{'experiment_config':config,'runs':{'eegnet':result}}); pd.DataFrame([{'run_name':'eegnet',**result['overall'],'subject_macro_auc':result['subject_macro_auc'],'hc_auc':result['health_groups']['HC']['pooled']['roc_auc'],'dep_auc':result['health_groups']['DEP']['pooled']['roc_auc']}]).to_csv(HERE/'summary.csv',index=False)
    print(json.dumps(result['overall'],ensure_ascii=False),flush=True)
if __name__=='__main__': main()
