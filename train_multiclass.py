"""Prepare, fit, freeze, then score person-specific multiclass EEGNet + baseline."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from networks import EEGNet
from multiclass_data import prepare,validate_config,digest
from multiclass_metrics import evaluate,trial_class_weights

def dump(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')

def load_prepared(work):
    work=Path(work);manifest=json.loads((work/'manifest.json').read_text());c=validate_config(manifest['config'])
    if manifest['failures']: raise ValueError('Resolve preparation failures before training.')
    data=np.load(work/'windows.npz',allow_pickle=False);X=data['X'];y=data['y']
    records=json.loads((work/'records.json').read_text())
    if len(X)!=len(y) or len(y)!=len(records): raise ValueError('Prepared-data shape mismatch.')
    if not np.array_equal(y,np.array([r['label'] for r in records])): raise ValueError('Prepared labels differ from cue provenance.')
    sets={p:{r['session'] for r in records if r['partition']==p} for p in ['train','validation','test']}
    if sets['train']&sets['validation'] or sets['train']&sets['test'] or sets['validation']&sets['test']:
        raise ValueError('Session leakage in prepared data.')
    for p in sets:
        if set(y[[i for i,r in enumerate(records) if r['partition']==p]])!=set(range(len(c['classes']))):
            raise ValueError(f'{p} lacks one or more classes.')
    return X,y,records,manifest,c

def bandpower(X,fs=256):
    n=X.shape[-1];freq=np.fft.rfftfreq(n,1/fs)
    spectrum=abs(np.fft.rfft((X-X.mean(axis=-1,keepdims=True))*np.hanning(n),axis=-1))**2
    values=[np.log(np.maximum(spectrum[...,(freq>=a)&(freq<b)].mean(axis=-1),1e-24)) for a,b in [(8,12),(12,16),(16,22),(22,30)]]
    return np.stack(values,axis=-1).reshape(len(X),-1)

def nn_probabilities(model,X,mean,std,batch=128):
    model.eval();out=[]
    with torch.inference_mode():
        for a in range(0,len(X),batch):
            z=(X[a:a+batch]-mean[None,:,None])/(std[None,:,None]+1e-8)
            out.append(torch.softmax(model(torch.from_numpy(z)),dim=1).numpy())
    return np.concatenate(out)

def baseline_probabilities(path,X):
    b=json.loads(Path(path).read_text());features=(bandpower(X)-np.array(b['mean']))/np.array(b['scale'])
    return softmax(features@np.array(b['coef']).T+np.array(b['intercept']),axis=1)

def fit(work):
    work=Path(work)
    if (work/'test_results.json').exists(): raise ValueError('This experiment has already exposed its test results. Declare a new experiment/test for model changes.')
    X,y,records,manifest,c=load_prepared(work);torch.set_num_threads(c['threads'])
    torch.manual_seed(c['seed']);np.random.seed(c['seed'])
    tr=np.array([i for i,r in enumerate(records) if r['partition']=='train'])
    va=np.array([i for i,r in enumerate(records) if r['partition']=='validation'])
    tr_rows=[records[i] for i in tr];va_rows=[records[i] for i in va]
    weights=trial_class_weights(y[tr],tr_rows)
    mean=X[tr].mean(axis=(0,2)).astype(np.float32);std=X[tr].std(axis=(0,2)).astype(np.float32)
    if np.any(std<1e-9): raise ValueError('Training channel has negligible variance.')
    # Baseline is fixed in advance; no NN-dependent test-set selection.
    features=bandpower(X[tr]);scaler=StandardScaler().fit(features,sample_weight=weights)
    baseline=LogisticRegression(C=1.,max_iter=2000)
    baseline.fit(scaler.transform(features),y[tr],sample_weight=weights*len(tr)/weights.sum())
    dump(work/'bandpower_baseline.json',dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
        coef=baseline.coef_.tolist(),intercept=baseline.intercept_.tolist(),classes=c['classes'],C=1.))
    ds=TensorDataset(torch.from_numpy((X[tr]-mean[None,:,None])/(std[None,:,None]+1e-8)),torch.from_numpy(y[tr]))
    sampler=WeightedRandomSampler(torch.from_numpy(weights),len(tr),replacement=True,generator=torch.Generator().manual_seed(c['seed']))
    loader=DataLoader(ds,batch_size=c['batch_size'],sampler=sampler)
    model=EEGNet(len(manifest['channels']),X.shape[-1],len(c['classes']),drop=c['dropout'])
    opt=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=1e-4)
    history=[];best=(-1.,float('-inf'));best_state=None;wait=0;t0=time.perf_counter()
    for epoch in range(1,c['epochs']+1):
        model.train();total=0.
        for xb,yb in loader:
            opt.zero_grad();loss=nn.functional.cross_entropy(model(xb),yb);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();total+=float(loss.detach())*len(yb)
        p=nn_probabilities(model,X[va],mean,std)
        metrics=evaluate(p,y[va],va_rows,c['classes'])
        val_loss=float(-np.log(np.maximum(p[np.arange(len(p)),y[va]],1e-12)).mean())
        score=(metrics['cue_trials']['balanced_accuracy'],-val_loss)
        history.append(dict(epoch=epoch,training_loss=total/len(tr),validation_cue_balanced_accuracy=score[0],validation_loss=val_loss))
        print(f"epoch {epoch}: loss={total/len(tr):.4f}, tuning cue BA={score[0]:.4f}",flush=True)
        if score>best:
            best=score;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};wait=0
        else: wait+=1
        if wait>=c['patience']:break
    model.load_state_dict(best_state)
    checkpoint=work/'eegnet_multiclass.pt'
    torch.save(dict(arch='eegnet',n_classes=len(c['classes']),n_channels=len(manifest['channels']),n_times=X.shape[-1],
        model_state=model.state_dict(),train_mean=mean,train_std=std),checkpoint)
    contract=dict(schema_version=2,task_mode='multiclass',role='recorded_within_person_reference',
        classes=c['classes'],channels=manifest['channels'],sfreq=c['sfreq'],n_times=X.shape[-1],
        normalization='fixed',filter=dict(mode='causal',fmin=8.,fmax=30.,order=4),
        artifact_limit_v=c['artifact_limit_v'],min_signal_std_v=c['min_signal_std_v'],
        sha256=digest(checkpoint),prepared_manifest_sha256=digest(work/'manifest.json'),
        experiment=c,verified=True,verified_meaning='Reproducible preprocessing and split; not live biological validation.')
    dump(str(checkpoint)+'.json',contract)
    tuning=dict(eegnet=evaluate(nn_probabilities(model,X[va],mean,std),y[va],va_rows,c['classes']),
        bandpower_baseline=evaluate(baseline_probabilities(work/'bandpower_baseline.json',X[va]),y[va],va_rows,c['classes']))
    dump(work/'training.json',dict(history=history,tuning=tuning,elapsed_s=time.perf_counter()-t0,
        best_epoch=max(range(len(history)),key=lambda i:(history[i]['validation_cue_balanced_accuracy'],-history[i]['validation_loss']))+1,
        test_used_for_training=False))
    dump(work/'frozen.json',dict(model_sha256=digest(checkpoint),baseline_sha256=digest(work/'bandpower_baseline.json'),
        manifest_sha256=digest(work/'manifest.json'),records_sha256=digest(work/'records.json'),
        windows_sha256=digest(work/'windows.npz'),contract_sha256=digest(str(checkpoint)+'.json')))
    print(json.dumps(tuning,indent=2),flush=True)

def verify_frozen(work):
    work=Path(work)
    frozen=json.loads((work/'frozen.json').read_text())
    for key,name in [('model_sha256','eegnet_multiclass.pt'),('baseline_sha256','bandpower_baseline.json'),
                     ('manifest_sha256','manifest.json'),('records_sha256','records.json'),('windows_sha256','windows.npz'),
                     ('contract_sha256','eegnet_multiclass.pt.json')]:
        if digest(work/name)!=frozen[key]: raise ValueError(f'Frozen artifact changed: {name}')
    return frozen

def test_once(work,root=None):
    work=Path(work)
    if (work/'test_results.json').exists(): raise ValueError('Test already evaluated; read its saved report.')
    frozen=verify_frozen(work)
    if root is not None:
        contract=json.loads((work/'eegnet_multiclass.pt.json').read_text())
        if not contract.get('policy',{}).get('validation_tuned'):raise ValueError('Tune/freeze the validation replay policy before testing.')
    X,y,records,manifest,c=load_prepared(work);torch.set_num_threads(c['threads'])
    indices=np.array([i for i,r in enumerate(records) if r['partition']=='test']);rows=[records[i] for i in indices]
    ck=torch.load(work/'eegnet_multiclass.pt',map_location='cpu',weights_only=False)
    model=EEGNet(ck['n_channels'],ck['n_times'],ck['n_classes']);model.load_state_dict(ck['model_state'])
    result=dict(experiment_id=c['experiment_id'],classes=c['classes'],chance_balanced_accuracy=1/len(c['classes']),
        frozen=frozen,partitions=manifest['partitions'],
        eegnet=evaluate(nn_probabilities(model,X[indices],ck['train_mean'],ck['train_std']),y[indices],rows,c['classes'],True,c['seed']),
        bandpower_baseline=evaluate(baseline_probabilities(work/'bandpower_baseline.json',X[indices]),y[indices],rows,c['classes'],True,c['seed']),
        scope='Recorded within-person three-class subset. Not prospective accuracy or nine-class/open-set performance.')
    if root is not None:
        from replay_multiclass import evaluate_frozen_replay
        result['continuous_replay']=evaluate_frozen_replay(root,work)
    dump(work/'test_results.json',result);print(json.dumps(result,indent=2),flush=True)
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','fit','test'])
    p.add_argument('--config',type=Path,default=Path('multiclass_experiment.json'))
    p.add_argument('--root',type=Path);p.add_argument('--work',type=Path,default=Path('session_reports/multiclass/sub08_three_class_v1'))
    a=p.parse_args()
    if a.stage=='prepare':
        if a.root is None: p.error('--root is required to prepare EDF recordings')
        if (a.work/'frozen.json').exists(): raise ValueError('Prepared experiment is frozen. Use a fresh work folder for another preparation.')
        prepare(a.root,json.loads(a.config.read_text()),a.work)
    elif a.stage=='fit':fit(a.work)
    else:
        if a.root is None:p.error('--root is required for the frozen continuous replay test')
        test_once(a.work,a.root)

if __name__=='__main__': main()
