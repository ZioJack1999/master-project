"""Predeclared neural feature search; sealed sessions are never scored here."""
import argparse,json,time,copy
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset,WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from multiclass_data import digest,validate_config,prepare
from multiclass_features import band_covariances,FeaturePreprocessor,FeatureNetwork,FeatureEnsembleRunner,MOTOR10
from multiclass_metrics import evaluate,trial_class_weights,cue_probabilities
from train_multiclass import load_prepared,bandpower,dump

def probabilities(model,features,batch=256):
    model.eval();result=[]
    with torch.inference_mode():
        for a in range(0,len(features),batch):result.append(torch.softmax(model(torch.from_numpy(features[a:a+batch])),dim=1).numpy())
    return np.concatenate(result)

def fit_member(train_x,train_y,weights,val_x,val_y,val_rows,c,preprocessing,family,n_channels,seed,train_rows=None):
    torch.manual_seed(seed);model=FeatureNetwork(family,n_channels,len(c['classes']),preprocessing,c['hidden_units'],c['dropout'])
    data=TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_y))
    draws=len(train_y)
    if train_rows is not None and c.get('draws_per_training_cue'):
        draws=min(draws,c['draws_per_training_cue']*len({r['trial'] for r in train_rows}))
    sampler=WeightedRandomSampler(torch.from_numpy(weights),draws,replacement=True,generator=torch.Generator().manual_seed(seed))
    loader=DataLoader(data,batch_size=c['batch_size'],sampler=sampler)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['learning_rate'],weight_decay=c['weight_decay'])
    best=(-1.,-1.,float('-inf'));state=None;wait=0;history=[]
    for epoch in range(1,c['epochs']+1):
        model.train();loss_sum=0.
        for x,y in loader:
            optimizer.zero_grad();loss=nn.functional.cross_entropy(model(x),y);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();loss_sum+=float(loss.detach())*len(y)
        p=probabilities(model,val_x);metrics=evaluate(p,val_y,val_rows,c['classes'])['cue_trials']
        truth,cue_p,runs=cue_probabilities(p,val_rows)
        nll=float(-np.log(np.maximum(cue_p[np.arange(len(truth)),truth],1e-12)).mean())
        score=(metrics['balanced_accuracy'],min(metrics['recall'].values()),-nll)
        row=dict(epoch=epoch,training_loss=loss_sum/draws,validation_cue_balanced_accuracy=score[0],validation_min_class_recall=score[1],validation_cue_loss=nll)
        if train_rows is not None and c.get('diagnostic_interval') and (epoch==1 or epoch%c['diagnostic_interval']==0):
            tp=probabilities(model,train_x);tm=evaluate(tp,train_y,train_rows,c['classes'])['cue_trials']
            ty,tprob,_=cue_probabilities(tp,train_rows)
            row.update(train_cue_balanced_accuracy=tm['balanced_accuracy'],train_cue_loss=float(-np.log(np.maximum(tprob[np.arange(len(ty)),ty],1e-12)).mean()))
        history.append(row)
        if score>best:best=score;state=copy.deepcopy(model.state_dict());wait=0
        else:wait+=1
        if epoch%20==0:print(f'  seed {seed} epoch {epoch}: tuning BA={score[0]:.3f}, best={best[0]:.3f}',flush=True)
        if wait>=c['patience']:break
    model.load_state_dict(state)
    print(f'  seed {seed} stopped {epoch}; best tuning BA={best[0]:.3f}',flush=True)
    return state,probabilities(model,val_x),history

def search(work,previous_manifest=None,config_path=Path('multiclass_improvement_experiment.json')):
    work=Path(work);out=work/'selected'
    if (out/'test_results.json').exists() or (work/'selection.json').exists():raise ValueError('Search already finalized. Preserve its frozen test and use new development data for another experiment.')
    base=validate_config(json.loads(Path(config_path).read_text()));torch.set_num_threads(base['threads'])
    prepared={};common=None;old_hashes=set()
    if previous_manifest and Path(previous_manifest).exists():
        old=json.loads(Path(previous_manifest).read_text());old_hashes={r['signal_sha256'] for r in old['runs']}
    for window in base['candidate_window_seconds']:
        folder=work/f'window_{int(window)}s';X,y,records,manifest,c=load_prepared(folder)
        for key in ['classes','train_sessions','validation_sessions','test_sessions']:
            if c[key]!=base[key]:raise ValueError(f'Prepared data differ from declared experiment: {key}')
        c['seed']=base['seed'];manifest['config']['seed']=base['seed']
        if any(r['signal_sha256'] in old_hashes for r in manifest['runs'] if r['partition']=='test'):
            raise ValueError('A proposed fresh test signal was used in the exposed experiment.')
        tr=np.array([i for i,r in enumerate(records) if r['partition']=='train']);va=np.array([i for i,r in enumerate(records) if r['partition']=='validation'])
        names={records[i]['trial'] for i in va};common=names if common is None else common&names
        # No covariance features or scores are computed for sealed test windows.
        prepared[window]=dict(X=X,y=y,records=records,manifest=manifest,c=c,tr=tr,va=va,
            train_cov=band_covariances(X[tr]),validation_cov=band_covariances(X[va]))
    if not common:raise ValueError('No shared eligible validation cues across window lengths.')
    work.mkdir(parents=True,exist_ok=True);results=[];best=None;start=time.perf_counter()
    for window,d in prepared.items():
        c=d['c'];tr=d['tr'];va=d['va'];records=d['records'];all_channels=d['manifest']['channels']
        keep=np.array([j for j,i in enumerate(va) if records[i]['trial'] in common]);va=va[keep]
        train_rows=[records[i] for i in tr];val_rows=[records[i] for i in va];weights=trial_class_weights(d['y'][tr],train_rows)
        for montage in base['candidate_montages']:
            channels=all_channels if montage=='all20' else MOTOR10;indices=[all_channels.index(ch) for ch in channels]
            train_cov=d['train_cov'][:,:,indices,:][:,:,:,indices];val_cov=d['validation_cov'][keep][:,:,indices,:][:,:,:,indices]
            for family in base['candidate_architectures']:
                name=f'{family}_{montage}_{int(window)}s';print('CANDIDATE',name,flush=True)
                pre=FeaturePreprocessor.fit(train_cov,weights,family,c['covariance_shrinkage'])
                tx=pre.transform(train_cov);vx=pre.transform(val_cov);member_states=[];predictions=[];histories=[]
                for seed in base['ensemble_seeds']:
                    state,p,history=fit_member(tx,d['y'][tr],weights,vx,d['y'][va],val_rows,c,pre.parameters,family,len(channels),seed)
                    member_states.append(state);predictions.append(p);histories.append(history)
                p=np.mean(predictions,axis=0);metrics=evaluate(p,d['y'][va],val_rows,c['classes'])
                truth,cue_p,runs=cue_probabilities(p,val_rows);nll=float(-np.log(np.maximum(cue_p[np.arange(len(truth)),truth],1e-12)).mean())
                score=(metrics['cue_trials']['balanced_accuracy'],min(metrics['cue_trials']['recall'].values()),-nll)
                payload=dict(arch='feature_ensemble',family=family,n_classes=len(c['classes']),n_channels=len(channels),n_times=round(window*c['sfreq']),
                    hidden_units=c['hidden_units'],dropout=c['dropout'],preprocessing=pre.parameters,member_states=member_states,
                    train_mean=np.zeros(len(channels),np.float32),train_std=np.ones(len(channels),np.float32))
                candidate=work/'candidates'/name;candidate.mkdir(parents=True,exist_ok=True);torch.save(payload,candidate/'model.pt')
                row=dict(name=name,window_s=window,montage=montage,channels=channels,family=family,validation=metrics,validation_cue_loss=nll,
                    seed_histories=histories,ensemble_seeds=base['ensemble_seeds'],test_scored=False)
                dump(candidate/'training.json',row);results.append(row)
                print(f"RESULT {name}: common tuning cue BA={score[0]:.3%}, recalls={metrics['cue_trials']['recall']}",flush=True)
                if best is None or score>best['score']:best=dict(score=score,row=row,payload=payload,data=d,indices=indices)
    out.mkdir(parents=True,exist_ok=True);selected=best['row'];d=best['data'];indices=best['indices'];c=d['c'];rows=d['records']
    X=d['X'][:,indices,:].copy();manifest=copy.deepcopy(d['manifest']);manifest['channels']=selected['channels']
    manifest['config'].update(selected_candidate=selected['name'],selection_common_validation_cues=len(common),
        model_architecture='feature_ensemble',ensemble_seeds=base['ensemble_seeds'])
    dump(out/'manifest.json',manifest);dump(out/'records.json',rows);np.savez_compressed(out/'windows.npz',X=X,y=d['y'])
    checkpoint=out/'eegnet_multiclass.pt';torch.save(best['payload'],checkpoint)
    tr=d['tr'];weights=trial_class_weights(d['y'][tr],[rows[i] for i in tr]);features=bandpower(X[tr]);scaler=StandardScaler().fit(features,sample_weight=weights)
    baseline=LogisticRegression(C=1.,max_iter=2000).fit(scaler.transform(features),d['y'][tr],sample_weight=weights*len(tr)/weights.sum())
    dump(out/'bandpower_baseline.json',dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),coef=baseline.coef_.tolist(),intercept=baseline.intercept_.tolist(),classes=c['classes'],C=1.))
    contract=dict(schema_version=2,task_mode='multiclass',arch='feature_ensemble',role='recorded_within_person_reference',
        classes=c['classes'],channels=selected['channels'],sfreq=c['sfreq'],n_times=X.shape[-1],normalization='feature_train_only',
        filter=dict(mode='causal',fmin=8.,fmax=30.,order=4),artifact_limit_v=c['artifact_limit_v'],min_signal_std_v=c['min_signal_std_v'],
        sha256=digest(checkpoint),prepared_manifest_sha256=digest(out/'manifest.json'),experiment=manifest['config'],verified=True,
        verified_meaning='Train-only neural features/scales; validation-selected candidate. No live accuracy certification.')
    dump(str(checkpoint)+'.json',contract)
    selection=dict(selected_candidate=selected['name'],selection_common_validation_cues=len(common),
        selection_rule=base['selection_metric'],candidates=results,elapsed_s=time.perf_counter()-start,
        target_balanced_accuracy=base['target_balanced_accuracy'],test_scored=False,
        prior_exposure_signal_overlap_checked=bool(old_hashes),fresh_test_overlap=0 if old_hashes else None)
    dump(work/'selection.json',selection);dump(out/'training.json',dict(selection=selection,tuning=dict(neural_ensemble=selected['validation']),test_used_for_training=False))
    dump(out/'frozen.json',dict(model_sha256=digest(checkpoint),baseline_sha256=digest(out/'bandpower_baseline.json'),
        manifest_sha256=digest(out/'manifest.json'),records_sha256=digest(out/'records.json'),windows_sha256=digest(out/'windows.npz'),contract_sha256=digest(str(checkpoint)+'.json')))
    print('FROZEN SELECTED MODEL',selected['name'],'validation BA',selected['validation']['cue_trials']['balanced_accuracy'],flush=True)
    return selection

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',nargs='?',choices=['prepare','search'],default='search')
    p.add_argument('--work',type=Path,default=Path('session_reports/multiclass/sub08_three_class_v2'));p.add_argument('--root',type=Path)
    p.add_argument('--config',type=Path,default=Path('multiclass_improvement_experiment.json'))
    p.add_argument('--previous-manifest',type=Path,default=Path('session_reports/multiclass/sub08_three_class_v1/manifest.json'))
    a=p.parse_args()
    if a.stage=='prepare':
        if a.root is None:p.error('--root is required for preparation')
        if (a.work/'selection.json').exists():raise ValueError('Experiment is finalized; preserve the frozen artifacts.')
        base=validate_config(json.loads(a.config.read_text()))
        for window in base['candidate_window_seconds']:
            prepare(a.root,dict(base,window_s=window),a.work/f'window_{int(window)}s')
    else:search(a.work,a.previous_manifest,a.config)
