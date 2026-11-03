"""Train/export a causal EEGNet with subject/run-disjoint train/validation/test sets.

Uses only windows contained in a single annotated cue. The supplied dataset has
3-second rest cues, so the default window is TWO seconds, not four.
"""
import argparse
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
import mne
import torch
from torch import nn
from torch.utils.data import TensorDataset,DataLoader,WeightedRandomSampler
from scipy.signal import butter,sosfilt
from sklearn.model_selection import GroupShuffleSplit,StratifiedShuffleSplit
from sklearn.metrics import balanced_accuracy_score,confusion_matrix
from networks import EEGNet
from create_personal_template import MONTAGES

COMMON_X24=['F3','Fz','F4','C3','Cz','C4','P3','Pz','P4','POz','O1','O2']

def event_windows(data,events,sfreq=256,n_times=512,settle=.5,stride=.5):
    """Return (start,end,label,cue_index); never cross a cue boundary."""
    spans=[]
    for cue,row in enumerate(events.to_dict('records')):
        if row['trial_type'] not in ('rest','action'): continue
        start=int(row['sample']);end=min(data.shape[1],start+round(float(row['duration'])*sfreq))
        for a in range(start+round(settle*sfreq),end-n_times+1,max(1,round(stride*sfreq))):
            spans.append((a,a+n_times,int(row['trial_type']=='action'),cue))
    return spans

def load_dataset(root,channels,sessions=None,subjects=None,n_times=512):
    xs,ys,groups,records=[],[],[],[]
    sos=butter(4,[8.,30.],fs=256,btype='bandpass',output='sos')
    files=sorted(Path(root).glob('sub-*/ses-*/eeg/*_eeg.edf'))
    if not files: raise ValueError('No EDF recordings found.')
    for path in files:
        sub=path.parents[2].name;ses=path.parents[1].name
        if subjects and int(sub[4:]) not in subjects: continue
        if sessions and int(ses[4:]) not in sessions: continue
        ev=path.with_name(path.name.replace('_eeg.edf','_events.tsv'))
        if not ev.exists(): raise ValueError(f'Missing events file: {ev.name}')
        raw=mne.io.read_raw_edf(path,preload=False,verbose=False)
        if not np.isclose(raw.info['sfreq'],256): raise ValueError(f'{path.name}: expected 256 Hz.')
        missing=set(channels)-set(raw.ch_names)
        if missing: raise ValueError(f'{path.name}: missing channels {sorted(missing)}')
        raw.pick(channels)
        data=raw.get_data().astype(np.float32)
        if not np.isfinite(data).all(): raise ValueError(f'{path.name}: nonfinite signal.')
        filtered=sosfilt(sos,data,axis=1).astype(np.float32)
        bad=(np.abs(filtered)>100e-6).any(axis=0)
        events=pd.read_csv(ev,sep='\t')
        kept=0
        for a,b,label,cue in event_windows(data,events,n_times=n_times):
            if bad[a:b].any() or np.any(filtered[:,a:b].std(axis=1)<1e-9): continue
            xs.append(filtered[:,a:b].copy());ys.append(label);groups.append(sub)
            records.append(dict(subject=sub,session=ses,run=path.stem.replace('_eeg',''),start=a,end=b,label=label,cue=cue))
            kept+=1
        print(f'{path.name}: {kept} clean cue-contained windows',flush=True)
    if not xs: raise ValueError('No clean cue-contained windows.')
    return np.stack(xs),np.array(ys),np.array(groups),records

def split_groups(y,groups,seed=42):
    unique=np.unique(groups)
    if len(unique)<5: raise ValueError('Need at least five independent groups for train/validation/test.')
    labels=[np.unique(y[groups==g]) for g in unique]
    if all(len(a)==1 for a in labels):
        # Personal data: split whole cue trials, stratified by cue class.
        gy=np.array([a[0] for a in labels])
        tv,te=next(StratifiedShuffleSplit(n_splits=1,test_size=.2,random_state=seed).split(unique,gy))
        tr,va=next(StratifiedShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1).split(unique[tv],gy[tv]))
        train=np.flatnonzero(np.isin(groups,unique[tv[tr]]))
        val=np.flatnonzero(np.isin(groups,unique[tv[va]]))
        test=np.flatnonzero(np.isin(groups,unique[te]))
    else:
        splitter=GroupShuffleSplit(n_splits=1,test_size=.2,random_state=seed)
        trainval,test=next(splitter.split(np.zeros(len(y)),y,groups))
        split2=GroupShuffleSplit(n_splits=1,test_size=.25,random_state=seed+1)
        tr,va=next(split2.split(np.zeros(len(trainval)),y[trainval],groups[trainval]))
        train,val=trainval[tr],trainval[va]
    for indices in (train,val,test):
        if set(y[indices])!={0,1}: raise ValueError('Each split must contain both classes.')
    assert not (set(groups[train])&set(groups[val]) or set(groups[train])&set(groups[test]) or set(groups[val])&set(groups[test]))
    return train,val,test

def train_export(X,y,groups,records,output,channels,epochs=40,patience=8,seed=42,batch_size=64,
                 initial_checkpoint=None,freeze_backbone=False,normalization='fixed',explicit_split=None,
                 validation_only=False,progress_callback=None,stop_event=None):
    torch.manual_seed(seed);np.random.seed(seed)
    train,val,test=explicit_split if explicit_split is not None else split_groups(y,groups,seed)
    for indices in ((train,val) if validation_only else (train,val,test)):
        if not len(indices) or set(y[indices])!={0,1}: raise ValueError('All three partitions must contain both classes.')
    if set(groups[train])&set(groups[val]) or set(groups[train])&set(groups[test]) or set(groups[val])&set(groups[test]):
        raise ValueError('Training, validation and test groups must be disjoint.')
    mean=X[train].mean(axis=(0,2)).astype(np.float32)
    std=(X[train].std(axis=(0,2))+1e-8).astype(np.float32)
    if normalization=='fixed': Xn=(X-mean[None,:,None])/(std[None,:,None]+1e-8)
    elif normalization=='per_window': Xn=(X-X.mean(axis=2,keepdims=True))/(X.std(axis=2,keepdims=True)+1e-8)
    else: raise ValueError('Unknown normalization mode.')
    ds=TensorDataset(torch.from_numpy(Xn[train]),torch.from_numpy(y[train]).long())
    counts=np.bincount(y[train],minlength=2)
    sampler=WeightedRandomSampler(torch.from_numpy((1./counts)[y[train]]),len(train),replacement=True,
                                  generator=torch.Generator().manual_seed(seed))
    dl=DataLoader(ds,batch_size=batch_size,sampler=sampler)
    model=EEGNet(len(channels),X.shape[2])
    if initial_checkpoint:
        from realtime_bci_decoder import RealtimeBCIDecoder
        parent=RealtimeBCIDecoder(initial_checkpoint,send_udp=False)
        if parent.channels!=channels or parent.window_samples!=X.shape[2]: raise ValueError('Personal/base model dimension mismatch.')
        model.load_state_dict(parent.model.state_dict())
    if freeze_backbone:
        for name,param in model.named_parameters(): param.requires_grad_(name.startswith('fc.'))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3,weight_decay=1e-4)
    criterion=nn.CrossEntropyLoss();best=-1.;best_state=None;wait=0;history=[]
    def probabilities(indices):
        model.eval();values=[]
        with torch.inference_mode():
            for batch in np.array_split(indices,max(1,int(np.ceil(len(indices)/batch_size)))):
                values.extend(torch.softmax(model(torch.from_numpy(Xn[batch])),dim=1)[:,1].numpy())
        return np.asarray(values)
    t0=time.perf_counter()
    for epoch in range(1,epochs+1):
        if stop_event is not None and stop_event.is_set(): raise InterruptedError('Personal calibration cancelled.')
        model.train();loss_sum=0.
        if freeze_backbone: model.block1.eval();model.block2.eval()
        for xb,yb in dl:
            if stop_event is not None and stop_event.is_set(): raise InterruptedError('Personal calibration cancelled.')
            opt.zero_grad();loss=criterion(model(xb),yb);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();loss_sum+=float(loss.detach())*len(yb)
        vp=probabilities(val);score=float(balanced_accuracy_score(y[val],vp>=.5))
        history.append(dict(epoch=epoch,train_loss=loss_sum/len(train),validation_balanced_accuracy=score))
        if progress_callback: progress_callback(history[-1])
        print(f'Epoch {epoch}: loss={history[-1]["train_loss"]:.4f}, val_bal={score:.4f}',flush=True)
        if score>best:
            best=score;best_state={k:v.detach().clone() for k,v in model.state_dict().items()};wait=0
        else: wait+=1
        if wait>=patience: break
    model.load_state_dict(best_state)
    if stop_event is not None and stop_event.is_set(): raise InterruptedError('Personal calibration cancelled.')
    vp=probabilities(val)
    threshold=float(max(np.linspace(.1,.9,81),key=lambda th:(balanced_accuracy_score(y[val],vp>=th),-abs(th-.5))))
    # Test labels are first used for metrics after model and threshold are frozen.
    tp=probabilities(test) if len(test) else np.empty(0)
    test_bal=float(balanced_accuracy_score(y[test],tp>=threshold)) if len(test) else None
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model_state=model.state_dict(),arch='eegnet',n_channels=len(channels),n_times=X.shape[2],
                    train_mean=mean,train_std=std),output)
    partitions={name:sorted(set(groups[idx])) for name,idx in [('train',train),('validation',val),('test',test)]}
    c=dict(channels=channels,sfreq=256,n_times=X.shape[2],normalization=normalization,
           filter=dict(mode='causal',fmin=8.,fmax=30.,order=4),classes=['rest','action'],verified=True,
           verified_meaning='Reproducible preprocessing contract; not proof of biological or live validation.',
           sha256=hashlib.sha256(output.read_bytes()).hexdigest(),artifact_limit_v=100e-6,
           artifact_detection='filtered_peak',raw_clipping=False,
           threshold=threshold,partitions=partitions,seed=seed,
           training_runs=sorted(set(records[i]['run'] for i in train)),
           validation_runs=sorted(set(records[i]['run'] for i in val)),
           test_runs=sorted(set(records[i]['run'] for i in test)))
    c['evaluation_scope']='validation_only_prospective_test_pending' if validation_only else 'held_out_offline_test'
    if initial_checkpoint:
        c['parent_sha256']=hashlib.sha256(Path(initial_checkpoint).read_bytes()).hexdigest()
        c['personal_model']=True
    Path(str(output)+'.json').write_text(json.dumps(c,indent=2),encoding='utf-8')
    report=dict(partitions=partitions,n_windows={name:len(idx) for name,idx in [('train',train),('validation',val),('test',test)]},
                threshold=threshold,history=history,test_balanced_accuracy=test_bal,
                validation_balanced_accuracy_at_threshold=float(balanced_accuracy_score(y[val],vp>=threshold)),
                test_confusion_matrix=confusion_matrix(y[test],tp>=threshold,labels=[0,1]).tolist() if len(test) else None,
                elapsed_s=time.perf_counter()-t0,
                warning='Offline windows overlap within trials. Counts are not independent trials; prospective self-test still required.')
    Path(str(output)+'.training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    pd.DataFrame([dict(records[i],p_action=float(p)) for i,p in zip(test,tp)]).to_csv(str(output)+'.test_predictions.csv',index=False)
    print(json.dumps(report,indent=2),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--sessions',nargs='+',type=int)
    p.add_argument('--subjects',nargs='+',type=int,required=True);p.add_argument('--channels',default=','.join(MONTAGES['eegmi_custom20']))
    p.add_argument('--window-s',type=float,default=2.);p.add_argument('--epochs',type=int,default=40)
    p.add_argument('--patience',type=int,default=8);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--split-by',choices=['subject','run','session'],default='session')
    p.add_argument('--normalization',choices=['fixed','per_window'],default='fixed')
    p.add_argument('--test-sessions',nargs='+',type=int);p.add_argument('--validation-sessions',nargs='+',type=int)
    p.add_argument('--validation-only',action='store_true',help='Use one subject for tuning; prospective self-test remains the independent test.')
    p.add_argument('--validation-subject',type=int)
    p.add_argument('--output',type=Path,default=Path('session_reports/models/eegnet_recorded.pt'))
    args=p.parse_args();torch.set_num_threads(args.threads)
    channels=[x.strip() for x in args.channels.split(',')]
    X,y,groups,records=load_dataset(args.root,channels,args.sessions,args.subjects,round(args.window_s*256))
    if args.split_by=='run': groups=np.array([r['run'] for r in records])
    if args.split_by=='session': groups=np.array([r['subject']+'_'+r['session'] for r in records])
    explicit=None
    if args.validation_only:
        if args.split_by!='subject' or args.validation_subject is None: raise ValueError('Validation-only mode requires --split-by subject and --validation-subject.')
        validation=groups==f'sub-{args.validation_subject:02d}'
        explicit=(np.flatnonzero(~validation),np.flatnonzero(validation),np.empty(0,dtype=int))
    if args.test_sessions or args.validation_sessions:
        if args.split_by!='session' or not args.test_sessions or not args.validation_sessions:
            raise ValueError('Explicit held-out sessions require --split-by session and both session lists.')
        if set(args.test_sessions)&set(args.validation_sessions): raise ValueError('Session sets overlap.')
        ses=np.array([int(r['session'][4:]) for r in records])
        te=np.isin(ses,args.test_sessions);va=np.isin(ses,args.validation_sessions)
        explicit=(np.flatnonzero(~te&~va),np.flatnonzero(va),np.flatnonzero(te))
    train_export(X,y,groups,records,args.output,channels,args.epochs,args.patience,args.seed,
                 normalization=args.normalization,explicit_split=explicit,validation_only=args.validation_only)
if __name__=='__main__': main()
