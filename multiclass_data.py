"""Strict, person-specific BCIFEST multiclass EDF preparation in volts."""
from pathlib import Path
from collections import Counter
import hashlib,json
import numpy as np
import pandas as pd
import mne
from scipy.signal import butter,sosfilt
from create_personal_template import MONTAGES

GESTURES=['rest','volar_grasp','finger_extension','opponens_grasp','wrist_flexion',
          'wrist_extension','wrist_pronation','wrist_supination','pinch_grasp']

def canonical(value):
    return '_'.join(str(value).strip().casefold().replace('_',' ').split())

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_config(config):
    c=dict(config)
    c['classes']=[canonical(x) for x in c['classes']]
    if len(c['classes'])<3 or len(set(c['classes']))!=len(c['classes']) or not set(c['classes'])<=set(GESTURES):
        raise ValueError('Choose at least three unique protocol classes.')
    sets=[set(c[k]) for k in ['train_sessions','validation_sessions','test_sessions']]
    if any(not s for s in sets) or any(sets[i]&sets[j] for i in range(3) for j in range(i)):
        raise ValueError('Nonempty train/validation/test sessions must be disjoint.')
    if c['sfreq']!=256 or c['window_s']<=0 or not 0<c['stride_s']<=c['window_s'] or c['settle_s']<0:
        raise ValueError('Unsupported sample rate or window/stride/settling.')
    if c['montage'] not in MONTAGES: raise ValueError('Unknown physical montage.')
    return c

def cue_spans(events,n_samples,config):
    """Round event onsets; cap to actual EDF end and next cue, never infer labels."""
    fs=config['sfreq'];classes=config['classes'];spans=[]
    if not {'onset','duration','trial_type'}<=set(events): raise ValueError('Missing event columns.')
    rows=events.to_dict('records')
    previous=-1.
    for i,r in enumerate(rows):
        onset=float(r['onset']);duration=float(r['duration'])
        if not np.isfinite([onset,duration]).all() or onset<0 or onset<previous or duration<=0:
            raise ValueError('Invalid, unsorted or nonpositive event timing.')
        previous=onset
        end=onset+duration
        if i+1<len(rows):
            next_start=float(rows[i+1]['onset'])
            if end-next_start>1.5/fs: raise ValueError('Cue events overlap.')
            end=min(end,next_start)
        a=round(onset*fs);b=min(n_samples,round(end*fs));label=canonical(r['trial_type'])
        spans.append(dict(cue=i,start=a,end=b,label=classes.index(label) if label in classes else -1,
                          name=label,clipped=round(end*fs)>n_samples))
    return spans

def extract_windows(filtered,events,config):
    n=round(config['window_s']*config['sfreq']);stride=round(config['stride_s']*config['sfreq'])
    settle=round(config['settle_s']*config['sfreq']);spans=cue_spans(events,filtered.shape[1],config)
    xs=[];records=[];counts=Counter();bad=(abs(filtered)>config['artifact_limit_v']).any(axis=0)
    for cue in spans:
        if cue['label']<0: counts['excluded_other_class_cues']+=1;continue
        counts['selected_cues']+=1
        if cue['end']-cue['start']<settle+n: counts['short_cues']+=1
        kept=0;cue_begin=len(xs)
        for a in range(cue['start']+settle,cue['end']-n+1,stride):
            b=a+n;window=filtered[:,a:b];counts['candidate_windows']+=1
            if not np.isfinite(window).all(): counts['nonfinite_windows']+=1;continue
            if bad[a:b].any(): counts['artifact_windows']+=1;continue
            if np.any(window.std(axis=1)<config['min_signal_std_v']): counts['flat_windows']+=1;continue
            xs.append(window.copy());records.append(dict(start=a,end=b,label=cue['label'],cue=cue['cue']))
            counts['kept_windows']+=1;kept+=1
        if 0<kept<config.get('min_windows_per_cue',1):
            del xs[cue_begin:];del records[cue_begin:]
            counts['kept_windows']-=kept;counts['insufficient_clean_cue_windows']+=1
        elif kept: counts['kept_cues']+=1
    return xs,records,dict(counts),spans

def prepare(root,config,output):
    c=validate_config(config);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    channels=MONTAGES[c['montage']];fs=c['sfreq'];sos=butter(4,[8,30],fs=fs,btype='bandpass',output='sos')
    xs=[];records=[];runs=[];failures=[];seen={}
    for partition,key in [('train','train_sessions'),('validation','validation_sessions'),('test','test_sessions')]:
        for ses in c[key]:
            folder=Path(root)/f"sub-{int(c['subject']):02d}"/f'ses-{ses:02d}'/'eeg'
            paths=sorted(folder.glob(f"*task-{c['task']}_run-*_events.tsv"))
            if not paths: raise ValueError(f'No task events for session {ses}.')
            for event_path in paths:
                edf=event_path.with_name(event_path.name.replace('_events.tsv','_eeg.edf'))
                run=edf.stem.removesuffix('_eeg')
                try:
                    events=pd.read_csv(event_path,sep='\t')
                    if events.empty: raise ValueError('Empty events table.')
                    raw=mne.io.read_raw_edf(edf,preload=False,verbose=False)
                    if not np.isclose(raw.info['sfreq'],fs) or not set(channels)<=set(raw.ch_names):
                        raise ValueError('EEG sample rate/montage mismatch.')
                    data=raw.get_data(picks=channels).astype(np.float32)
                    if not np.isfinite(data).all(): raise ValueError('Nonfinite EDF data.')
                    signal_sha=hashlib.sha256(data.tobytes()).hexdigest()
                    if signal_sha in seen: raise ValueError(f'Duplicate EEG signal with {seen[signal_sha]}; resolve before fitting.')
                    seen[signal_sha]=run
                    spans=cue_spans(events,data.shape[1],c)
                    # Check the independent annotations stored with the EDF, when present.
                    annotations=[(float(a),canonical(d)) for a,d in zip(raw.annotations.onset,raw.annotations.description)]
                    if not annotations: raise ValueError('EDF lacks cue annotations; verify timing manually before inclusion.')
                    for r in events.to_dict('records'):
                        if not any(abs(float(r['onset'])-a)<=1.5/fs and canonical(r['trial_type'])==d for a,d in annotations):
                            raise ValueError('EDF annotations disagree with event onset/label.')
                    filtered=sosfilt(sos,data,axis=1).astype(np.float32)
                    windows,details,counts,spans=extract_windows(filtered,events,c)
                    for window,r in zip(windows,details):
                        r.update(run=run,session=ses,partition=partition,trial=f"{run}:cue-{r['cue']}")
                        xs.append(window);records.append(r)
                    runs.append(dict(run=run,session=ses,partition=partition,n_samples=data.shape[1],
                        edf_sha256=digest(edf),events_sha256=digest(event_path),signal_sha256=signal_sha,
                        counts=counts,event_classes=dict(Counter(s['name'] for s in spans)),
                        clipped_cues=sum(s['clipped'] for s in spans),
                        original_channel_order=raw.ch_names,annotations_checked=True,
                        reference='Not specified by the available EDF header; verify acquisition configuration.'))
                    print(f"{partition} {run}: {len(windows)} windows / {counts.get('kept_cues',0)} cues",flush=True)
                except Exception as exc:
                    failures.append(dict(run=run,partition=partition,error=f'{type(exc).__name__}: {exc}'))
                    print(f'EXCLUDED {run}: {exc}',flush=True)
    manifest=dict(config=c,channels=channels,runs=runs,failures=failures,partitions={})
    for partition in ['train','validation','test']:
        rows=[r for r in records if r['partition']==partition]
        counts={name:sum(r['label']==i for r in rows) for i,name in enumerate(c['classes'])}
        trials={name:len(set(r['trial'] for r in rows if r['label']==i)) for i,name in enumerate(c['classes'])}
        manifest['partitions'][partition]=dict(windows=counts,clean_cue_trials=trials,
            runs=len(set(r['run'] for r in rows)),sessions=sorted(set(r['session'] for r in rows)))
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (output/'records.json').write_text(json.dumps(records),encoding='utf-8')
    if not xs: raise ValueError('No eligible signal windows; review preparation failures.')
    np.savez_compressed(output/'windows.npz',X=np.stack(xs),y=np.array([r['label'] for r in records],np.int64))
    for partition,p in manifest['partitions'].items():
        if min(p['clean_cue_trials'].values())<3: raise ValueError(f'{partition}: fewer than three clean trials per class.')
    # Timing/header/duplicate faults require review, not silent best-effort exclusions.
    if failures: raise ValueError(f'{len(failures)} run failures: review manifest before training.')
    return manifest
