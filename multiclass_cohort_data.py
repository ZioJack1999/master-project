"""Recover native event annotations and build private, window-covariance caches."""
from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
import mne
from scipy.signal import butter,sosfilt
from multiclass_data import canonical,cue_spans,extract_windows,digest,GESTURES
from multiclass_features import band_covariances
from create_personal_template import MONTAGES

def events_for_recording(raw,event_path,config):
    event_path=Path(event_path);events=None
    try:events=pd.read_csv(event_path,sep='\t')
    except (FileNotFoundError,pd.errors.EmptyDataError):pass
    recovered=events is None or events.empty
    annotations=pd.DataFrame(dict(onset=raw.annotations.onset,duration=raw.annotations.duration,trial_type=[canonical(s) for s in raw.annotations.description]))
    if recovered:
        if not config.get('recover_empty_events_from_edf'):raise ValueError('Missing/empty events; recovery is disabled.')
        if annotations.empty:raise ValueError('No recorded annotations from which to recover labels.')
        if not set(annotations.trial_type)<=set(GESTURES):raise ValueError('Unknown native annotation labels; manual review required.')
        events=annotations.copy()
    else:
        if not {'onset','duration','trial_type'}<=set(events):raise ValueError('Event table missing required columns.')
        events=events.copy();events['trial_type']=events.trial_type.map(canonical)
        for r in events.to_dict('records'):
            matches=annotations[(abs(annotations.onset-float(r['onset']))<=1.5/config['sfreq'])&(annotations.trial_type==r['trial_type'])]
            if matches.empty:raise ValueError('TSV/EDF onset or class disagreement.')
    cue_spans(events,raw.n_times,config)
    return events,('recovered_native_edf_annotations' if recovered else 'original_tsv_matched_to_edf')

def prepare_cohort(root,config,work):
    root=Path(root);work=Path(work);work.mkdir(parents=True,exist_ok=True)
    if (work/'study_results.json').exists() or (work/'cases').exists():raise ValueError('Study fitting has begun; preserve its frozen input data.')
    channels=MONTAGES[config['montage']];fs=config['sfreq'];sos=butter(4,[8,30],fs=fs,btype='bandpass',output='sos')
    chunks=[];records=[];runs=[];failures=[];excluded=[];seen={}
    for subject in config['subjects']:
        for session in config['sessions'][subject]:
            paths=sorted((root/f'sub-{subject}'/f'ses-{session:02d}'/'eeg').glob(f"*task-{config['task']}_run-*_eeg.edf"))
            if not paths:raise ValueError(f'No EDF runs for {subject}/{session}.')
            for edf in paths:
                run=edf.stem.removesuffix('_eeg');event_path=edf.with_name(run+'_events.tsv')
                if run in config.get('excluded_runs',{}):excluded.append(dict(run=run,reason=config['excluded_runs'][run]));continue
                try:
                    raw=mne.io.read_raw_edf(edf,preload=False,verbose=False)
                    if not np.isclose(raw.info['sfreq'],fs) or not set(channels)<=set(raw.ch_names):raise ValueError('Sample rate or electrode mismatch.')
                    events,source=events_for_recording(raw,event_path,config)
                    signal=raw.get_data(picks=channels).astype(np.float32)
                    if not np.isfinite(signal).all():raise ValueError('Nonfinite EEG.')
                    import hashlib
                    signal_hash=hashlib.sha256(signal.tobytes()).hexdigest()
                    if signal_hash in seen:raise ValueError(f'Duplicate raw EEG with {seen[signal_hash]}; do not treat duplicate IDs as independent people.')
                    seen[signal_hash]=run
                    filtered=sosfilt(sos,signal,axis=1).astype(np.float32)
                    xs,details,counts,spans=extract_windows(filtered,events,config)
                    if xs:
                        chunks.append(band_covariances(np.stack(xs),fs))
                        for r in details:
                            r.update(subject=subject,session=session,run=run,trial=f"{run}:cue-{r['cue']}",label_name=config['classes'][r['label']])
                            records.append(r)
                    event_hash=digest(event_path) if event_path.exists() else None
                    if source=='recovered_native_edf_annotations':
                        folder=work/'recovered_events'/f'sub-{subject}';folder.mkdir(parents=True,exist_ok=True)
                        events.to_csv(folder/event_path.name,sep='\t',index=False)
                    runs.append(dict(subject=subject,session=session,run=run,events_source=source,
                        edf_sha256=digest(edf),original_events_sha256=event_hash,signal_sha256=signal_hash,
                        counts=counts,cue_counts=dict(Counter(events.trial_type)),n_samples=signal.shape[1],
                        reference='Physical reference not established by EDF labels.',
                        annotation_check='Source of recovered labels; no independent TSV corroboration.' if source.startswith('recovered') else 'TSV onset/class matched to stored EDF annotations.'))
                    print(f"QA {subject}/{session:02d} {run.split('_run-')[-1]}: {len(xs)} windows; {source}",flush=True)
                except Exception as exc:
                    failures.append(dict(subject=subject,session=session,run=run,error=f'{type(exc).__name__}: {exc}'))
                    print('FAILED',run,str(exc),flush=True)
    manifest=dict(config=config,channels=channels,runs=runs,failures=failures,declared_exclusions=excluded,subjects={})
    for subject in config['subjects']:
        rr=[r for r in records if r['subject']==subject]
        manifest['subjects'][subject]=dict(runs=sum(r['subject']==subject for r in runs),
            recovered_event_files=sum(r['subject']==subject and r['events_source'].startswith('recovered') for r in runs),
            clean_cues={name:len({r['trial'] for r in rr if r['label_name']==name}) for name in config['classes']},
            windows=len(rr))
    (work/'cohort_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    if failures:raise ValueError(f'{len(failures)} run failures; review before fitting any model.')
    if not chunks:raise ValueError('No clean windows.')
    np.save(work/'covariances.npy',np.concatenate(chunks))
    (work/'records.json').write_text(json.dumps(records),encoding='utf-8')
    print(json.dumps(manifest['subjects'],indent=2),flush=True);return manifest

def load_cohort(work):
    work=Path(work);manifest=json.loads((work/'cohort_manifest.json').read_text())
    if manifest['failures']:raise ValueError('Unresolved preparation failures.')
    records=json.loads((work/'records.json').read_text());cov=np.load(work/'covariances.npy',mmap_mode='r')
    if len(records)!=len(cov):raise ValueError('Cohort covariance/provenance mismatch.')
    return cov,records,manifest
