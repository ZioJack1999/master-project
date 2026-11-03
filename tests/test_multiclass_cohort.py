import json
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from multiclass_cohort_data import events_for_recording
from study_multiclass_transfer import person_class_cue_weights,case_indices,trial_subsample,run_case

CLASSES=['rest','volar_grasp','finger_extension']

def annotation_raw(labels=None):
    labels=labels if labels is not None else ['Rest','Volar Grasp','Finger Extension']
    return SimpleNamespace(n_times=20*256,annotations=SimpleNamespace(
        onset=np.array([0.,6.,12.])[:len(labels)],duration=np.full(len(labels),6.),description=labels))

def test_empty_event_recovery_uses_actual_annotations_and_preserves_original(tmp_path):
    p=tmp_path/'empty.tsv';original=b'onset\tduration\ttrial_type\n';p.write_bytes(original)
    c=dict(sfreq=256,classes=CLASSES,recover_empty_events_from_edf=True)
    events,source=events_for_recording(annotation_raw(),p,c)
    assert list(events.trial_type)==CLASSES and list(events.duration)==[6,6,6]
    assert source=='recovered_native_edf_annotations' and p.read_bytes()==original
    with pytest.raises(ValueError,match='No recorded'):events_for_recording(annotation_raw([]),p,c)
    with pytest.raises(ValueError,match='Unknown'):events_for_recording(annotation_raw(['Peace']),p,c)

def test_nonempty_event_conflict_is_not_repaired_silently(tmp_path):
    p=tmp_path/'events.tsv';pd.DataFrame(dict(onset=[1.],duration=[6.],trial_type=['Rest'])).to_csv(p,sep='\t',index=False)
    with pytest.raises(ValueError,match='disagreement'):events_for_recording(annotation_raw(),p,dict(sfreq=256,classes=CLASSES,recover_empty_events_from_edf=True))

def fixture_records():
    records=[]
    for subject in ['04','07','08']:
        for session in [1,2]:
            for label,name in enumerate(CLASSES):
                for cue in range(3):
                    for window in range(2):
                        records.append(dict(subject=subject,session=session,run=f'{subject}_{session}_{cue}',
                            trial=f'{subject}_{session}_{label}_{cue}',label_name=name,label=label))
    return records

def test_held_person_never_enters_fit_or_source_validation():
    records=fixture_records();c=dict(source_validation_sessions={s:[2] for s in ['04','07','08']})
    idx,y=case_indices(records,CLASSES,c,held_subject='07')
    assert {records[i]['subject'] for i in idx['test']}=={'07'}
    assert {records[i]['subject'] for p in ['train','validation'] for i in idx[p]}=={'04','08'}
    assert not set(idx['train'])&set(idx['validation'])
    subset=trial_subsample(idx['train'],y,records,.25,42)
    for trial in {records[i]['trial'] for i in subset}:
        assert {i for i in idx['train'] if records[i]['trial']==trial}<={int(i) for i in subset}

def test_person_class_cue_weights_balance_large_and_small_people():
    records=fixture_records();records=[r for r in records if r['subject']!='07']
    records += [dict(r) for r in records if r['subject']=='04']
    y=np.array([r['label'] for r in records]);w=person_class_cue_weights(y,records)
    for subject in ['04','08']:
        for label in range(3):assert sum(v for v,r in zip(w,records) if r['subject']==subject and r['label']==label)==pytest.approx(1/6)
    records[0]['trial']=records[2]['trial']
    # Explicitly corrupt one cue's true label.
    y[0]=(y[2]+1)%3
    with pytest.raises(ValueError,match='mixes'):person_class_cue_weights(y,records)

def test_fold_freezes_before_touching_target_covariances(tmp_path,monkeypatch):
    records=fixture_records();c=dict(source_validation_sessions={s:[2] for s in ['04','07','08']},
        classes=CLASSES,family='log_covariance',hidden_units=8,dropout=.25,covariance_shrinkage=.1,
        ensemble_seeds=[42],epochs=1,patience=1,batch_size=16,learning_rate=.001,weight_decay=.01,
        diagnostic_interval=1,draws_per_training_cue=2,window_s=4.,sfreq=256,
        artifact_limit_v=100e-6,min_signal_std_v=1e-9,experiment_id='synthetic_transfer_fixture',seed=42)
    idx,y=case_indices(records,CLASSES,c,held_subject='07')
    random=np.random.default_rng(1).normal(size=(len(records),4,4,6));cov=(random@random.swapaxes(-1,-2)).astype('float32')*1e-11
    cov[idx['test']]*=1e8
    import study_multiclass_transfer
    original=study_multiclass_transfer.fit_payload
    def guarded(train_cov,train_y,train_rows,val_cov,*args,**kwargs):
        assert np.max(abs(train_cov))<1e-5 and np.max(abs(val_cov))<1e-5
        assert all(r['subject']!='07' for r in train_rows)
        return original(train_cov,train_y,train_rows,val_cov,*args,**kwargs)
    monkeypatch.setattr(study_multiclass_transfer,'fit_payload',guarded)
    r=run_case(tmp_path,'fixture',CLASSES,idx,y,cov,records,dict(channels=['A','B','C','D']),c,dict(fixture=True),held_subject='07')
    assert r['test_subjects']==['07'] and r['train_subjects']==['04','08']
    assert (tmp_path/'cases'/'fixture'/'frozen.json').exists()
    assert r['training_diagnostics']['seed_histories'][0]['epochs'][0]['train_cue_loss']>=0
