import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from networks import EEGNet
from multiclass_data import canonical,cue_spans,extract_windows,validate_config
from multiclass_metrics import cue_probabilities,evaluate,trial_class_weights
from train_multiclass import bandpower,baseline_probabilities,fit,test_once as score_once
from multiclass_decoder import MulticlassDecoder,MulticlassPolicy
from replay_multiclass import policy_metrics

def config():
    return dict(classes=['Rest','Volar Grasp','Finger_Extension'],train_sessions=[21],
        validation_sessions=[26],test_sessions=[28],sfreq=256,window_s=2.,stride_s=.25,settle_s=.5,
        montage='x24_stock20',artifact_limit_v=100e-6,min_signal_std_v=1e-9)

def test_class_names_and_session_isolation():
    c=validate_config(config());assert c['classes']==['rest','volar_grasp','finger_extension']
    c['test_sessions']=[21]
    with pytest.raises(ValueError,match='disjoint'):validate_config(c)
    c=config();c['classes']=['rest','Move','finger_extension']
    with pytest.raises(ValueError,match='protocol'):validate_config(c)

def test_multiclass_windows_never_cross_cues_or_recording_end():
    c=validate_config(config());t=np.arange(6*256)/256
    x=np.tile(10e-6*np.sin(2*np.pi*12*t),(20,1))
    events=pd.DataFrame([dict(onset=0,duration=3,trial_type='Rest'),dict(onset=3,duration=5,trial_type='Volar_Grasp')])
    xs,rows,counts,spans=extract_windows(x,events,c)
    assert spans[1]['clipped']
    assert all((r['start']>=128 and r['end']<=768) if r['label']==0 else (r['start']>=896 and r['end']<=1536) for r in rows)
    events['onset']=events['onset'].astype(float);events.loc[1,'onset']=2.9
    with pytest.raises(ValueError,match='overlap'):cue_spans(events,x.shape[1],c)

def test_other_gestures_and_artifacts_are_not_rest():
    c=validate_config(config());t=np.arange(6*256)/256
    x=np.tile(10e-6*np.sin(2*np.pi*12*t),(20,1));x[:,128:768]=.001
    events=pd.DataFrame([dict(onset=0,duration=3,trial_type='Rest'),dict(onset=3,duration=3,trial_type='Pinch Grasp')])
    xs,rows,counts,spans=extract_windows(x,events,c)
    assert not xs and not rows and counts['artifact_windows']>0
    assert counts['excluded_other_class_cues']==1

def test_weights_balance_classes_and_whole_cues():
    y=np.array([0]*8+[0]*2+[1]*3+[2]*5)
    rows=[dict(trial=t) for t,n in [('a',8),('b',2),('c',3),('d',5)] for _ in range(n)]
    w=trial_class_weights(y,rows)
    np.testing.assert_allclose([w[y==i].sum() for i in range(3)],1.)
    assert w[:8].sum()==pytest.approx(w[8:10].sum())

def test_metrics_count_cues_instead_of_overlapping_windows():
    rows=[dict(trial='a',run='run1',label=0)]*10+[dict(trial='b',run='run2',label=1),dict(trial='c',run='run3',label=2)]
    y=np.array([r['label'] for r in rows]);p=np.eye(3)[y]
    truth,probs,runs=cue_probabilities(p,rows)
    assert len(truth)==3
    result=evaluate(p,y,rows,['rest','volar_grasp','finger_extension'])
    assert result['cue_trials']['n']==3 and result['windows']['n']==12
    assert result['cue_trials']['balanced_accuracy']==1.

def test_three_class_network_and_bandpower_features():
    t=np.arange(512)/256;x=np.tile(np.sin(2*np.pi*10*t),(2,20,1)).astype('float32')*1e-5
    assert EEGNet(20,512,3)(torch.from_numpy(x)).shape==(2,3)
    f=bandpower(x).reshape(2,20,4)
    assert np.all(f[:,:,0]>f[:,:,3])

def test_uncertain_policy_dwell_and_fault_reset():
    policy=MulticlassPolicy(3)
    assert policy.update([.34,.33,.33])['class_id']==-1
    assert policy.update([.01,.98,.01])['class_id']==-1
    for _ in range(8): packet=policy.update([.01,.98,.01])
    assert packet['class_id']==1
    assert policy.update([0,0,0],'artifact')['class_id']==-1
    assert policy.update([.01,.98,.01])['class_id']==-1

def test_policy_metrics_rejects_count_as_misses_and_unknowns_stay_separate():
    rows=[dict(trial=f't{label}',label=label,quality='ok',class_id=label if label!=2 else -1) for label in range(3) for _ in range(4)]
    rows += [dict(trial='unknown',label=-1,quality='ok',class_id=1)]*4
    result=policy_metrics(rows,['rest','volar_grasp','finger_extension'])
    assert result['balanced_accuracy']==pytest.approx(2/3)
    assert result['confusion_matrix'][2][-1]==1
    assert result['other_gesture_acceptance_rate']==1.

def test_fit_freezes_before_one_time_test(tmp_path):
    # Synthetic fixture proves partition behavior, not participant performance.
    c=validate_config(config());c.update(seed=3,threads=1,dropout=.5,learning_rate=.001,batch_size=8,
        epochs=1,patience=1,experiment_id='synthetic_fixture')
    rows=[];y=[]
    for partition,ses in [('train',21),('validation',26),('test',28)]:
        for label in range(3):
            for cue in range(3):
                for window in range(2):
                    rows.append(dict(partition=partition,session=ses,label=label,run=f'{ses}_{cue}',trial=f'{ses}_{label}_{cue}'))
                    y.append(label)
    X=np.random.default_rng(3).normal(0,5e-6,(len(y),20,512)).astype('float32')
    np.savez_compressed(tmp_path/'windows.npz',X=X,y=np.array(y,dtype='int64'))
    (tmp_path/'records.json').write_text(json.dumps(rows))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(config=c,channels=[str(i) for i in range(20)],failures=[],partitions={})))
    fit(tmp_path)
    assert (tmp_path/'frozen.json').exists() and not (tmp_path/'test_results.json').exists()
    contract=tmp_path/'eegnet_multiclass.pt.json';original=contract.read_bytes()
    contract.write_bytes(original+b' ')
    with pytest.raises(ValueError,match='changed'):score_once(tmp_path)
    contract.write_bytes(original)
    decoder=MulticlassDecoder(tmp_path/'eegnet_multiclass.pt')
    whole=decoder.process_chunk_all(X[0,:,:512]);decoder.reset();chunks=[]
    for a in range(0,512,31):chunks.extend(decoder.process_chunk_all(X[0,:,a:a+31]))
    np.testing.assert_allclose(whole[0]['probabilities_raw'],chunks[0]['probabilities_raw'],atol=1e-7)
    result=score_once(tmp_path);assert result['eegnet']['cue_trials']['n']==9
    with pytest.raises(ValueError,match='already'):score_once(tmp_path)
    with pytest.raises(ValueError,match='exposed'):fit(tmp_path)
