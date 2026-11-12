from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import pytest
import torch
from networks import EEGNet
from realtime_bci_decoder import RingBuffer,RealtimeBCIDecoder
from lsl_acquisition import StreamAdapter,select_stream
from protocols import CueProtocol,score_trials
from train_lab_model import event_windows,split_groups
from bci_studio_gui import Studio

torch.set_num_threads(1)
CHANNELS=['F3','Fz','F4','C3','Cz','C4','P3','Pz','P4','POz','O1','O2']
@pytest.fixture
def checkpoint(tmp_path):
    torch.manual_seed(10);p=tmp_path/'test.pt';m=EEGNet(12,512)
    torch.save(dict(model_state=m.state_dict(),arch='eegnet',n_channels=12,n_times=512,
                    train_mean=np.zeros(12,dtype='float32'),train_std=np.full(12,5e-6,dtype='float32')),p)
    c=dict(channels=CHANNELS,sfreq=256,n_times=512,normalization='fixed',
           filter=dict(mode='causal',fmin=8.,fmax=30.,order=4),classes=['rest','action'],verified=True,
           sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    Path(str(p)+'.json').write_text(json.dumps(c),encoding='utf-8')
    return p,c

def decoder(checkpoint,**kwargs): return RealtimeBCIDecoder(checkpoint[0],send_udp=False,**kwargs)

def test_ring_wrap_and_oversize():
    r=RingBuffer(2,11);x=np.arange(102).reshape(2,51)
    r.push_chunk(x[:,:3]);r.push_chunk(x[:,3:7]);r.push_chunk(x[:,7:40]);r.push_chunk(x[:,40:])
    np.testing.assert_array_equal(r.get(),x[:,-11:])

@pytest.mark.parametrize('chunk_size',[1,12,48,64,127,10000])
def test_chunk_size_invariance(checkpoint,chunk_size):
    x=np.random.default_rng(4).normal(0,5e-6,(12,900)).astype('float32')
    a=decoder(checkpoint);b=decoder(checkpoint)
    expected=a.process_chunk_all(x,emit=False);actual=[]
    for i in range(0,x.shape[1],chunk_size): actual+=b.process_chunk_all(x[:,i:i+chunk_size],emit=False)
    assert [o['sample_end'] for o in actual]==list(range(512,901,64))
    np.testing.assert_allclose([o['p_raw'] for o in expected],[o['p_raw'] for o in actual],atol=1e-6)
    np.testing.assert_allclose(a.ring.get(),b.ring.get(),atol=1e-12)

def test_invalid_shapes_nonfinite_and_warmup(checkpoint):
    d=decoder(checkpoint)
    assert d.process_chunk_all(np.zeros((12,511)))==[]
    with pytest.raises(ValueError): d.process_chunk_all(np.zeros((64,12)))
    with pytest.raises(ValueError): d.process_chunk_all(np.full((12,1),np.nan))
    assert d.samples_seen==0

def test_quality_neutralizes_artifact_and_flatline(checkpoint):
    d=decoder(checkpoint,threshold=.01,dwell_steps=1)
    x=np.random.default_rng(2).normal(0,5e-6,(12,1024)).astype('float32')
    assert d.process_chunk_all(x[:,:512])[-1]['grasp_command']==1
    x[0,512]=.001
    results=d.process_chunk_all(x[:,512:])
    assert results[0]['quality']=='artifact' and results[0]['continuous_flex']==0.
    d.reset_state()
    o=d.process_chunk_all(np.zeros((12,512)))[0]
    assert o['quality']=='flat_signal' and o['grasp_command']==0

def test_dwell_applies_to_flex_and_reset(checkpoint):
    d=decoder(checkpoint,threshold=.01,dwell_steps=2)
    x=np.random.default_rng(2).normal(0,5e-6,(12,600)).astype('float32')
    out=d.process_chunk_all(x)
    assert out[0]['grasp_command']==0 and out[0]['continuous_flex']==0
    assert out[1]['grasp_command']==1
    d.reset_state();assert d.dwell_counter==0 and d.samples_seen==0

def test_contract_hash_and_verification(checkpoint):
    p,c=checkpoint;c['verified']=False;Path(str(p)+'.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='unverified'): decoder(checkpoint)
    assert decoder(checkpoint,allow_unverified=True)
    c['sha256']='wrong';Path(str(p)+'.json').write_text(json.dumps(c))
    with pytest.raises(ValueError,match='SHA256'): decoder(checkpoint,allow_unverified=True)

def metadata(labels,units=None,sfreq=256):
    return dict(name='test',source_id='test',sfreq=sfreq,channel_count=len(labels),labels=labels,units=units or ['uV']*len(labels))

def test_reorders_30_channels_and_converts_microvolts(checkpoint):
    labels=['Epoch','Offset','Hour','Min','Sec','MilliSec']+CHANNELS[::-1]+['EKG','AUX1','AUX2','AUX3']
    a=StreamAdapter(metadata(labels),checkpoint[1]);x=np.tile(np.arange(len(labels)),(64,1))
    ts=10+np.arange(64)/256
    out,_=a.transform(x,ts,ts[-1])
    np.testing.assert_allclose(out[:,0],np.array([labels.index(c) for c in CHANNELS])*1e-6)

@pytest.mark.parametrize('case',['missing','duplicate','units','rate'])
def test_rejects_incompatible_metadata(checkpoint,case):
    labels=CHANNELS.copy();units=['uV']*12;rate=256
    if case=='missing': labels[0]='Fp1'
    if case=='duplicate': labels[0]='C3'
    if case=='units': units=['']*12
    if case=='rate': rate=500
    with pytest.raises(ValueError): StreamAdapter(metadata(labels,units,rate),checkpoint[1])

@pytest.mark.parametrize('bad_ts,now',[(np.arange(64)/256,2.),(np.zeros(64),0.),(10+np.arange(64)/256,9.)])
def test_rejects_stale_and_invalid_clock(checkpoint,bad_ts,now):
    a=StreamAdapter(metadata(CHANNELS),checkpoint[1])
    with pytest.raises(ValueError): a.transform(np.ones((64,12)),bad_ts,now)

def test_detects_stream_gap(checkpoint):
    a=StreamAdapter(metadata(CHANNELS),checkpoint[1]);ts=np.arange(64)/256
    a.transform(np.ones((64,12)),ts,ts[-1])
    with pytest.raises(ValueError,match='dropped'): a.transform(np.ones((64,12)),ts+2,ts[-1]+2)

def test_selection_requires_unambiguous_source():
    class Info:
        def name(self): return 'test'
        def source_id(self): return 'test'
    with pytest.raises(ValueError): select_stream([Info(),Info()],{})
    with pytest.raises(ValueError): select_stream([],{})

def test_short_rest_windows_stay_in_one_cue():
    events=pd.DataFrame([dict(sample=0,duration=3.,trial_type='rest'),dict(sample=832,duration=5.,trial_type='action')])
    spans=event_windows(np.zeros((2,3000)),events,n_times=512)
    rest=[s for s in spans if s[2]==0]
    assert rest and all(b<=768 for a,b,_,_ in rest)
    assert not [s for s in event_windows(np.zeros((2,3000)),events,n_times=1024) if s[2]==0]

def test_partitions_have_no_group_overlap():
    g=np.repeat(np.arange(11),20);y=np.tile([0,1],110)
    tr,va,te=split_groups(y,g)
    assert not(set(g[tr])&set(g[va]) or set(g[tr])&set(g[te]) or set(g[va])&set(g[te]))

def test_calibration_requires_actual_clean_predictions():
    p=CueProtocol('calibration',0,2.,trials_per_class=2)
    for t in (8,16,24,32): p.advance(t)
    with pytest.raises(ValueError,match='insufficient'): p.finish(.5)

def test_cue_windows_after_transition_only():
    p=CueProtocol('calibration',0,2.,trials_per_class=2)
    packet=dict(p_smooth=.8,grasp_command=1,quality='ok')
    p.add(packet,2.);assert not p.rows
    p.add(packet,3.);assert len(p.rows)==1
    p.advance(8.);p.add(packet,9.);assert len(p.rows)==1

def test_chance_is_not_high_accuracy():
    result=score_trials([0,1]*10,[1]*20,permutations=50)
    assert result['balanced_accuracy']==.5 and result['permutation_p']==1.

def test_studio_has_no_fake_calibration_and_freezes_validation(checkpoint,tmp_path):
    p,c=checkpoint;s=Studio(dict(checkpoint=str(p)),root=tmp_path,clock=lambda:100.)
    assert s.snapshot()['calib_acc'] is None
    with pytest.raises(ValueError): s.action('/api/calibrate',{})
    s.action('/api/toggle_demo',{});s.action('/api/toggle_stream',{})
    with pytest.raises(ValueError,match='Synthetic'): s.action('/api/validate',{})
    s.action('/api/calibrate',{})
    with pytest.raises(ValueError,match='frozen'): s.action('/api/set_threshold',{'val':['.7']})
    s.action('/api/stop',{});assert s.protocol is None
    s.decoder.close()

def test_session_saves_full_rate_data_and_threshold_at_each_row(checkpoint,tmp_path):
    p,c=checkpoint;s=Studio(dict(checkpoint=str(p)),root=tmp_path,clock=lambda:100.)
    s.action('/api/toggle_demo',{});s.action('/api/toggle_stream',{})
    x=np.random.default_rng(3).normal(0,5e-6,(12,640)).astype('float32');ts=97+np.arange(640)/256
    s.ingest(x,ts,100.)
    s.action('/api/set_threshold',{'val':['.7']});s.pause()
    saved=s.save();base=Path(s.state['last_saved'])
    data=np.load(str(base)+'_raw.npz')
    assert data['eeg_volts'].shape==(12,640)
    df=pd.read_csv(str(base)+'_predictions.csv');assert set(df.threshold)=={.5}
    meta=json.loads(Path(str(base)+'_metadata.json').read_text())
    assert meta['source']=='synthetic_demo' and not meta['calibration_is_validation']
    assert base.with_name(base.name+'_plot.png').exists()

def test_calibration_is_computed_from_actual_rows():
    p=CueProtocol('calibration',0,2.,trials_per_class=2)
    for i in range(4):
        label=int(p.labels[p.index])
        for t in (3.,4.,5.,6.,7.):
            p.add(dict(p_smooth=.8 if label else .2,grasp_command=label,quality='ok'),p.start+t)
        p.advance(p.start+8.)
    result=p.finish(.5)
    assert result['kind']=='calibration_fit_only' and result['fit_balanced_accuracy']==1.
    assert .2<result['threshold']<=.8


def test_filtered_offset_does_not_destroy_eeg(checkpoint):
    from scipy.signal import sosfilt
    d=decoder(checkpoint)
    t=np.arange(3072)/256
    x=np.tile(500e-6+10e-6*np.sin(2*np.pi*12*t),(12,1)).astype('float32')
    expected=sosfilt(d.sos,x,axis=1)
    outputs=d.process_chunk_all(x)
    np.testing.assert_allclose(d.ring.get(),expected[:,-512:],atol=1e-10)
    assert outputs[-1]['quality']=='ok'

def test_trial_group_split_is_stratified():
    groups=np.repeat(np.arange(20),8);y=np.repeat(np.tile([0,1],10),8)
    train,val,test=split_groups(y,groups)
    for indices in (train,val,test): assert set(y[indices])=={0,1}
    assert not(set(groups[train])&set(groups[val]) or set(groups[train])&set(groups[test]))


def test_personal_training_excludes_validation_and_reloads(checkpoint,tmp_path):
    from train_personal_model import load_personal
    from train_lab_model import train_export
    p,c=checkpoint;base=tmp_path/'personal_fixture'
    fs=256;n=20*8*fs;t=np.arange(n)/fs
    raw=np.tile(7e-6*np.sin(2*np.pi*12*t),(12,1)).astype('float32')
    np.savez_compressed(str(base)+'_raw.npz',eeg_volts=raw,lsl_timestamps=t,channels=np.array(CHANNELS),sfreq=fs)
    # Artificial test fixture, not participant data or performance evidence.
    Path(str(base)+'_metadata.json').write_text(json.dumps(dict(source='live')))
    pd.DataFrame([dict(run_id='fixture',kind='calibration' if i<18 else 'validation',trial=i+1,
                       label=i%2,start=i*8.,end=(i+1)*8.) for i in range(20)]).to_csv(str(base)+'_events.csv',index=False)
    X,y,g,records,channels=load_personal(base,p)
    assert len(set(g))==18 and all(int(r['run'].split('_')[-1])<=18 for r in records)
    out=tmp_path/'personal.pt'
    result=train_export(X,y,g,records,out,channels,epochs=1,initial_checkpoint=p,freeze_backbone=True)
    restored=RealtimeBCIDecoder(out,send_udp=False)
    parent=RealtimeBCIDecoder(p,send_udp=False)
    for key,value in restored.model.block1.state_dict().items():
        assert torch.equal(value,parent.model.block1.state_dict()[key])
    assert set(result['partitions']['train']).isdisjoint(result['partitions']['test'])
    meta=json.loads(Path(str(out)+'.json').read_text());assert meta['personal_model']
    Path(str(base)+'_metadata.json').write_text(json.dumps(dict(source='synthetic_demo')))
    with pytest.raises(ValueError,match='LIVE'): load_personal(base,p)

def test_threshold_never_changes_model_probabilities(checkpoint):
    a=decoder(checkpoint,threshold=.2);b=decoder(checkpoint,threshold=.8)
    x=np.random.default_rng(7).normal(0,5e-6,(12,768)).astype('float32')
    aa=a.process_chunk_all(x);bb=b.process_chunk_all(x)
    np.testing.assert_array_equal([r['p_raw'] for r in aa],[r['p_raw'] for r in bb])
    np.testing.assert_array_equal([r['p_smooth'] for r in aa],[r['p_smooth'] for r in bb])


def test_uses_checkpoint_threshold_by_default(checkpoint):
    p,c=checkpoint;c['threshold']=.87
    Path(str(p)+'.json').write_text(json.dumps(c))
    assert decoder(checkpoint).threshold==.87
    assert decoder(checkpoint,threshold=.4).threshold==.4


def test_untrained_template_cannot_emit_predictions_or_validate(tmp_path):
    from create_personal_template import create_template
    import socket
    p=tmp_path/'template.pt';c=create_template(p)
    receiver=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1',0));receiver.settimeout(.1)
    d=RealtimeBCIDecoder(p,udp_port=receiver.getsockname()[1])
    try:
        d.process_chunk_all(np.random.default_rng(2).normal(0,5e-6,(20,640)))
        with pytest.raises(socket.timeout): receiver.recvfrom(8192)
        assert d.packets_sent==0
    finally: d.close();receiver.close()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(('127.0.0.1', 0))
        receiver.settimeout(.1)
        s = Studio(dict(checkpoint=str(p), udp_port=receiver.getsockname()[1]), root=tmp_path, clock=lambda:100.)
        try:
            s.state['is_connected'] = True
            s.action('/api/toggle_stream')
            data = np.random.default_rng(2).normal(0, 5e-6, (20, 640)).astype('float32')
            s.ingest(data, 96.+np.arange(640)/256)
            with pytest.raises(socket.timeout):
                receiver.recvfrom(8192)
            with pytest.raises(ValueError, match='template'):
                s.action('/api/validate')
        finally:
            s.decoder.close()
            s.udp_sock.close()


def test_scope_uses_every_filtered_sample_and_replay_is_labeled(checkpoint,tmp_path,monkeypatch):
    import mne
    p,c=checkpoint
    x=np.random.default_rng(11).normal(0,5e-6,(12,640)).astype('float32')
    class Recording:
        info={'sfreq':256}
        def get_data(self,picks):
            assert picks==CHANNELS
            return x
    monkeypatch.setattr(mne.io,'read_raw_edf',lambda *args,**kwargs:Recording())
    s=Studio(dict(checkpoint=str(p),replay_edf=str(tmp_path/'fixture_eeg.edf')),root=tmp_path,clock=lambda:100.)
    assert s.state['source']=='recorded_replay' and s.decoder.sock is None
    s.action('/api/toggle_stream',{})
    with pytest.raises(ValueError,match='Recorded replay'): s.action('/api/validate',{})
    s.ingest(x,97+np.arange(640)/256,100.)
    from scipy.signal import sosfilt
    expected=sosfilt(s.decoder.sos,x,axis=1)[CHANNELS.index('C3'),-512:]*1e6
    np.testing.assert_allclose(s.snapshot()['wave_c3'],expected,rtol=1e-5,atol=1e-5)
    assert np.std(s.snapshot()['wave_c3'])>1.

def test_gui_trains_personal_nn_then_freezes_it_for_fresh_validation(checkpoint,tmp_path):
    p,c=checkpoint;now=[1000.]
    s=Studio(dict(checkpoint=str(p),personalize_on_calibration=True,require_personal_calibration=True,
        personal_calibration_pairs=9,personal_training_epochs=1,personal_initialization='scratch'),root=tmp_path,clock=lambda:now[0])
    s.state['is_connected']=True;s.action('/api/toggle_stream',{})
    with pytest.raises(ValueError,match='personal calibration'): s.action('/api/validate',{})
    s.action('/api/calibrate',{'participant':['P001']})
    assert len(s.protocol.labels)==18
    for i in range(18):
        start=s.protocol.start;label=int(s.protocol.labels[s.protocol.index])
        t=np.arange(2048)/256
        raw=np.tile((4e-6 if not label else 12e-6)*np.sin(2*np.pi*12*t),(12,1)).astype('float32')
        s.ingest(raw,start+t,start+8.)
        now[0]=start+8.;s.refresh_protocol(now[0])
    assert s.training_thread is not None
    s.training_thread.join(timeout=30.)
    assert not s.training_thread.is_alive()
    assert s.state['personalized'],s.state['error']
    assert not s.state['is_training'] and s.state['participant_id']=='P001'
    assert s.checkpoint!=p
    assert s.contract['initialization']=='scratch' and s.contract['participant_id']=='P001'
    assert 'parent_sha256' not in s.contract
    assert s.state['calibration_result']['kind']=='personal_nn_calibration'
    s.action('/api/toggle_stream',{});s.action('/api/validate',{})
    assert s.protocol.kind=='validation'
    before=s.contract['sha256']
    with pytest.raises(ValueError,match='frozen'): s.action('/api/set_threshold',{'val':['.3']})
    assert s.contract['sha256']==before
    s.action('/api/stop',{});s.decoder.close()
