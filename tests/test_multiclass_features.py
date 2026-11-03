import json
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy.signal import butter,sosfilt
from multiclass_data import digest
from multiclass_features import band_covariances,FeaturePreprocessor,FeatureNetwork,FeatureEnsembleRunner
from multiclass_decoder import MulticlassDecoder
from improve_multiclass import search
from calibrate_multiclass import select_bias
from test_multiclass import config

torch.set_num_threads(1)

def test_covariance_has_frequency_and_spatial_information():
    t=np.arange(512)/256
    x=np.stack([np.sin(2*np.pi*10*t),2*np.sin(2*np.pi*10*t)])*1e-5
    cov=band_covariances(x[None])
    assert cov.shape==(1,4,2,2)
    assert cov[0,0,0,0]>100*cov[0,3,0,0]
    assert cov[0,0,0,1]==pytest.approx(2*cov[0,0,0,0],rel=1e-6)
    assert np.linalg.eigvalsh(cov).min()>-1e-15

def test_class_offsets_are_selected_from_development_rows_and_zero_wins_tie():
    rows=[dict(trial=f't{i}',run=f'run{i}',label=i) for i in range(3)]
    p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]])
    best,candidates=select_bias(p,np.arange(3),rows,['rest','volar_grasp','finger_extension'],[-.5,0.,.5])
    assert best['bias']==[0.,0.,0.] and len(candidates)==9

@pytest.mark.parametrize('family',['filterbank_power','log_covariance'])
def test_feature_network_trains_and_streaming_matches_offline(tmp_path,family):
    raw=np.random.default_rng(10).normal(0,5e-6,(12,4,512)).astype('float32')
    filtered=sosfilt(butter(4,[8,30],fs=256,btype='bandpass',output='sos'),raw,axis=-1).astype('float32')
    cov=band_covariances(filtered);pre=FeaturePreprocessor.fit(cov,np.ones(12),family)
    model=FeatureNetwork(family,4,3,pre.parameters)
    loss=torch.nn.functional.cross_entropy(model(torch.from_numpy(pre.transform(cov))),torch.arange(12)%3);loss.backward()
    assert model.head[-1].weight.grad.abs().sum()>0
    if family=='filterbank_power':assert model.spatial.grad.abs().sum()>0
    payload=dict(arch='feature_ensemble',family=family,n_channels=4,n_classes=3,n_times=512,
        hidden_units=32,dropout=.25,preprocessing=pre.parameters,member_states=[model.state_dict()],train_mean=np.zeros(4),train_std=np.ones(4))
    p=tmp_path/'feature.pt';torch.save(payload,p)
    c=dict(schema_version=2,task_mode='multiclass',classes=['rest','volar_grasp','finger_extension'],channels=['A','B','C','D'],
        sfreq=256,n_times=512,normalization='feature_train_only',filter=dict(mode='causal',fmin=8.,fmax=30.,order=4),
        sha256=digest(p),artifact_limit_v=100e-6,min_signal_std_v=1e-9)
    Path(str(p)+'.json').write_text(json.dumps(c))
    expected=FeatureEnsembleRunner(payload).predict(filtered[:1])[0]
    decoder=MulticlassDecoder(p);rows=[]
    for a in range(0,512,31):rows.extend(decoder.process_chunk_all(raw[0,:,a:a+31]))
    np.testing.assert_allclose(rows[0]['probabilities_raw'],expected,atol=2e-6)

def test_candidate_search_never_features_sealed_test_windows(tmp_path,monkeypatch):
    c=config();c.update(experiment_id='synthetic_search_fixture',subject='08',task='olrandmcmitrain',
        target_balanced_accuracy=.7,candidate_window_seconds=[2.,4.],candidate_montages=['all20'],
        candidate_architectures=['filterbank_power','log_covariance'],ensemble_seeds=[42],seed=42,
        threads=1,hidden_units=8,dropout=.25,epochs=1,patience=1,batch_size=16,learning_rate=.001,
        weight_decay=.01,covariance_shrinkage=.1,selection_metric='declared_fixture')
    cfg=tmp_path/'config.json';cfg.write_text(json.dumps(c));work=tmp_path/'experiment'
    for window in [2,4]:
        folder=work/f'window_{window}s';folder.mkdir(parents=True);rows=[];ys=[]
        for part,ses in [('train',21),('validation',26),('test',28)]:
            for label in range(3):
                for cue in range(3):
                    rows.append(dict(partition=part,session=ses,label=label,trial=f'{ses}_{label}_{cue}',run=f'run{ses}'))
                    ys.append(label)
        X=np.random.default_rng(window).normal(0,5e-6,(len(ys),4,window*256)).astype('float32')
        X[np.array([r['partition']=='test' for r in rows])]=.001
        np.savez_compressed(folder/'windows.npz',X=X,y=np.array(ys,dtype='int64'))
        (folder/'records.json').write_text(json.dumps(rows))
        (folder/'manifest.json').write_text(json.dumps(dict(config=dict(c,window_s=window),channels=['A','B','C','D'],failures=[],runs=[],partitions={})))
    import improve_multiclass
    original=improve_multiclass.band_covariances
    def guarded(X):
        assert np.max(abs(X))<1e-4,'Test signal leaked into development feature fitting.'
        return original(X)
    monkeypatch.setattr(improve_multiclass,'band_covariances',guarded)
    result=search(work,config_path=cfg)
    assert not result['test_scored'] and len(result['candidates'])==4
    assert (work/'selected'/'frozen.json').exists()
    assert not (work/'selected'/'test_results.json').exists()
