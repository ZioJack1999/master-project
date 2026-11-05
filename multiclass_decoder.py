"""Sample-clocked multiclass inference. No actuator transport is enabled here."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import butter,sosfilt
from networks import EEGNet
from realtime_bci_decoder import RingBuffer
from model_utils import digest

DEFAULT_POLICY=dict(ema_alpha=.3,dwell_updates=2,min_confidence=.5,min_margin=.05)

class MulticlassPolicy:
    def __init__(self,n_classes,policy=None):
        self.n_classes=n_classes;self.config=dict(DEFAULT_POLICY,**(policy or {}));self.reset()
        if not 0<self.config['ema_alpha']<=1 or self.config['dwell_updates']<1 or not 0<=self.config['min_confidence']<=1 or not 0<=self.config['min_margin']<=1:
            raise ValueError('Invalid multiclass policy.')
    def reset(self):self.ema=None;self.candidate=-1;self.dwell=0
    def update(self,probabilities,quality='ok'):
        p=np.asarray(probabilities,dtype=float)
        if p.shape!=(self.n_classes,) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1.,atol=1e-5):
            if quality=='ok': raise ValueError('Invalid multiclass probability vector.')
        if quality!='ok':
            self.reset();return dict(class_id=-1,confidence=0.,margin=0.,probabilities_ema=[0.]*self.n_classes)
        alpha=self.config['ema_alpha'];self.ema=p if self.ema is None else alpha*p+(1-alpha)*self.ema
        order=np.argsort(self.ema);best=int(order[-1]);confidence=float(self.ema[best]);margin=float(confidence-self.ema[order[-2]])
        candidate=best if confidence>=self.config['min_confidence'] and margin>=self.config['min_margin'] else -1
        self.dwell=self.dwell+1 if candidate==self.candidate else 1;self.candidate=candidate
        decision=candidate if candidate>=0 and self.dwell>=self.config['dwell_updates'] else -1
        return dict(class_id=decision,confidence=confidence,margin=margin,probabilities_ema=self.ema.tolist())

class MulticlassDecoder:
    def __init__(self,checkpoint,policy=None):
        path=Path(checkpoint);c=json.loads(Path(str(path)+'.json').read_text())
        if c.get('task_mode')!='multiclass' or c.get('schema_version')!=2 or len(c['classes'])<3: raise ValueError('Multiclass schema v2 checkpoint required.')
        if c['sha256']!=digest(path): raise ValueError('Multiclass model SHA256 mismatch.')
        if c['filter']!=dict(mode='causal',fmin=8.,fmax=30.,order=4) or c['normalization'] not in ('fixed','feature_train_only'): raise ValueError('Unsupported multiclass preprocessing.')
        ck=torch.load(path,map_location='cpu',weights_only=False)
        self.contract=c;self.channels=c['channels'];self.classes=c['classes'];self.n_ch=len(self.channels)
        self.sfreq=c['sfreq'];self.window_samples=c['n_times'];self.step_samples=round(.25*self.sfreq)
        if ck['n_channels']!=self.n_ch or ck['n_classes']!=len(self.classes) or ck['n_times']!=self.window_samples: raise ValueError('Multiclass dimensions differ from contract.')
        self.arch=ck.get('arch','eegnet');self.feature_runner=None
        if self.arch=='feature_ensemble':
            if c['normalization']!='feature_train_only':raise ValueError('Feature ensemble requires train-only feature normalization.')
            from multiclass_features import FeatureEnsembleRunner
            self.feature_runner=FeatureEnsembleRunner(ck);self.model=None
        elif self.arch=='eegnet':
            if c['normalization']!='fixed':raise ValueError('EEGNet requires fixed normalization.')
            self.model=EEGNet(self.n_ch,self.window_samples,len(self.classes));self.model.load_state_dict(ck['model_state']);self.model.eval()
        else:raise ValueError('Unknown multiclass model architecture.')
        self.mean=np.asarray(ck['train_mean'],np.float32);self.std=np.asarray(ck['train_std'],np.float32)
        if self.mean.shape!=(self.n_ch,) or self.std.shape!=(self.n_ch,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<1e-9):raise ValueError('Invalid multiclass normalization.')
        self.sos=butter(4,[8,30],fs=self.sfreq,btype='bandpass',output='sos')
        self.policy=MulticlassPolicy(len(self.classes),policy or c.get('policy'));self.reset()
    def reset(self):
        self.ring=RingBuffer(self.n_ch,self.window_samples);self.bad=RingBuffer(1,self.window_samples)
        self.zi=np.zeros((len(self.sos),self.n_ch,2));self.samples_seen=0;self.next_prediction=self.window_samples;self.policy.reset()
    def process_chunk_all(self,raw):
        raw=np.asarray(raw,np.float32)
        if raw.ndim!=2 or raw.shape[0]!=self.n_ch:raise ValueError('Expected channels x samples in volts.')
        if not np.isfinite(raw).all():self.reset();raise ValueError('Nonfinite EEG; multiclass state reset.')
        position=0;outputs=[]
        while position<raw.shape[1]:
            n=min(raw.shape[1]-position,self.next_prediction-self.samples_seen)
            filtered,self.zi=sosfilt(self.sos,raw[:,position:position+n],axis=1,zi=self.zi)
            self.ring.push_chunk(filtered);self.bad.push_chunk((abs(filtered)>self.contract['artifact_limit_v']).any(axis=0)[None,:])
            self.samples_seen+=n;position+=n
            if self.samples_seen!=self.next_prediction:continue
            self.next_prediction+=self.step_samples;x=self.ring.get()
            quality='artifact' if self.bad.get().any() else 'flat_signal' if np.any(x.std(axis=1)<self.contract['min_signal_std_v']) else 'ok'
            p=np.zeros(len(self.classes))
            if quality=='ok':
                if self.feature_runner:p=self.feature_runner.predict(x[None])[0]
                else:
                    with torch.inference_mode():p=torch.softmax(self.model(torch.from_numpy((x-self.mean[:,None])/(self.std[:,None]+1e-8))[None]),dim=1)[0].numpy()
            decision=self.policy.update(p,quality)
            outputs.append(dict(sample_end=self.samples_seen,quality=quality,probabilities_raw=p.tolist(),
                class_name=self.classes[decision['class_id']] if decision['class_id']>=0 else 'uncertain',**decision))
        return outputs
