"""Causal EEGNet decoder. Input is strictly channels x samples in volts."""
from pathlib import Path
import hashlib
import json
import socket
import time
import numpy as np
import torch
from scipy.signal import butter, sosfilt
from networks import EEGNet

class RingBuffer:
    def __init__(self, n_ch, n_samples):
        self.n_ch, self.n_samples = n_ch, n_samples
        self.buf = np.zeros((n_ch, n_samples), dtype=np.float32)
        self.idx = 0
    def push_chunk(self, chunk):
        chunk = np.asarray(chunk)
        if chunk.ndim != 2 or chunk.shape[0] != self.n_ch:
            raise ValueError('Expected channels x samples; transpose at the source adapter.')
        n = chunk.shape[1]
        if n >= self.n_samples:
            self.idx += n
            self.buf[:] = np.roll(chunk[:, -self.n_samples:], self.idx % self.n_samples, axis=1)
            return
        i = self.idx % self.n_samples
        first = min(n, self.n_samples-i)
        self.buf[:, i:i+first] = chunk[:, :first]
        self.buf[:, :n-first] = chunk[:, first:]
        self.idx += n
    def push(self, sample): self.push_chunk(np.asarray(sample)[:, None])
    def get(self):
        i = self.idx % self.n_samples
        return np.concatenate((self.buf[:, i:], self.buf[:, :i]), axis=1)
    @property
    def is_full(self): return self.idx >= self.n_samples

def load_contract(checkpoint, contract_path=None):
    path = Path(contract_path or str(checkpoint)+'.json')
    if not path.exists():
        raise ValueError(f'Missing model contract: {path}. See README_LAB_GUIDE.md.')
    c = json.loads(path.read_text(encoding='utf-8-sig'))
    for key in ['channels','sfreq','n_times','normalization','filter','classes','verified']:
        if key not in c: raise ValueError(f'Model contract is missing {key}')
    if c.get('sha256') != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
        raise ValueError('Checkpoint SHA256 does not match its model contract.')
    if c['classes'] != ['rest','action']: raise ValueError('Expected output classes [rest, action].')
    if not c['channels'] or len(set(x.casefold() for x in c['channels'])) != len(c['channels']):
        raise ValueError('Model channel names must be unique and nonempty.')
    if c['normalization'] not in ('per_window','fixed'): raise ValueError('Invalid normalization mode.')
    if c['sfreq'] <= 0 or c['n_times'] < 32: raise ValueError('Invalid sample rate or window length.')
    return c

class RealtimeBCIDecoder:
    def __init__(self, ckpt_path, sfreq=None, epoch_len_s=None, step_len_s=.25,
                 threshold=None, ema_alpha=.3, dwell_steps=2, fmin=8., fmax=30.,
                 udp_host='127.0.0.1', udp_port=5005, device='cpu',
                 contract_path=None, allow_unverified=False, send_udp=True):
        self.ckpt_path = Path(ckpt_path).resolve()
        self.contract = load_contract(self.ckpt_path, contract_path)
        if threshold is None: threshold=float(self.contract.get('threshold',.5))
        if not self.contract['verified'] and not allow_unverified:
            raise ValueError('Legacy model preprocessing/provenance is unverified. Use a newly trained '
                             'contract or set allow_unverified_model for exploration only.')
        self.channels = self.contract['channels']
        self.sfreq = float(self.contract['sfreq'])
        self.window_samples = int(self.contract['n_times'])
        if sfreq is not None and not np.isclose(sfreq, self.sfreq): raise ValueError('Sample rate differs from contract; no implicit resampling.')
        if epoch_len_s is not None and round(epoch_len_s*self.sfreq) != self.window_samples: raise ValueError('Window length differs from contract.')
        if not 0 < ema_alpha <= 1 or dwell_steps < 1 or not 0 < threshold < 1: raise ValueError('Invalid decoder parameters.')
        self.step_samples = round(step_len_s*self.sfreq)
        if not 1 <= self.step_samples <= self.window_samples: raise ValueError('Invalid step length.')
        self.ema_alpha, self.dwell_steps, self.threshold = ema_alpha, dwell_steps, float(threshold)
        self.device = torch.device(device)
        # User-supplied trusted local checkpoints contain numpy arrays.
        ckpt = torch.load(self.ckpt_path, map_location=self.device, weights_only=False)
        self.n_ch = int(ckpt['n_channels'])
        if self.n_ch != len(self.channels) or int(ckpt['n_times']) != self.window_samples: raise ValueError('Checkpoint dimensions differ from contract.')
        if ckpt.get('arch','eegnet') != 'eegnet': raise ValueError('Live engine supports EEGNet; export an EEGNet checkpoint.')
        weights = dict(ckpt['model_state'])
        for key in ('weight','bias'):
            if f'classifier.{key}' in weights: weights[f'fc.{key}'] = weights.pop(f'classifier.{key}')
        self.model = EEGNet(n_ch=self.n_ch, n_t=self.window_samples).to(self.device)
        self.model.load_state_dict(weights)
        self.model.eval()
        self.train_mean = np.asarray(ckpt.get('train_mean',np.zeros(self.n_ch)),dtype=np.float32)
        self.train_std = np.asarray(ckpt.get('train_std',np.ones(self.n_ch)),dtype=np.float32)
        if self.train_mean.shape != (self.n_ch,) or self.train_std.shape != (self.n_ch,): raise ValueError('Normalization shape mismatch.')
        if not np.isfinite(self.train_mean).all() or not np.isfinite(self.train_std).all() or np.any(self.train_std <= 0): raise ValueError('Invalid normalization statistics.')
        filt = self.contract['filter']
        if filt['mode'] != 'causal' and not allow_unverified: raise ValueError('Model was not trained with causal preprocessing.')
        self.sos = butter(int(filt.get('order',4)),[filt['fmin'],filt['fmax']],btype='bandpass',fs=self.sfreq,output='sos')
        self.artifact_limit = float(self.contract.get('artifact_limit_v',100e-6))
        self.min_signal_std = float(self.contract.get('min_signal_std_v',1e-9))
        self.sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM) if send_udp else None
        self.udp_target = (udp_host,int(udp_port))
        self.packets_sent = 0
        self.reset_state()
    def reset_state(self):
        self.ring = RingBuffer(self.n_ch,self.window_samples)
        self.quality_ring = RingBuffer(1,self.window_samples)
        self.zi = np.zeros((len(self.sos),self.n_ch,2))
        self.samples_seen = 0
        self.next_prediction = self.window_samples
        self.ema_prob, self.ema_ready, self.dwell_counter = 0.,False,0
        self.last_filtered = None
        self.last_chunk_filtered = None
    def normalize_window(self, window):
        if self.contract['normalization'] == 'per_window':
            return (window-window.mean(axis=-1,keepdims=True))/(window.std(axis=-1,keepdims=True)+1e-8)
        return (window-self.train_mean[:,None])/(self.train_std[:,None]+1e-8)
    def set_user_baseline(self, mean, std, threshold):
        raise ValueError('Normalization belongs to the trained checkpoint. Fit a personal model using labeled calibration cues.')
    def send(self, packet):
        if self.sock is not None:
            self.sock.sendto(json.dumps(packet,allow_nan=False).encode(),self.udp_target)
            self.packets_sent += 1
    def send_rest(self, reason='paused'):
        self.send(dict(timestamp=time.time(),p_raw=0.,p_smooth=0.,grasp_command=0,continuous_flex=0.,state_label='REST',quality=reason))
    def process_chunk_all(self, raw_chunk, emit=True):
        # A random architecture template can collect EEG, but cannot control a hand.
        emit = emit and self.contract.get('role') != 'personal_training_template'
        raw = np.asarray(raw_chunk,dtype=np.float32)
        if raw.ndim != 2 or raw.shape[0] != self.n_ch: raise ValueError(f'Expected ({self.n_ch}, samples) volts; got {raw.shape}.')
        if not np.isfinite(raw).all():
            self.reset_state()
            if emit: self.send_rest('nonfinite_input')
            raise ValueError('Nonfinite EEG samples; stream reset.')
        outputs, pos, filter_parts = [],0,[]
        while pos < raw.shape[1]:
            count = min(raw.shape[1]-pos,self.next_prediction-self.samples_seen)
            part = raw[:,pos:pos+count]
            # Reject artifacts in the same band used by training. Clipping raw
            # offsets before the high-pass destroys otherwise usable EEG.
            filtered,self.zi = sosfilt(self.sos,part,axis=1,zi=self.zi)
            filter_parts.append(filtered.astype(np.float32))
            bad = (np.abs(filtered)>self.artifact_limit).any(axis=0).astype(np.float32)
            self.ring.push_chunk(filtered)
            self.quality_ring.push_chunk(bad[None,:])
            self.samples_seen += count
            pos += count
            if self.samples_seen != self.next_prediction: continue
            self.next_prediction += self.step_samples
            window = self.ring.get()
            self.last_filtered = window
            quality = 'artifact' if self.quality_ring.get().any() else 'flat_signal' if np.any(window.std(axis=1)<self.min_signal_std) else 'ok'
            p_action = 0.
            if quality == 'ok':
                with torch.inference_mode():
                    tensor=torch.from_numpy(self.normalize_window(window)).unsqueeze(0).to(self.device)
                    p_action=float(torch.softmax(self.model(tensor),dim=1)[0,1].cpu())
                if not np.isfinite(p_action): raise ValueError('Model returned nonfinite probability.')
                self.ema_prob=self.ema_alpha*p_action+(1-self.ema_alpha)*self.ema_prob if self.ema_ready else p_action
                self.ema_ready=True
                self.dwell_counter=self.dwell_counter+1 if self.ema_prob>=self.threshold else 0
            else: self.ema_prob,self.dwell_counter,self.ema_ready=0.,0,False
            decision=int(quality=='ok' and self.dwell_counter>=self.dwell_steps)
            packet=dict(timestamp=time.time(),sample_end=self.samples_seen,p_raw=p_action,p_smooth=self.ema_prob,
                        grasp_command=decision,continuous_flex=float(np.clip((self.ema_prob-self.threshold)/(1-self.threshold),0,1)) if decision else 0.,
                        state_label='ACTION' if decision else 'REST',quality=quality,threshold=self.threshold)
            if emit: self.send(packet)
            outputs.append(packet)
        self.last_chunk_filtered=np.concatenate(filter_parts,axis=1) if filter_parts else np.empty((self.n_ch,0),np.float32)
        return outputs
    def process_chunk(self,raw_chunk):
        outputs=self.process_chunk_all(raw_chunk)
        return outputs[-1] if outputs else None
    def close(self):
        if self.sock is not None:
            try: self.send_rest('shutdown')
            finally:
                self.sock.close()
                self.sock=None

def main():
    from bci_studio_gui import main as studio_main
    studio_main()
if __name__=='__main__': main()
