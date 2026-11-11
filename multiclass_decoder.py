"""Sample-clocked multiclass inference. No actuator transport is enabled here."""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.signal import butter,sosfilt
from networks import EEGNet
from temporal_networks import TemporalEEGEnsemble, TemporalEEGNet
from realtime_bci_decoder import RingBuffer
from dual_band_filter import DualBandFilter, DualBandRingBuffer
from model_utils import digest

DEFAULT_POLICY=dict(ema_alpha=.3,dwell_updates=2,min_confidence=.5,min_margin=.05)

HIERARCHICAL_ARCHS = (
    'hierarchical_dual_band',
    'hierarchical_two_stage',
    'hierarchical_eegnet',
    'HierarchicalTwoStageEEGNet',
    'hierarchical_ensemble',
    'HierarchicalEnsemble',
    'HierarchicalTwoStageDecoder',
)

DUAL_BAND_CNN_ARCHS = (
    'dual_band_cnn',
    'DualBandTemporalEEGNet',
    'dual_band_temporal_cnn',
)


def build_hierarchical_diagnostics(arch, alignment_active, p_rest, p_move, p_dir_grasp, p_dir_ext,
                                   gate_threshold=0.50, mrcp_band=(0.3, 3.0), smr_band=(4.0, 40.0),
                                   quality='ok'):
    gate_open = bool(quality == 'ok' and p_move >= gate_threshold)
    p_grasp = float(p_move * p_dir_grasp) if quality == 'ok' else 0.0
    p_ext = float(p_move * p_dir_ext) if quality == 'ok' else 0.0
    stage1 = dict(
        p_rest=float(p_rest),
        p_movement=float(p_move),
        gate_threshold=float(gate_threshold),
        gate_open=gate_open,
        decision='movement' if gate_open else 'rest',
        band_hz=[float(smr_band[0]), float(smr_band[1])],
    )
    stage2 = dict(
        p_grasp_conditional=float(p_dir_grasp),
        p_extension_conditional=float(p_dir_ext),
        p_grasp_joint=p_grasp,
        p_extension_joint=p_ext,
        margin=float(abs(p_dir_grasp - p_dir_ext)),
        decision=('volar_grasp' if p_dir_grasp >= p_dir_ext else 'finger_extension') if quality == 'ok' else 'uncertain',
        mrcp_band_hz=[float(mrcp_band[0]), float(mrcp_band[1])],
        smr_band_hz=[float(smr_band[0]), float(smr_band[1])],
    )
    hierarchical = dict(
        active=True,
        arch=str(arch),
        resting_alignment_active=bool(alignment_active),
        quality=str(quality),
        stage1=stage1,
        stage2=stage2,
        p_gate_rest=float(p_rest),
        p_gate_movement=float(p_move),
        p_dir_grasp=float(p_dir_grasp),
        p_dir_extension=float(p_dir_ext),
        gated=not gate_open,
    )
    return stage1, stage2, hierarchical


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
    def __init__(self,checkpoint,policy=None,device='cpu'):
        self.device = torch.device(device)
        path=Path(checkpoint);c=json.loads(Path(str(path)+'.json').read_text())
        if (c.get('task_mode') not in ('multiclass', 'binary') and c.get('role') != 'personal_training_template') or (c.get('schema_version', 2) not in (1, 2) and c.get('role') != 'personal_training_template') or len(c['classes']) < 2:
            raise ValueError('Multiclass or binary schema checkpoint required.')
        if c['sha256']!=digest(path): raise ValueError('Multiclass model SHA256 mismatch.')
        filt = c.get('filter', {'mode': 'causal', 'order': 4, 'fmin': 8.0, 'fmax': 30.0})
        if filt.get('mode', 'causal') != 'causal' or filt.get('order', 4) not in (2, 4) or c.get('normalization', 'fixed') not in ('fixed', 'per_window', 'feature_train_only', 'unsupervised_resting_alignment', 'resting_alignment', 'euclidean_alignment'):
            raise ValueError('Unsupported multiclass preprocessing.')
        fmin = float(filt.get('fmin', 8.0))
        fmax = float(filt.get('fmax', 30.0))
        ck=torch.load(path,map_location='cpu',weights_only=False)
        self.contract=c;self.channels=c['channels'];self.classes=c['classes'];self.n_ch=len(self.channels)
        self.sfreq=c['sfreq'];self.window_samples=c.get('n_times', ck.get('n_times', 512));self.step_samples=round(.25*self.sfreq)
        if ck['n_channels']!=self.n_ch or ck.get('n_classes', len(self.classes))!=len(self.classes) or ck.get('n_times', self.window_samples)!=self.window_samples:
            raise ValueError('Multiclass dimensions differ from contract.')
        self.arch=ck.get('arch', c.get('arch', 'eegnet'));self.feature_runner=None;self.alignment=None
        self.is_hierarchical = self.arch in HIERARCHICAL_ARCHS
        self.is_dual_band = (
            self.is_hierarchical
            or (self.arch in DUAL_BAND_CNN_ARCHS)
            or bool(filt.get('dual_band'))
            or ('mrcp' in ck.get('bands', {}))
        )
        self.gate_threshold = float(ck.get('gate_threshold', c.get('gate_threshold', 0.50)))

        # Optional Unsupervised Resting Alignment / Euclidean Alignment
        align_state = ck.get('resting_alignment') or ck.get('alignment') or c.get('resting_alignment')
        if align_state:
            from hierarchical_classifier import UnsupervisedRestingAlignment
            self.alignment = UnsupervisedRestingAlignment().load_state_dict(align_state)

        # Dual-Band causal filter bank (0.3-3.0 Hz MRCP + 4.0-40.0 Hz SMR)
        if self.is_dual_band:
            bands = ck.get('bands', {})
            mrcp_band = tuple(filt.get('mrcp_band', bands.get('mrcp', (0.3, 3.0))))
            smr_band = tuple(filt.get('smr_band', bands.get('smr', (4.0, 40.0))))
            mrcp_order = int(filt.get('mrcp_order', 2))
            smr_order = int(filt.get('smr_order', 4))
            self.dual_filter = DualBandFilter(
                sfreq=self.sfreq,
                mrcp_band=(float(mrcp_band[0]), float(mrcp_band[1])),
                mrcp_order=mrcp_order,
                smr_band=(float(smr_band[0]), float(smr_band[1])),
                smr_order=smr_order,
            )
        else:
            self.dual_filter = None

        if self.arch=='feature_ensemble':
            if c['normalization']!='feature_train_only':raise ValueError('Feature ensemble requires train-only feature normalization.')
            from multiclass_features import FeatureEnsembleRunner
            self.feature_runner=FeatureEnsembleRunner(ck,device=self.device);self.model=None
        elif self.arch=='temporal_cnn':
            if c.get('arch') != self.arch or c['normalization'] not in ('fixed', 'unsupervised_resting_alignment', 'resting_alignment', 'euclidean_alignment') or not ck.get('member_states'):
                raise ValueError('Temporal CNN requires fixed normalization and trained members.')
            standardize = ck.get('window_standardization', False)
            if not isinstance(standardize, bool) or standardize != c.get('window_standardization', False):
                raise ValueError('Temporal CNN window standardization differs from the contract.')
            self.model = TemporalEEGEnsemble(self.n_ch, len(self.classes), ck['member_states'], standardize).to(self.device).eval()
        elif self.arch in ('TemporalEEGNet', 'temporal_eegnet'):
            standardize = ck.get('window_standardization', True)
            self.model = TemporalEEGNet(self.n_ch, len(self.classes), window_standardization=standardize)
            self.model.load_state_dict(ck['model_state'])
            self.model.to(self.device).eval()
        elif self.is_hierarchical:
            from hierarchical_classifier import HierarchicalTwoStageEEGNet, HierarchicalEnsemble
            standardize = ck.get('window_standardization', True)
            if 'stage1_state' in ck and 'stage2_state' in ck:
                from multisubject_pipeline import DualBandTemporalEEGNet, HierarchicalTwoStageDecoder
                s1 = TemporalEEGNet(self.n_ch, 2, window_standardization=standardize)
                s1.load_state_dict(ck['stage1_state'])
                s2 = DualBandTemporalEEGNet(self.n_ch, 2, window_standardization=standardize)
                s2.load_state_dict(ck['stage2_state'])
                self.model = HierarchicalTwoStageDecoder(
                    s1, s2, rest_threshold=1.0 - self.gate_threshold,
                    classes=self.classes, device=str(self.device)
                )
            elif ck.get('member_states'):
                self.model = HierarchicalEnsemble(
                    self.n_ch, ck['member_states'], classes=self.classes,
                    window_standardization=standardize
                ).to(self.device).eval()
            else:
                self.model = HierarchicalTwoStageEEGNet(
                    self.n_ch,
                    classes=self.classes,
                    window_standardization=standardize,
                    sfreq=self.sfreq,
                )
                self.model.load_state_dict(ck['model_state'])
                self.model.to(self.device).eval()
        elif self.arch in DUAL_BAND_CNN_ARCHS:
            from multisubject_pipeline import DualBandTemporalEEGNet
            standardize = ck.get('window_standardization', True)
            if ck.get('member_states'):
                self.model = torch.nn.ModuleList([
                    DualBandTemporalEEGNet(self.n_ch, len(self.classes), window_standardization=standardize)
                    for _ in ck['member_states']
                ])
                for m, st in zip(self.model, ck['member_states']):
                    m.load_state_dict(st)
                self.model.to(self.device).eval()
            else:
                self.model = DualBandTemporalEEGNet(
                    self.n_ch, len(self.classes), window_standardization=standardize
                )
                self.model.load_state_dict(ck['model_state'])
                self.model.to(self.device).eval()
        elif self.arch=='eegnet':
            if c['normalization'] not in ('fixed', 'unsupervised_resting_alignment', 'resting_alignment', 'euclidean_alignment'):
                raise ValueError('EEGNet requires fixed normalization.')
            self.model = EEGNet(self.n_ch, self.window_samples, len(self.classes))
            self.model.load_state_dict(ck['model_state'])
            self.model.to(self.device).eval()
        else:raise ValueError('Unknown multiclass model architecture.')
        self.mean=np.asarray(ck.get('train_mean', np.zeros(self.n_ch)),np.float32);self.std=np.asarray(ck.get('train_std', np.ones(self.n_ch)),np.float32)
        if self.mean.shape!=(self.n_ch,) or self.std.shape!=(self.n_ch,) or not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or np.any(self.std<1e-9):raise ValueError('Invalid multiclass normalization.')
        self.sos=butter(int(filt.get('order', 4)),[fmin,fmax],fs=self.sfreq,btype='bandpass',output='sos')
        self.policy=MulticlassPolicy(len(self.classes),policy or c.get('policy'));self.reset()
    def reset(self):
        self.ring=RingBuffer(self.n_ch,self.window_samples);self.raw_ring=RingBuffer(self.n_ch,self.window_samples);self.bad=RingBuffer(1,self.window_samples)
        if self.dual_filter is not None:
            self.dual_ring = DualBandRingBuffer(
                self.n_ch,
                self.window_samples,
                dual_filter=self.dual_filter,
                artifact_limit_v=float(self.contract.get('artifact_limit_v', 100e-6)),
                min_signal_std_v=float(self.contract.get('min_signal_std_v', 1e-9)),
            )
        else:
            self.dual_ring = None
        self.last_mrcp_window = None
        self.last_smr_window = None
        self.zi=np.zeros((len(self.sos),self.n_ch,2));self.samples_seen=0;self.next_prediction=self.window_samples;self.policy.reset()
    def _predict_hierarchical(self, smr_x, mrcp_x):
        smr_t = torch.from_numpy(smr_x[None]).to(self.device, dtype=torch.float32)
        mrcp_t = torch.from_numpy(mrcp_x[None]).to(self.device, dtype=torch.float32)
        if hasattr(self.model, 'stage1_gating') and hasattr(self.model, 'stage2_directional'):
            logits_gate = self.model.stage1_gating(smr_t)
            logits_dir = self.model.stage2_directional(smr_t, mrcp_t)
            probs_gate = torch.softmax(logits_gate, dim=1)[0].cpu().numpy()
            probs_dir = torch.softmax(logits_dir, dim=1)[0].cpu().numpy()
        elif hasattr(self.model, 'stage1_model') and hasattr(self.model, 'stage2_model'):
            mb_t = torch.cat([mrcp_t, smr_t], dim=1)
            logits_gate = self.model.stage1_model(smr_t)
            logits_dir = self.model.stage2_model(mb_t)
            probs_gate = torch.softmax(logits_gate, dim=1)[0].cpu().numpy()
            probs_dir = torch.softmax(logits_dir, dim=1)[0].cpu().numpy()
        else:
            # HierarchicalEnsemble
            gate_list = [torch.softmax(m.stage1_gating(smr_t), dim=1) for m in self.model]
            dir_list = [torch.softmax(m.stage2_directional(smr_t, mrcp_t), dim=1) for m in self.model]
            probs_gate = torch.stack(gate_list, dim=0).mean(dim=0)[0].cpu().numpy()
            probs_dir = torch.stack(dir_list, dim=0).mean(dim=0)[0].cpu().numpy()
        p_rest = float(probs_gate[0])
        p_move = float(probs_gate[1])
        p_dir_grasp = float(probs_dir[0])
        p_dir_ext = float(probs_dir[1])
        if len(self.classes) == 2:
            p = np.array([p_rest, p_move], dtype=float)
        else:
            p = np.array([p_rest, p_move * p_dir_grasp, p_move * p_dir_ext], dtype=float)
        p = p / max(1e-12, float(p.sum()))
        return p, p_rest, p_move, p_dir_grasp, p_dir_ext
    def process_chunk_all(self,raw):
        raw=np.asarray(raw,np.float32)
        if raw.ndim!=2 or raw.shape[0]!=self.n_ch:raise ValueError('Expected channels x samples in volts.')
        if not np.isfinite(raw).all():self.reset();raise ValueError('Nonfinite EEG; multiclass state reset.')
        position=0;outputs=[]
        while position<raw.shape[1]:
            n=min(raw.shape[1]-position,self.next_prediction-self.samples_seen)
            part=raw[:,position:position+n]
            filtered,self.zi=sosfilt(self.sos,part,axis=1,zi=self.zi)
            self.ring.push_chunk(filtered);self.raw_ring.push_chunk(part);self.bad.push_chunk((abs(filtered)>self.contract.get('artifact_limit_v', 100e-6)).any(axis=0)[None,:])
            if self.dual_ring is not None:
                self.dual_ring.push_chunk(part)
            self.samples_seen+=n;position+=n
            if self.samples_seen!=self.next_prediction:continue
            self.next_prediction+=self.step_samples;x=self.ring.get();raw_x=self.raw_ring.get()
            if self.dual_ring is not None:
                dual_win = self.dual_ring.get_windows()
                mrcp_x = dual_win['mrcp']
                smr_x = dual_win['smr']
                self.last_mrcp_window = mrcp_x
                self.last_smr_window = smr_x
                quality = dual_win['quality']
            else:
                mrcp_x = None
                smr_x = None
                quality='artifact' if self.bad.get().any() else 'flat_signal' if np.any(x.std(axis=1)<self.contract.get('min_signal_std_v', 1e-9)) else 'ok'
            p=np.zeros(len(self.classes))
            stage1_diag = stage2_diag = hierarchical_diag = None
            if quality=='ok':
                if self.feature_runner:
                    x_in = self.alignment.transform(x) if self.alignment is not None else x
                    p=self.feature_runner.predict(x_in[None])[0]
                elif self.is_hierarchical:
                    with torch.inference_mode():
                        if self.alignment is not None:
                            if hasattr(self.alignment, 'transform_dual_band'):
                                mrcp_in, smr_in = self.alignment.transform_dual_band(mrcp_x, smr_x)
                            elif isinstance(self.alignment, dict):
                                mrcp_in = self.alignment['mrcp'].transform(mrcp_x) if 'mrcp' in self.alignment else mrcp_x
                                smr_in = self.alignment['smr'].transform(smr_x) if 'smr' in self.alignment else smr_x
                            else:
                                smr_in = self.alignment.transform(smr_x)
                                mrcp_in = self.alignment.transform(mrcp_x)
                        else:
                            smr_in = smr_x
                            mrcp_in = mrcp_x
                        p, p_rest, p_move, p_dir_grasp, p_dir_ext = self._predict_hierarchical(smr_in, mrcp_in)
                        stage1_diag, stage2_diag, hierarchical_diag = build_hierarchical_diagnostics(
                            self.arch, self.alignment is not None, p_rest, p_move, p_dir_grasp, p_dir_ext,
                            gate_threshold=self.gate_threshold,
                            mrcp_band=self.dual_filter.mrcp_band,
                            smr_band=self.dual_filter.smr_band,
                            quality=quality,
                        )
                elif self.arch in DUAL_BAND_CNN_ARCHS:
                    with torch.inference_mode():
                        if self.alignment is not None:
                            if hasattr(self.alignment, 'transform_dual_band'):
                                mrcp_in, smr_in = self.alignment.transform_dual_band(mrcp_x, smr_x)
                            elif isinstance(self.alignment, dict):
                                mrcp_in = self.alignment['mrcp'].transform(mrcp_x) if 'mrcp' in self.alignment else mrcp_x
                                smr_in = self.alignment['smr'].transform(smr_x) if 'smr' in self.alignment else smr_x
                            else:
                                smr_in = self.alignment.transform(smr_x)
                                mrcp_in = self.alignment.transform(mrcp_x)
                        else:
                            smr_in = smr_x
                            mrcp_in = mrcp_x
                        mb_x = np.concatenate([mrcp_in, smr_in], axis=0)
                        batch = torch.from_numpy(mb_x[None]).to(self.device, dtype=torch.float32)
                        if isinstance(self.model, torch.nn.ModuleList):
                            p = torch.stack([torch.softmax(m(batch), dim=1) for m in self.model], dim=0).mean(dim=0)[0].cpu().numpy()
                        else:
                            p = torch.softmax(self.model(batch), dim=1)[0].cpu().numpy()
                else:
                    with torch.inference_mode():
                        x_in = self.alignment.transform(x) if self.alignment is not None else x
                        normalized = (x_in - self.mean[:, None]) / (self.std[:, None] + 1e-8)
                        batch = torch.from_numpy(normalized[None]).to(self.device)
                        p = torch.softmax(self.model(batch), dim=1)[0].cpu().numpy()
            elif self.is_hierarchical:
                stage1_diag, stage2_diag, hierarchical_diag = build_hierarchical_diagnostics(
                    self.arch, self.alignment is not None, 0.0, 0.0, 0.0, 0.0,
                    gate_threshold=self.gate_threshold,
                    mrcp_band=self.dual_filter.mrcp_band,
                    smr_band=self.dual_filter.smr_band,
                    quality=quality,
                )
            decision=self.policy.update(p,quality)
            pkt = dict(sample_end=self.samples_seen,quality=quality,probabilities_raw=p.tolist(),
                class_name=self.classes[decision['class_id']] if decision['class_id']>=0 else 'uncertain',**decision)
            if stage1_diag is not None:
                pkt['stage1_diagnostics'] = stage1_diag
                pkt['stage2_diagnostics'] = stage2_diag
                pkt['hierarchical_diagnostics'] = hierarchical_diag
            outputs.append(pkt)
        return outputs
