"""Multiclass LSL calibration/validation console. Display and recording only."""
import argparse
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse
import uuid
import webbrowser
import numpy as np
from scipy.signal import butter, sosfilt
from threeclass_acquisition import connect
from multiclass_decoder import MulticlassDecoder
from model_utils import digest
from realtime_bci_decoder import RingBuffer
from threeclass_cues import CLASSES, ThreeClassProtocol
from threeclass_model_store import list_personal_models, validate_personal_model
from threeclass_decisions import FIXED_POLICY as POLICY, cue_decision, mean_probability_decision

ROOT = Path(__file__).resolve().parent


def clock_now():
    from pylsl import local_clock
    return local_clock()


def score_validation(protocol, permutations=2000, seed=42):
    if not protocol.done or protocol.kind != 'validation' or not all(e['complete'] for e in protocol.events):
        raise ValueError('Only complete fresh validation runs can be scored.')
    classes = protocol.classes
    n_classes = len(classes)
    truth, decisions, comparisons, details = [], [], [], []
    for event in protocol.events:
        rows = [r for r in protocol.predictions if r['trial'] == event['trial']]
        outcome = cue_decision(rows, n_classes=n_classes)
        decision = outcome['class_id']
        comparison = mean_probability_decision(rows, n_classes=n_classes)
        truth.append(classes.index(event['label']))
        decisions.append(decision)
        comparisons.append(comparison)
        details.append(dict(trial=event['trial'], truth=event['label'],
            decision=classes[decision] if decision >= 0 else 'rejected',
            mean_probability_decision=classes[comparison] if comparison >= 0 else 'rejected',
            eligible_updates=outcome['eligible_updates'], accepted_updates=outcome['accepted_updates']))
    y, pred = np.array(truth), np.array(decisions)
    recall = {c:float(np.mean(pred[y == i] == i)) for i, c in enumerate(classes)}
    ba = float(np.mean(list(recall.values())))
    cm = np.zeros((n_classes, n_classes+1), dtype=int)
    for a, b in zip(y, pred):
        cm[a, b if b >= 0 else n_classes] += 1
    rng = np.random.default_rng(seed)
    null = []
    for _ in range(permutations):
        shuffled = rng.permutation(y)
        null.append(np.mean([np.mean(pred[shuffled == i] == i) for i in range(n_classes)]))
    bootstrap = []
    for _ in range(permutations):
        bootstrap.append(np.mean([np.mean(pred[rng.choice(np.flatnonzero(y == i), np.sum(y == i), replace=True)] == i)
                                  for i in range(n_classes)]))
    accepted = pred >= 0
    pooled = np.asarray(comparisons)
    return dict(kind='fresh_frozen_personal_validation', status='complete' if accepted.all() else 'complete_with_rejections',
        metrics=dict(balanced_accuracy=ba, accuracy=float(np.mean(y == pred)), recall=recall,
            n_cues=len(y), n_scored=int(accepted.sum()), n_rejected=int((~accepted).sum()),
            coverage=float(accepted.mean()), confusion_matrix=cm.tolist(),
            confusion_columns=classes+['rejected'],
            cue_stratified_bootstrap_95ci=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
            permutation_p=float((1+np.sum(np.array(null) >= ba-1e-12))/(1+permutations)),
            unit_of_analysis='One majority deployed decision per new cue. Rejected cues count as errors.',
            uncertainty_scope='Within-run cue bootstrap/permutations; limited pilot, not across-person/day confidence.'),
        comparison=dict(rule='Mean of clean window probabilities per cue',
            balanced_accuracy=float(np.mean([np.mean(pooled[y == i] == i) for i in range(n_classes)])),
            n_correct=int(np.sum(pooled == y)), n_rejected=int(np.sum(pooled < 0))),
        cues=details)


class ThreeClassStudio:
    def __init__(self, config, root=ROOT, clock=clock_now):
        self.root, self.config, self.clock = Path(root), dict(config), clock
        self.device = config.get('device', 'cpu')
        reference = config.get('checkpoint')
        self.reference_checkpoint = (self.root/reference).resolve() if reference else None
        self.checkpoint = self.reference_checkpoint
        default_classes = ['rest', 'right_hand_grasp'] if config.get('task') == 'binary' or config.get('udp_port') == 5005 or 'personal_calibration_pairs' in config or (reference and 'stock20' in str(reference)) else CLASSES
        self.classes = list(config.get('classes', default_classes))
        if len(self.classes) == 2:
            self.task = 'binary'
            self.task_label = 'binary'
        elif self.classes == CLASSES:
            self.task = 'threeclass'
            self.task_label = 'three-class'
        else:
            self.task = f'{len(self.classes)}class'
            self.task_label = f'{len(self.classes)}-class'
        self.policy = dict(POLICY, **config.get('policy', {}))
        if 'channels' not in config:
            if self.reference_checkpoint:
                contract_file = Path(str(self.reference_checkpoint) + '.json')
                if contract_file.exists():
                    c_data = json.loads(contract_file.read_text(encoding='utf-8'))
                    self.channels = list(c_data.get('channels', []))
            if not getattr(self, 'channels', None):
                from threeclass_acquisition import CHANNELS_20
                self.channels = list(CHANNELS_20)
        else:
            self.channels = list(config['channels'])
        if (len(self.channels) < 2 or any(not isinstance(c, str) or not c or c != c.strip() for c in self.channels)
                or len({c.casefold() for c in self.channels}) != len(self.channels)):
            raise ValueError('Select twenty unique actual electrode sites.')
        self.channel_indices = {c.casefold():i for i, c in enumerate(self.channels)}
        if config.get('sfreq', 256) != 256 or config.get('window_s', 4.) not in (2., 4.) or config.get('stride_s', .25) != .25:
            raise ValueError('Three-class studio requires 256 Hz, two- or four-second windows and .25-second stride.')
        self.metadata_contract = dict(channels=self.channels, sfreq=256)
        self.reports = (self.root/config.get('reports_dir', f'session_reports/{self.task}')).resolve()
        self.reports.mkdir(parents=True, exist_ok=True)
        self.lock, self.stop_event, self.cancel_fit = threading.RLock(), threading.Event(), threading.Event()
        self.stop = self.stop_event
        self.inlet = self.adapter = self.decoder = self.protocol = None
        self.training_thread = None
        self.connect_requested = False
        self.connection_generation = 0
        self.last_data = None
        self.last_ui_seen = self.clock()
        self.run_serial = 0
        self.udp_port = config.get('udp_port', 5005 if self.task == 'binary' else None)
        self.udp_host = config.get('udp_host', '127.0.0.1')
        self.udp_sock = None
        if self.udp_port:
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_target = (self.udp_host, int(self.udp_port))
        else:
            self.udp_target = None
        self.state = dict(connected=False, connecting=False, streaming=False, mode='live', status='Connect EEG to begin.', error='',
            participant=config.get('participant', 'self'), stream_name='', model_role='untrained',
            model_sha256='', model_device=self.device, model_ready=False, training=False, training_progress='', protocol_kind='',
            cue='', cue_index=0, cue_total=0, cue_remaining_s=0., raw_probabilities=[0.]*len(self.classes),
            ema_probabilities=[0.]*len(self.classes), decision='uncertain', quality='warming_up', result=None,
            last_saved='', window_s=4., stride_s=.25,
            model_scope='Personal calibration is required before fresh validation.',
            cue_run_id='', awaiting_cue_ack=False, model_choices=[], rejected_models=[],
            calibration_diagnostics=None, classes=self.classes, task_label=self.task_label,
            calibration_total=len(self.classes)*int(config.get('calibration_cues_per_class', 12)),
            validation_total=len(self.classes)*int(config.get('validation_cues_per_class', 10)),
            cue_duration_s=float(config.get('cue_duration_s', 8.)),
            stage1_diagnostics=None, stage2_diagnostics=None, hierarchical_diagnostics=None)
        self.state['quality_details'] = {}
        if re.fullmatch(r'[A-Za-z0-9_-]{1,40}', self.state['participant']) is None:
            raise ValueError('Use a short pseudonymous participant ID: letters, digits, underscore or hyphen.')
        self.results = []
        self._load_reference()
        self._clear_recording()
        self._refresh_models()

    def _send_udp(self, packet):
        if self.udp_sock is not None:
            try:
                self.udp_sock.sendto(json.dumps(packet, allow_nan=False).encode('utf-8'), self.udp_target)
            except OSError:
                pass

    def _send_rest_udp(self, reason='paused'):
        prob_dict = {c: (1.0 if 'rest' in c.lower() else 0.0) for c in self.classes}
        self._send_udp(dict(
            timestamp=time.time(),
            sample_end=0,
            class_name='rest',
            class_id=0,
            grasp_command=0,
            extension_command=0,
            continuous_flex=0.0,
            continuous_splay=0.0,
            probabilities=prob_dict,
            probabilities_raw=[1.0 if i == 0 else 0.0 for i in range(len(self.classes))],
            probabilities_ema=[1.0 if i == 0 else 0.0 for i in range(len(self.classes))],
            quality=reason,
            p_raw=0.0,
            p_smooth=0.0,
            state_label='REST',
            threshold=float(self.config.get('threshold', 0.5)),
        ))

    def close(self):
        with self.lock:
            self.cancel_fit.set()
            self.stop_event.set()
            self._send_rest_udp('shutdown')
            if self.udp_sock:
                try:
                    self.udp_sock.close()
                except OSError:
                    pass
                self.udp_sock = None

    def _refresh_models(self):
        if len(self.channels) != 20:
            self.state.update(model_choices=[], rejected_models=[])
            return dict(models=[], rejected=[])
        found = list_personal_models(self.reports, self.state['participant'], self.channels, self.classes)
        self.state.update(model_choices=found['models'], rejected_models=found['rejected'])
        return found

    def _sync_hierarchical_state(self, quality='warming_up'):
        if self.decoder and getattr(self.decoder, 'is_hierarchical', False):
            from multiclass_decoder import build_hierarchical_diagnostics
            s1, s2, hier = build_hierarchical_diagnostics(
                self.decoder.arch,
                self.decoder.alignment is not None,
                0.0, 0.0, 0.0, 0.0,
                gate_threshold=getattr(self.decoder, 'gate_threshold', 0.50),
                mrcp_band=self.decoder.dual_filter.mrcp_band if self.decoder.dual_filter else (0.3, 3.0),
                smr_band=self.decoder.dual_filter.smr_band if self.decoder.dual_filter else (4.0, 40.0),
                quality=quality,
            )
            self.state.update(
                stage1_diagnostics=s1,
                stage2_diagnostics=s2,
                hierarchical_diagnostics=hier,
            )
        else:
            self.state.update(
                stage1_diagnostics=None,
                stage2_diagnostics=None,
                hierarchical_diagnostics=None,
            )

    @staticmethod
    def _calibration_summary(diagnostics):
        train, tune = diagnostics.get('training', {}), diagnostics.get('tuning', {})
        deployed = tune.get('deployment_policy', {})
        return dict(scope='Calibration tuning only; collect a fresh frozen validation block.',
            train_raw_cue_ba=train.get('cue_trials', {}).get('balanced_accuracy'),
            tuning_raw_cue_ba=tune.get('cue_trials', {}).get('balanced_accuracy'),
            tuning_deployed_cue_ba=deployed.get('balanced_accuracy'),
            tuning_deployed_coverage=deployed.get('coverage'),
            tuning_deployed_recall=deployed.get('recall', {}),
            clean_cues=diagnostics.get('counts', {}).get('clean_cues'),
            complete_cues=diagnostics.get('counts', {}).get('complete_cues'),
            training_cues=len(diagnostics.get('train_cues', [])), tuning_cues=len(diagnostics.get('tuning_cues', [])),
            split_scope=diagnostics.get('split_scope', 'Saved calibration split.'))

    def _load_reference(self):
        self.decoder = None
        self.checkpoint = self.reference_checkpoint
        self.state.update(model_ready=False, model_role='untrained', model_sha256='', calibration_diagnostics=None,
                          model_scope='Personal calibration is required before fresh validation.')
        if self.reference_checkpoint is None:
            self._sync_hierarchical_state('warming_up')
            return
        reference = MulticlassDecoder(self.reference_checkpoint, device=self.device)
        ref_classes = [c if c != 'action' else 'right_hand_grasp' for c in reference.classes]
        stu_classes = [c if c != 'action' else 'right_hand_grasp' for c in self.classes]
        if ref_classes != stu_classes:
            raise ValueError('Reference model class order differs from the selected profile.')
        if reference.contract.get('role') == 'personal_training_template':
            self.decoder = None
            self.state.update(model_role='untrained', model_sha256=reference.contract['sha256'],
                model_scope='Random architecture template. Run personal calibration before validation.')
        elif [c.casefold() for c in reference.channels] == [c.casefold() for c in self.channels]:
            self.decoder = reference
            if getattr(reference, 'is_hierarchical', False):
                reference_name = f'hierarchical dual-band ({reference.arch})'
            elif reference.arch in ('dual_band_cnn', 'DualBandTemporalEEGNet'):
                reference_name = 'dual-band CNN'
            elif reference.arch == 'temporal_cnn':
                reference_name = 'temporal CNN'
            else:
                reference_name = 'reference model'
            self.state.update(model_role='recorded_reference', model_sha256=reference.contract['sha256'],
                model_scope=f'Recorded participant 08 {reference_name}. Fit your own calibration before validation.')
        else:
            self.state['model_scope'] = 'Selected physical montage needs personal fitting; reference predictions unavailable.'
        self._sync_hierarchical_state('warming_up')

    def _clear_recording(self):
        self.raw, self.times, self.segment_ids, self.events, self.predictions = [], [], [], [], []
        self.segment = 0
        self.session_id = uuid.uuid4().hex
        self.session_started = datetime.now(timezone.utc).isoformat()
        self.waves = {c:deque(maxlen=1024) for c in ['time', 'C3', 'Cz', 'C4']}
        self.history = {c:deque(maxlen=240) for c in ['time']+self.classes+
                        ['raw_'+c for c in self.classes]+['ema_'+c for c in self.classes]}
        self._reset_segment()

    def _reset_segment(self):
        self.segment += 1
        n_ch = len(self.channels)
        self.plot_sos = butter(4, [8, 30], fs=256, btype='bandpass', output='sos')
        self.plot_zi = np.zeros((len(self.plot_sos), n_ch, 2))
        window_samples = self.decoder.window_samples if self.decoder else round(self.config.get('window_s', 4.)*256)
        self.quality_buffer = RingBuffer(n_ch, window_samples)
        if self.decoder:
            self.decoder.reset()
        if self.adapter:
            self.adapter.reset_timestamps()
        self.state.update(quality='warming_up', decision='uncertain', raw_probabilities=[0.]*len(self.classes),
                          ema_probabilities=[0.]*len(self.classes), quality_details={}, window_s=window_samples/256,
                          prediction_timing=None, prediction_display_ack_ms=None)
        self._sync_hierarchical_state('warming_up')

    def snapshot(self, seen=False):
        with self.lock:
            if seen:
                self.last_ui_seen = self.clock()
            return dict(self.state, waves={k:list(v) for k, v in self.waves.items()},
                        probability_history={k:list(v) for k, v in self.history.items()})

    def _abort(self, reason):
        self._send_rest_udp(reason)
        if self.protocol:
            now = self.clock()
            self.protocol.abort(now)
            self.events.extend(self.protocol.events)
            self.results.append(dict(kind=self.protocol.kind, status='aborted', reason=reason,
                                     run_id=self.protocol.run_id))
            self.protocol = None
            self.state['result'] = self.results[-1]
        self.state.update(streaming=False, protocol_kind='', cue='', cue_remaining_s=0., status=reason,
                          cue_run_id='', awaiting_cue_ack=False)
        self._reset_segment()

    def save(self):
        self._send_rest_udp('saved')
        folder = self.reports/self.state['participant']/self.session_id
        folder.mkdir(parents=True, exist_ok=True)
        raw = np.concatenate(self.raw, axis=1) if self.raw else np.empty((20, 0), np.float32)
        times = np.concatenate(self.times) if self.times else np.empty(0)
        segments = np.concatenate(self.segment_ids) if self.segment_ids else np.empty(0, dtype=int)
        np.savez_compressed(folder/'eeg.npz', raw=raw, timestamps=times, segments=segments,
                            channels=np.array(self.channels), sfreq=256., unit='V')
        metadata = dict(schema_version=1, task_mode=f'{self.task}_lab', source=self.state['mode'],
            participant=self.state['participant'], channels=self.channels, sfreq=256, unit='V',
            session_started=self.session_started, model_path=str(self.checkpoint) if self.checkpoint else None,
            model_sha256=self.state['model_sha256'], model_role=self.state['model_role'],
            reference_sha256=digest(self.reference_checkpoint) if self.reference_checkpoint else None, config=self.config,
            timing='Prediction timestamps are EEG sample times. Decoder start/completion and sampled browser paint acknowledgement receipt use the same server clock. Acknowledgements include browser scheduling and round-trip overhead.',
            classes=self.classes, results=self.results, protocol_in_progress=self.protocol is not None,
            active_frozen=getattr(self.protocol, 'frozen', None))
        active_events = self.protocol.events if self.protocol else []
        for name, data in [('metadata.json', metadata), ('events.json', self.events+active_events),
                           ('predictions.json', self.predictions)]:
            (folder/name).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')
        self.state['last_saved'] = str(folder)
        return str(folder)

    def action(self, path, body=None):
        body = body or {}
        with self.lock:
            if path == '/api/toggle_stream':
                if self.state['streaming']:
                    self.action('/api/stop', body)
                else:
                    self.action('/api/start', body)
                return self.snapshot()
            elif path == '/api/set_threshold':
                val = float(body.get('val', [body.get('threshold', 0.5)])[0] if isinstance(body.get('val'), list) else body.get('val', body.get('threshold', 0.5)))
                self.config['threshold'] = val
                return self.snapshot()
            elif path == '/api/export_figure':
                folder = self.save()
                return dict(status='saved', folder=folder)
            if path in ('/api/start', '/api/calibrate', '/api/validate') and body.get('participant', self.state['participant']) != self.state['participant']:
                raise ValueError('Reconnect EEG with the new participant ID before starting or validating a recording.')
            if path == '/api/stop':
                self.connection_generation += 1
                self.connect_requested = False
                self.state['connecting'] = False
                self.cancel_fit.set()
                self._abort(body.get('reason', 'Stopped by user.'))
                self.save()
            elif path == '/api/save':
                if self.state['training']:
                    raise ValueError('Calibration is already saved. Wait for fitting or stop it first.')
                self._abort('Saved and stopped.')
                self.save()
            elif self.state['training']:
                raise ValueError('Wait for personal fitting or stop it before changing the session.')
            elif path == '/api/cue-presented':
                if not self.protocol or not self.state['streaming'] or self.state['mode'] != 'live':
                    raise ValueError('No live cue is awaiting display acknowledgement.')
                self.protocol.acknowledge(body.get('run_id'), body.get('trial'), self.clock())
                self._refresh(self.clock())
            elif path == '/api/prediction-painted':
                index = body.get('index')
                if (body.get('session_id') == self.session_id and type(index) is int
                        and 0 <= index < len(self.predictions)):
                    record = self.predictions[index]
                    if 'paint_ack_at' not in record:
                        record['paint_ack_at'] = self.clock()
                        self.state['prediction_display_ack_ms'] = 1000*(record['paint_ack_at']-record['timestamp'])
                return {}
            elif path in ('/api/models', '/api/load-model'):
                if self.protocol or self.state['streaming']:
                    raise ValueError('Stop and save the current recording before selecting a saved personal model.')
                if body.get('participant', self.state['participant']) != self.state['participant']:
                    raise ValueError('Apply the participant ID before selecting their saved model.')
                self._refresh_models()
                if path == '/api/load-model':
                    candidate = validate_personal_model(body.get('model_id'), self.state['participant'], self.channels, self.reports, self.classes)
                    decoder = MulticlassDecoder(candidate['checkpoint'], device=self.device)
                    if [c.casefold() for c in decoder.channels] != [c.casefold() for c in self.channels] or decoder.classes != self.classes:
                        raise ValueError('Saved model channels/classes do not match the selected profile.')
                    if self.task != 'threeclass' and decoder.policy.config != self.policy:
                        raise ValueError('Saved model policy differs from the fixed profile policy.')
                    self.save()
                    self.decoder, self.checkpoint = decoder, Path(candidate['checkpoint'])
                    self._clear_recording()
                    diagnostics_path = self.checkpoint.parent/'training.json'
                    summary = self._calibration_summary(json.loads(diagnostics_path.read_text())) if diagnostics_path.exists() else None
                    self.state.update(model_ready=True, model_role='personal', model_sha256=candidate['sha256'],
                        result=None, calibration_diagnostics=summary,
                        model_scope='Saved personal model loaded. Validate on new cues; cap/reference changes may require recalibration.',
                        status='Personal model loaded. Connect EEG and collect a fresh validation block.', error='')
            elif path == '/api/connect':
                if self.protocol or self.state['streaming']:
                    raise ValueError('Stop and save the active recording before reconnecting.')
                participant = body.get('participant', self.state['participant'])
                if re.fullmatch(r'[A-Za-z0-9_-]{1,40}', participant) is None:
                    raise ValueError('Use a short pseudonymous ID: letters, digits, underscore or hyphen.')
                if participant != self.state['participant']:
                    self.save()
                    self.state['participant'] = participant
                    self.results = []
                    self.state['result'] = None
                    self._load_reference()
                    self._clear_recording()
                    self._refresh_models()
                if self.inlet:
                    self.inlet.close_stream()
                self.inlet = self.adapter = None
                self.connection_generation += 1
                self.state.update(mode='live', connected=False, connecting=True, error='', status='Connecting to EEG…')
                self.connect_requested = True
            elif path == '/api/demo':
                if self.protocol or self.state['streaming']:
                    raise ValueError('Stop and save before changing data source.')
                self.save()
                if self.inlet:
                    self.inlet.close_stream()
                self.inlet = self.adapter = None
                self.connection_generation += 1
                self.connect_requested = False
                mode = 'live' if self.state['mode'] == 'demo' else 'demo'
                self.results = []
                self.state.update(mode=mode, connected=mode == 'demo', connecting=False, result=None,
                                  status='Synthetic demo.' if mode == 'demo' else 'Connect EEG to begin.')
                self._clear_recording()
            elif path == '/api/reset':
                if self.protocol or self.state['streaming']:
                    raise ValueError('Stop and save before resetting.')
                self.save()
                self._clear_recording()
                self.state.update(result=None, status='New recording; personal model retained.')
            elif path == '/api/start':
                self._start_stream()
            elif path in ('/api/calibrate', '/api/validate'):
                if self.state['mode'] != 'live' or not self.state['connected']:
                    raise ValueError('A fresh live EEG connection is required; demo cannot calibrate or validate.')
                if self.protocol:
                    raise ValueError('A cue run is already active.')
                kind = 'calibration' if path.endswith('calibrate') else 'validation'
                if kind == 'validation' and not self.state['model_ready']:
                    raise ValueError(f'Fit your personal {self.task_label} calibration before validation.')
                if kind == 'validation' and digest(self.checkpoint) != self.state['model_sha256']:
                    raise ValueError('Personal model file changed. Reload or fit a new calibration before validation.')
                if self.raw:
                    self._abort('Starting a separate cue recording.')
                    self.save()
                self._clear_recording()
                self.state.update(result=None, error='')
                if kind == 'calibration':
                    self.state['model_ready'] = False
                    self._load_reference()
                self._start_stream()
                count = int(self.config.get('calibration_cues_per_class', 12) if kind == 'calibration'
                            else self.config.get('validation_cues_per_class', 10))
                self.protocol = ThreeClassProtocol(kind, self.clock(), count,
                    float(self.config.get('cue_duration_s', 8.)), int(self.config.get('seed', 42))+self.run_serial,
                    require_ack=True, classes=self.classes, window_s=self.state['window_s'], settle_s=self.config.get('settle_s', .5))
                self.run_serial += 1
                self.protocol.frozen = dict(model_sha256=self.state['model_sha256'],
                    contract_sha256=digest(str(self.checkpoint)+'.json') if self.checkpoint else '',
                    policy=dict(self.decoder.policy.config) if self.decoder else dict(self.policy),
                    window_s=self.protocol.window_s, settle_s=self.protocol.settle_s)
                self.last_ui_seen = self.clock()
                self.state.update(protocol_kind=kind, status=f'{kind.title()} in progress.')
                self._refresh(self.clock())
            else:
                raise ValueError('Unknown operation.')
            return self.snapshot()

    def _start_stream(self):
        if not self.state['connected']:
            raise ValueError('Connect live EEG, or choose synthetic demo to preview the interface.')
        if not self.state['streaming']:
            if self.inlet:
                self.inlet.flush()
            self._reset_segment()
            self.state.update(streaming=True,
                              status='Synthetic preview running.' if self.state['mode'] == 'demo' else 'Recording EEG.',
                              error='')

    def _refresh(self, now):
        p = self.protocol
        if not p:
            return
        if now-self.last_ui_seen > 3.:
            self._abort('Cue display connection timed out; run aborted.')
            self.save()
            return
        try:
            p.advance(now)
        except ValueError as exc:
            self._abort(str(exc)+' Run aborted.')
            self.state['error'] = str(exc)
            self.save()
            return
        if not p.done:
            self.state.update(cue=self.classes[p.labels[p.index]], cue_index=p.index+1,
                cue_total=len(p.labels), cue_run_id=p.run_id, awaiting_cue_ack=p.awaiting_ack,
                cue_remaining_s=p.duration if p.awaiting_ack else max(0., p.duration-now+p.events[-1]['onset']))
            return
        self.events.extend(p.events)
        self.protocol = None
        self.state.update(streaming=False, protocol_kind='', cue='', cue_remaining_s=0.,
                          cue_run_id='', awaiting_cue_ack=False)
        if p.kind == 'validation':
            try:
                if digest(self.checkpoint) != p.frozen['model_sha256'] or digest(str(self.checkpoint)+'.json') != p.frozen['contract_sha256']:
                    raise ValueError('The frozen validation model/contract changed; this run cannot be scored.')
                if self.decoder.policy.config != p.frozen['policy']:
                    raise ValueError('The frozen validation decision policy changed; this run cannot be scored.')
                result = score_validation(p)
                result.update(run_id=p.run_id, frozen=p.frozen, participant=self.state['participant'], source=self.state['mode'])
                self.results.append(result)
                self.state.update(result=result, status='Validation complete. Save the result and record another fresh block.')
            except ValueError as exc:
                self.state.update(error=str(exc), status='Validation failed.')
                self.results.append(dict(kind='validation', status='failed', error=str(exc)))
            self.save()
        else:
            self.results.append(dict(kind='calibration_collection', status='complete', n_cues=len(p.events), run_id=p.run_id))
            self.save()
            self._fit(p)

    def _fit(self, protocol):
        from threeclass_personal import fit_personal
        raw = np.concatenate(self.raw, axis=1).copy()
        times = np.concatenate(self.times).copy()
        segments = np.concatenate(self.segment_ids).copy()
        output = self.reports/self.state['participant']/('model_'+protocol.run_id)
        self.cancel_fit.clear()
        self.state.update(training=True, model_ready=False, training_progress='Preparing clean calibration cues.', status=f'Fitting personal {self.task_label} model…')
        participant = self.state['participant']
        def progress(message):
            with self.lock:
                self.state['training_progress'] = str(message)
        def task():
            try:
                result = fit_personal(raw, times, protocol.events, self.channels, output,
                    config=dict(self.config, participant=participant, calibration_run_id=protocol.run_id),
                    cancel=self.cancel_fit, progress=progress, segments=segments, device=self.device)
                decoder = MulticlassDecoder(result['checkpoint'], policy=self.policy, device=self.device)
                if decoder.channels != self.channels or decoder.classes != self.classes:
                    raise ValueError('Personal calibration produced an incompatible montage or class order.')
                with self.lock:
                    if self.cancel_fit.is_set() or self.stop_event.is_set():
                        raise ValueError('Personal fitting cancelled.')
                    self.decoder, self.checkpoint = decoder, Path(result['checkpoint'])
                    self.state.update(model_ready=True, model_role='personal', model_sha256=decoder.contract['sha256'],
                        window_s=decoder.window_samples/decoder.sfreq,
                        model_scope='Personal model. Validate on NEW cues with weights/settings frozen.',
                        training_progress='Personal model ready.', status='Personal model ready. Start a fresh validation block.', error='',
                        calibration_diagnostics=self._calibration_summary(result['diagnostics']))
                    self._refresh_models()
                    self.results.append(dict(kind='personal_calibration_fit', diagnostics=result['diagnostics'], checkpoint=str(self.checkpoint)))
                    self.save()
            except Exception as exc:
                with self.lock:
                    self.state.update(model_ready=False, error=str(exc), status='Personal fitting failed or was stopped.')
                    self.results.append(dict(kind='personal_calibration_fit', status='failed', error=str(exc)))
                    self.save()
            finally:
                with self.lock:
                    self.state['training'] = False
        self.training_thread = threading.Thread(target=task, daemon=True)
        self.training_thread.start()

    def ingest(self, raw, timestamps):
        if not self.state['streaming']:
            return
        raw = np.asarray(raw, np.float32)
        ts = np.asarray(timestamps, dtype=float)
        if raw.shape != (20, len(ts)) or not len(ts) or not np.isfinite(raw).all() or not np.isfinite(ts).all():
            raise ValueError('Invalid EEG/timestamp geometry.')
        if np.any(np.diff(ts) <= 0) or (self.times and ts[0] <= self.times[-1][-1]):
            raise ValueError('EEG timestamps moved backwards.')
        if np.any(np.diff(ts) > float(self.config.get('max_gap_s', .1))):
            raise ValueError('EEG timestamps contain a recording gap.')
        if self.times and self.segment_ids[-1][-1] == self.segment and ts[0]-self.times[-1][-1] > float(self.config.get('max_gap_s', .1)):
            raise ValueError('EEG timestamps contain a recording gap.')
        self.raw.append(raw.copy())
        self.times.append(ts.copy())
        self.segment_ids.append(np.full(len(ts), self.segment, np.int32))
        filtered, self.plot_zi = sosfilt(self.plot_sos, raw, axis=1, zi=self.plot_zi)
        self.quality_buffer.push_chunk(filtered)
        if self.quality_buffer.is_full:
            current = self.quality_buffer.get()
            peaks = np.max(np.abs(current), axis=1)
            std = np.std(current, axis=1)
            artifact_limit = self.decoder.contract['artifact_limit_v'] if self.decoder else float(self.config.get('artifact_limit_v', 100e-6))
            flat_limit = self.decoder.contract['min_signal_std_v'] if self.decoder else float(self.config.get('min_signal_std_v', 1e-9))
            self.state['quality_details'] = dict(
                artifact_channels=[ch for ch, peak in zip(self.channels, peaks) if peak > artifact_limit],
                flat_channels=[ch for ch, value in zip(self.channels, std) if value < flat_limit],
                peak_uV={ch:float(value*1e6) for ch, value in zip(self.channels, peaks)},
                scope='Current filtered EEG window across all selected sites.')
        self.waves['time'].extend(ts.tolist())
        for c in ['C3', 'Cz', 'C4']:
            self.waves[c].extend((filtered[self.channel_indices[c.casefold()]]*1e6).tolist()
                                 if c.casefold() in self.channel_indices else [None]*len(ts))
        if not self.decoder:
            details = self.state['quality_details']
            quality = ('artifact' if details.get('artifact_channels') else
                       'flat_signal' if details.get('flat_channels') else
                       'recording_untrained' if self.quality_buffer.is_full else 'warming_up')
            self.state.update(quality=quality, decision='uncertain')
            return
        before = self.decoder.samples_seen
        started_at = self.clock()
        packets = self.decoder.process_chunk_all(raw)
        ready_at = self.clock() if packets else None
        for packet in packets:
            offset = packet['sample_end']-before-1
            sample_time = float(ts[offset])
            protocol = self.protocol
            cue_trial = protocol.index+1 if protocol and protocol.add(packet, sample_time) else None
            record = dict(packet, timestamp=sample_time, segment=self.segment, decoder_started_at=started_at,
                prediction_ready_at=ready_at, model_sha256=self.state['model_sha256'], mode=self.state['mode'],
                cue_run_id=protocol.run_id if cue_trial is not None else None, cue_trial=cue_trial)
            self.state['prediction_timing'] = dict(session_id=self.session_id, index=len(self.predictions),
                sample_to_ready_ms=1000*(ready_at-sample_time), decoder_chunk_ms=1000*(ready_at-started_at),
                cue_run_id=record['cue_run_id'], cue_trial=cue_trial)
            self.predictions.append(record)
            p, ema = packet['probabilities_raw'], packet['probabilities_ema']
            self.state.update(raw_probabilities=p, ema_probabilities=ema, decision=packet['class_name'], quality=packet['quality'])
            if 'stage1_diagnostics' in packet:
                self.state['stage1_diagnostics'] = packet['stage1_diagnostics']
                self.state['stage2_diagnostics'] = packet['stage2_diagnostics']
                self.state['hierarchical_diagnostics'] = packet['hierarchical_diagnostics']
            self.history['time'].append(sample_time)
            for i, c in enumerate(self.classes):
                self.history[c].append(p[i])
                self.history['raw_'+c].append(p[i])
                self.history['ema_'+c].append(ema[i])
            if self.udp_sock:
                grasp_idx = next((i for i, c in enumerate(self.classes) if 'grasp' in c.lower()), 1 if len(self.classes) > 1 else None)
                ext_idx = next((i for i, c in enumerate(self.classes) if 'ext' in c.lower() or 'open' in c.lower()), 2 if len(self.classes) > 2 else None)
                rest_idx = next((i for i, c in enumerate(self.classes) if 'rest' in c.lower()), 0)

                raw_id = packet.get('class_id')
                active_id = int(raw_id) if raw_id is not None else -1
                quality = packet.get('quality', 'ok')
                class_name = packet.get('class_name', 'rest')

                threshold = float(self.config.get('threshold', self.policy.get('min_confidence', 0.5)))

                grasp_command = 1 if (quality == 'ok' and active_id >= 0 and active_id == grasp_idx) else 0
                extension_command = 1 if (quality == 'ok' and active_id >= 0 and active_id == ext_idx) else 0

                p_grasp_ema = float(ema[grasp_idx]) if (grasp_idx is not None and grasp_idx < len(ema)) else 0.0
                p_ext_ema = float(ema[ext_idx]) if (ext_idx is not None and ext_idx < len(ema)) else 0.0

                continuous_flex = float(np.clip((p_grasp_ema - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)) if grasp_command else 0.0
                continuous_splay = float(np.clip((p_ext_ema - threshold) / max(1e-6, 1.0 - threshold), 0.0, 1.0)) if extension_command else 0.0

                prob_dict = {c: float(p[i]) for i, c in enumerate(self.classes)}
                p_raw_val = float(p[grasp_idx]) if (grasp_idx is not None and grasp_idx < len(p)) else float(p[1] if len(p) > 1 else p[0])
                p_smooth_val = float(ema[grasp_idx]) if (grasp_idx is not None and grasp_idx < len(ema)) else float(ema[1] if len(ema) > 1 else ema[0])

                state_label = 'REST'
                if grasp_command:
                    state_label = 'GRASP'
                elif extension_command:
                    state_label = 'EXTENSION'
                elif class_name not in ('uncertain', 'rest'):
                    state_label = 'ACTION'

                udp_packet = dict(
                    timestamp=sample_time,
                    sample_end=packet['sample_end'],
                    class_name=class_name,
                    class_id=active_id,
                    grasp_command=grasp_command,
                    extension_command=extension_command,
                    continuous_flex=continuous_flex,
                    continuous_splay=continuous_splay,
                    probabilities=prob_dict,
                    probabilities_raw=[float(x) for x in p],
                    probabilities_ema=[float(x) for x in ema],
                    quality=quality,
                    p_raw=p_raw_val,
                    p_smooth=p_smooth_val,
                    state_label=state_label,
                    threshold=threshold,
                )
                self._send_udp(udp_packet)

    def worker(self):
        rng = np.random.default_rng(42)
        next_demo = 0.
        while not self.stop_event.is_set():
            with self.lock:
                now = self.clock()
                try:
                    self._refresh(now)
                except Exception as exc:
                    self.cancel_fit.set()
                    self._abort('Cue run stopped because recording or fitting preparation failed.')
                    self.state['error'] = str(exc)
                    self.results.append(dict(status='failed', error=str(exc)))
                    try:
                        self.save()
                    except OSError as save_exc:
                        self.state['error'] += f' Could not save: {save_exc}'
                should_connect = self.connect_requested
                connection_generation = self.connection_generation
                self.connect_requested = False
                mode = self.state['mode']
            if should_connect:
                try:
                    inlet, adapter = connect(self.config, self.metadata_contract)
                    with self.lock:
                        if connection_generation != self.connection_generation or self.state['mode'] != 'live' or self.stop_event.is_set():
                            inlet.close_stream()
                            continue
                        self.inlet, self.adapter = inlet, adapter
                        self.last_data = self.clock()
                        self.state.update(connecting=False, stream_name=adapter.metadata['name'], status='Waiting for fresh EEG samples.', error='')
                except Exception as exc:
                    with self.lock:
                        if connection_generation == self.connection_generation:
                            self.state.update(connected=False, connecting=False, error=str(exc), status='Connection failed; check preflight.')
            if mode == 'demo':
                with self.lock:
                    if self.state['streaming'] and now >= next_demo:
                        start = (self.times[-1][-1]+1/256 if self.times and
                                 self.segment_ids[-1][-1] == self.segment else now-63/256)
                        ts = start+np.arange(64)/256
                        self.ingest(rng.normal(0, 5e-6, (20, 64)).astype(np.float32), ts)
                        next_demo = now+.25
                self.stop_event.wait(.01)
                continue
            with self.lock:
                inlet, adapter = self.inlet, self.adapter
            if inlet is not None:
                try:
                    chunk, ts = inlet.pull_chunk(timeout=.03, max_samples=64)
                    now = self.clock()
                    with self.lock:
                        if inlet is not self.inlet:
                            continue
                        if len(chunk):
                            raw, stamps = adapter.transform(chunk, ts, now)
                            self.last_data = now
                            self.state['connected'] = True
                            self.ingest(raw, stamps)
                        elif self.last_data is not None and now-self.last_data > 1.:
                            raise ValueError('EEG stream timed out. Reconnect before starting a new cue run.')
                except Exception as exc:
                    with self.lock:
                        self.state.update(connected=False, error=str(exc))
                        self._abort('EEG connection lost; run aborted.')
                        self.save()
                        if self.inlet:
                            self.inlet.close_stream()
                        self.inlet = self.adapter = None
            self.stop_event.wait(.01)
        with self.lock:
            self._abort('Studio closed.')
            self.save()
            if self.inlet:
                self.inlet.close_stream()


class Handler(BaseHTTPRequestHandler):
    studio = None
    def log_message(self, *args): pass
    def respond(self, code, value, content_type=None, html=False):
        if html:
            content_type = 'text/html'
        elif content_type is None:
            content_type = 'application/json'

        if isinstance(value, bytes):
            payload = value
        elif content_type == 'application/json':
            payload = json.dumps(value, allow_nan=False).encode('utf-8')
        else:
            payload = str(value).encode('utf-8')

        self.send_response(code)
        if content_type.startswith('text/') or content_type in ('application/javascript', 'application/json'):
            if 'charset' not in content_type:
                content_type = f'{content_type}; charset=utf-8'
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(payload)))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Origin, Accept')
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Origin, Accept')
        self.send_header('Content-Length', '0')
        self.end_headers()

    def do_GET(self):
        url = urlparse(self.path)
        path = url.path
        if path in ('/', '/index.html'):
            html_file = self.studio.root / 'threeclass_studio.html'
            if not html_file.exists():
                html_file = self.studio.root / 'studio.html'
            self.respond(200, html_file.read_text(encoding='utf-8'), html=True)
        elif path in ('/virtual_hand', '/virtual_hand.html', '/hand'):
            hand_file = self.studio.root / 'virtual_hand.html'
            if hand_file.exists():
                self.respond(200, hand_file.read_text(encoding='utf-8'), html=True)
            else:
                self.respond(404, dict(error='virtual_hand.html not found.'))
        elif path.startswith('/static/'):
            static_dir = (self.studio.root / 'static').resolve()
            safe_rel = path[len('/static/'):]
            file_path = (static_dir / safe_rel).resolve()
            try:
                if not file_path.is_relative_to(static_dir):
                    self.respond(403, dict(error='Forbidden.'))
                    return
            except (ValueError, AttributeError):
                self.respond(403, dict(error='Forbidden.'))
                return
            if file_path.is_file():
                mime = 'application/octet-stream'
                if file_path.suffix == '.js':
                    mime = 'application/javascript'
                elif file_path.suffix == '.css':
                    mime = 'text/css'
                elif file_path.suffix in ('.html', '.htm'):
                    mime = 'text/html'
                elif file_path.suffix == '.json':
                    mime = 'application/json'
                else:
                    guessed = mimetypes.guess_type(str(file_path))[0]
                    if guessed:
                        mime = guessed
                self.respond(200, file_path.read_bytes(), content_type=mime)
            else:
                self.respond(404, dict(error='Static asset not found.'))
        elif path == '/api/state':
            self.respond(200, self.studio.snapshot(seen=True))
        else:
            self.respond(404, dict(error='Not found.'))
    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
            self.respond(403, dict(error='Local console origin required.'))
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if length > 10000:
                raise ValueError('Request too large.')
            body = json.loads(self.rfile.read(length) or b'{}') if length > 0 else {}
            if not isinstance(body, dict):
                raise ValueError('Expected a JSON object.')
            from urllib.parse import urlparse, parse_qs
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if query:
                for k, v in query.items():
                    if k not in body:
                        body[k] = v[0] if len(v) == 1 else v
            self.respond(200, self.studio.action(url.path, body))
        except (ValueError, RuntimeError) as exc:
            self.respond(409, dict(error=str(exc)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'threeclass_config.json')
    parser.add_argument('--port', type=int, default=8051)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--open-virtual-hand', action='store_true', help='Open 3D virtual hand window on launch.')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()
    import torch
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    torch.set_num_threads(int(config.get('torch_threads', 1)))
    studio = ThreeClassStudio(config)
    if args.demo:
        studio.action('/api/demo')
    handler = type('ThreeClassHandler', (Handler,), dict(studio=studio))
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    worker = threading.Thread(target=studio.worker, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{args.port}'
    print(f'{studio.task_label.capitalize()} lab: {url}\nPrivate reports: {studio.reports}', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
        if args.open_virtual_hand:
            webbrowser.open(f'{url}/virtual_hand')
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        studio.cancel_fit.set()
        studio.stop_event.set()
        worker.join(5.)
        if studio.training_thread:
            studio.training_thread.join(15.)
        server.server_close()


if __name__ == '__main__':
    main()
