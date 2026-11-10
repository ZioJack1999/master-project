"""Unified Binary BCI Studio: validated LSL acquisition, modern calibration, and prospective cues.

Unified binary studio GUI implementation based on threeclass_studio:
- Configured for 2 classes: Rest and Right-Hand Grasp.
- Runs on port 8052 by default (Launch_BINARY_X24.bat).
- Retains UDP Port 5005 broadcasting for external virtual hand actuation:
  {"grasp_command": 0/1, "continuous_flex": 0.0-1.0, "p_raw": float, "p_smooth": float, ...}
- Employs the unified modern visual layout, cue card, probability bars, model loader, and session saving.
"""
import argparse
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse
import webbrowser
import numpy as np
import torch

from realtime_bci_decoder import RealtimeBCIDecoder, RingBuffer, load_contract
from lsl_acquisition import connect
from protocols import CueProtocol
from model_utils import digest
from threeclass_cues import ThreeClassProtocol
from threeclass_model_store import list_personal_models, validate_personal_model

ROOT = Path(__file__).resolve().parent
BINARY_CLASSES = ['rest', 'right_hand_grasp']


def clock_now():
    from pylsl import local_clock
    return local_clock()


class Studio:
    def __init__(self, config, root=ROOT, clock=clock_now):
        self.root = Path(root)
        self.config = dict(config)
        self.clock = clock
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.stop_event = self.stop
        self.cancel_fit = threading.Event()
        self.training_cancel = self.cancel_fit

        self.classes = list(config.get('classes', BINARY_CLASSES))
        self.task = 'binary'
        self.task_label = 'binary'

        checkpoint_val = config.get('checkpoint')
        self.checkpoint = (self.root / checkpoint_val).resolve() if checkpoint_val else None
        self.base_checkpoint = self.checkpoint
        self.reference_checkpoint = self.checkpoint
        self.contract = load_contract(self.checkpoint, config.get('contract')) if self.checkpoint else {}

        self.reports = (self.root / config.get('reports_dir', 'session_reports')).resolve()
        self.reports.mkdir(parents=True, exist_ok=True)

        self.udp_host = config.get('udp_host', '127.0.0.1')
        self.udp_port = int(config.get('udp_port', 5005))
        self.udp_target = (self.udp_host, self.udp_port)
        self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if self.udp_port else None
        self.packets_sent = 0

        self.inlet = self.adapter = self.decoder = self.protocol = None
        self.device = config.get('device', 'cpu')

        default_threshold = float(config.get('threshold', self.contract.get('threshold', 0.5)))
        self.state = dict(
            # Modern UI state
            connected=False, connecting=False, streaming=False, mode='live',
            status='Connect EEG to begin.', error='',
            participant=config.get('participant_id', config.get('participant', 'self')),
            stream_name='', model_role=self.contract.get('role', 'untrained'),
            model_sha256=self.contract.get('sha256', ''), model_device=self.device,
            model_ready=bool(self.contract.get('verified') and self.contract.get('role') != 'personal_training_template'),
            training=False, training_progress='', protocol_kind='',
            cue='', cue_index=0, cue_total=0, cue_remaining_s=0.,
            raw_probabilities=[0.0, 0.0], ema_probabilities=[0.0, 0.0],
            decision='uncertain', quality='warming_up', result=None, last_saved='',
            window_s=float(self.contract.get('n_times', 512)) / float(self.contract.get('sfreq', 256)),
            stride_s=0.25,
            model_scope='Personal calibration is required before fresh validation.' if self.contract.get('role') == 'personal_training_template' else 'Binary model loaded.',
            cue_run_id='', awaiting_cue_ack=False, model_choices=[], rejected_models=[],
            calibration_diagnostics=None, classes=self.classes, task_label=self.task_label,
            calibration_total=2 * int(config.get('personal_calibration_pairs', 9)),
            validation_total=2 * int(config.get('validation_pairs', 10)),
            cue_duration_s=float(config.get('trial_duration_s', 8.0)),
            quality_details={},

            # Legacy compatibility state
            is_connected=False, is_streaming=False, demo_mode=False, is_calibrating=False,
            calib_phase='idle', calib_timer=0., calib_progress=0., calib_trial=0, total_trials=0,
            calib_acc=None, calib_th=0.5, threshold=default_threshold,
            p_raw=0., p_smooth=0., grasp_command=0, continuous_flex=0., latency_ms=0., step_count=0,
            udp_target=f"{self.udp_host}:{self.udp_port}", packets_sent=0,
            model_verified=bool(self.contract.get('verified', False)),
            validation_result=None, calibration_result=None, source='live', sample_age_s=None,
            is_training=False, training_status='', personalized=False,
            participant_id=config.get('participant_id', config.get('participant', 'self')),
            model_path=str(self.checkpoint) if self.checkpoint else '',
            personalization_enabled=config.get('personalize_on_calibration', True),
            calibration_cues=2 * int(config.get('personal_calibration_pairs', 9)),
            trial_duration_s=float(config.get('trial_duration_s', 8.0)),
        )

        self.hist = {key: deque([0.] * 60, maxlen=60) for key in ['eeg_c3', 'eeg_cz', 'eeg_c4', 'prob_history', 'ema_history']}
        self.scope = {key: deque([0.] * 512, maxlen=512) for key in ['wave_c3', 'wave_cz', 'wave_c4']}
        self.waves = {c: deque(maxlen=1024) for c in ['time', 'C3', 'Cz', 'C4']}
        self.history = {c: deque(maxlen=240) for c in ['time'] + self.classes + ['raw_' + c for c in self.classes] + ['ema_' + c for c in self.classes]}

        self.rows = []
        self.predictions = []
        self.raw = []
        self.timestamps = []
        self.times = []
        self.segment_ids = []
        self.events = []
        self.results = []
        self.raw_segments = []
        self.raw_sample_count = 0

        self.started_at = None
        self.last_data = None
        self.last_ui_seen = self.clock()
        self.segment = 0
        self.run_serial = 0
        self.session_id = ''
        self.session_started = datetime.now(timezone.utc).isoformat()
        self.rng = np.random.default_rng(42)

        self.replay_data = None
        self.replay_events = []
        self.replay_position = 0
        self.replay_next = 0.
        self.replay_anchor = None

        if config.get('replay_edf'):
            import mne
            recording = Path(config['replay_edf']).resolve()
            raw_edf = mne.io.read_raw_edf(recording, preload=False, verbose=False)
            if not np.isclose(raw_edf.info['sfreq'], self.contract.get('sfreq', 256)):
                raise ValueError('Replay sample rate differs from model.')
            self.replay_data = raw_edf.get_data(picks=self.contract['channels']).astype(np.float32)
            events_tsv = recording.with_name(recording.name.replace('_eeg.edf', '_events.tsv'))
            if events_tsv.exists():
                import pandas as pd
                self.replay_events = pd.read_csv(events_tsv, sep='\t').to_dict('records')
            self.state.update(source='recorded_replay', mode='replay', stream_name=recording.name,
                              is_connected=True, connected=True, recorded_cue='unlabeled')

        self.load_decoder(False)
        self._clear_recording()
        self._refresh_models()

    def _refresh_models(self):
        channels = self.contract.get('channels', [])
        if len(channels) == 20:
            found = list_personal_models(self.reports, self.state['participant'], channels, self.classes)
            self.state.update(model_choices=found.get('models', []), rejected_models=found.get('rejected', []))
            return found
        return dict(models=[], rejected=[])

    def load_decoder(self, demo):
        if self.decoder:
            self.decoder.close()
        self.decoder = None
        if self.checkpoint is None:
            return
        try:
            self.decoder = RealtimeBCIDecoder(
                self.checkpoint,
                threshold=self.state['threshold'],
                udp_host=self.udp_host,
                udp_port=self.udp_port,
                contract_path=self.config.get('contract'),
                allow_unverified=demo or self.config.get('allow_unverified_model', False),
                send_udp=False,
            )
            self.state['error'] = ''
        except Exception as exc:
            self.state['error'] = str(exc)

    def _clear_recording(self):
        import uuid
        self.raw.clear()
        self.timestamps.clear()
        self.times.clear()
        self.segment_ids.clear()
        self.events.clear()
        self.predictions.clear()
        self.rows.clear()
        self.raw_segments.clear()
        self.raw_sample_count = 0
        self.segment = 0
        self.started_at = None
        self.session_id = uuid.uuid4().hex
        self.session_started = datetime.now(timezone.utc).isoformat()
        for h in self.hist.values():
            h.clear()
            h.extend([0.] * 60)
        for s in self.scope.values():
            s.clear()
            s.extend([0.] * 512)
        for w in self.waves.values():
            w.clear()
        for h in self.history.values():
            h.clear()
        self._reset_segment()

    def _reset_segment(self):
        from scipy.signal import butter
        self.segment += 1
        n_ch = len(self.contract.get('channels', [])) or 20
        self.plot_sos = butter(4, [8, 30], fs=256, btype='bandpass', output='sos')
        self.plot_zi = np.zeros((len(self.plot_sos), n_ch, 2))
        win_samples = self.decoder.window_samples if self.decoder else round(self.state.get('window_s', 2.0) * 256)
        self.quality_buffer = RingBuffer(n_ch, win_samples)
        if self.decoder:
            self.decoder.reset_state()
        if self.adapter:
            self.adapter.reset_timestamps()
        self.state.update(
            quality='warming_up', decision='uncertain',
            raw_probabilities=[0.0, 0.0], ema_probabilities=[0.0, 0.0],
            p_raw=0.0, p_smooth=0.0, grasp_command=0, continuous_flex=0.0,
            quality_details={}, prediction_timing=None, prediction_display_ack_ms=None,
        )

    def snapshot(self, seen=False):
        with self.lock:
            if seen:
                self.last_ui_seen = self.clock()
            is_conn = bool(self.state.get('connected') or self.state.get('is_connected'))
            self.state['connected'] = is_conn
            self.state['is_connected'] = is_conn
            is_stream = bool(self.state.get('streaming') or self.state.get('is_streaming'))
            self.state['streaming'] = is_stream
            self.state['is_streaming'] = is_stream
            self.state['demo_mode'] = (self.state.get('mode') == 'demo' or self.state.get('demo_mode', False))
            self.state['source'] = 'synthetic_demo' if self.state['demo_mode'] else ('recorded_replay' if self.replay_data is not None else 'live')
            self.state['step_count'] = len(self.predictions)
            return dict(
                self.state,
                waves={k: list(v) for k, v in self.waves.items()},
                probability_history={k: list(v) for k, v in self.history.items()},
                **{k: list(v) for k, v in self.hist.items()},
                **{k: list(v) for k, v in self.scope.items()}
            )

    def _send_udp(self, packet):
        if self.udp_sock is not None:
            try:
                self.udp_sock.sendto(json.dumps(packet, allow_nan=False).encode('utf-8'), self.udp_target)
                self.packets_sent += 1
                self.state['packets_sent'] = self.packets_sent
            except OSError:
                pass

    def _send_rest_udp(self, reason='paused'):
        self._send_udp(dict(
            timestamp=time.time(),
            sample_end=self.decoder.samples_seen if self.decoder else 0,
            p_raw=0.0,
            p_smooth=0.0,
            grasp_command=0,
            continuous_flex=0.0,
            state_label='REST',
            quality=reason,
            threshold=float(self.state.get('threshold', 0.5)),
        ))

    def _abort(self, reason):
        self._send_rest_udp(reason)
        if self.protocol:
            now = self.clock()
            if hasattr(self.protocol, 'abort'):
                self.protocol.abort(now)
            else:
                event = dict(self.protocol.events[-1]) if self.protocol.events else {}
                event['end'] = now
                event['aborted'] = reason
            self.events.extend(self.protocol.events)
            self.results.append(dict(kind=self.protocol.kind, status='aborted', reason=reason,
                                     run_id=getattr(self.protocol, 'run_id', '')))
            self.protocol = None
            self.state['result'] = self.results[-1]
        self.state.update(
            streaming=False, is_streaming=False, protocol_kind='', cue='', cue_remaining_s=0.,
            status=reason, is_calibrating=False, calib_phase='idle',
            cue_run_id='', awaiting_cue_ack=False, grasp_command=0, continuous_flex=0.,
            p_raw=0., p_smooth=0., quality=reason,
        )
        self._reset_segment()

    def pause(self, reason='paused'):
        self._abort(reason)

    def reset(self):
        self.pause('reset')
        self._clear_recording()
        self.replay_position = 0
        self.replay_anchor = None
        self.replay_next = 0.
        self.state.update(step_count=0, validation_result=None, result=None)
        if not self.state['personalized']:
            self.state.update(calib_acc=None, calibration_result=None)

    def action(self, path, query_or_body=None):
        body = {}
        if isinstance(query_or_body, dict):
            for k, v in query_or_body.items():
                body[k] = v[0] if isinstance(v, list) and len(v) == 1 else v

        with self.lock:
            if path == '/api/toggle_stream':
                if self.state.get('is_training'):
                    raise ValueError('Personal model fitting is in progress. Wait or cancel first.')
                if self.protocol:
                    raise ValueError('Stop the cue run using Stop / Cancel first.')
                if self.state.get('streaming') or self.state.get('is_streaming'):
                    self.pause()
                else:
                    if self.decoder is None:
                        raise ValueError(self.state['error'] or 'No model loaded.')
                    if not self.state.get('demo_mode') and not self.state.get('is_connected') and not self.state.get('connected'):
                        raise ValueError('No fresh, compatible EEG stream.')
                    self.decoder.reset_state()
                    self.segment += 1
                    self.state.update(streaming=True, is_streaming=True)
                    self.raw_segments.append(dict(segment=self.segment, raw_start=self.raw_sample_count))
                    self.replay_anchor = None
                    self.replay_next = 0.
            elif path == '/api/start':
                if not self.state['connected'] and self.state['mode'] != 'demo':
                    raise ValueError('Connect live EEG, or choose synthetic demo to preview the interface.')
                self.decoder.reset_state() if self.decoder else None
                self.state.update(streaming=True, is_streaming=True, status='Recording EEG.', error='')
            elif path == '/api/stop':
                self.cancel_fit.set()
                self._abort(body.get('reason', 'cancelled'))
                self.save()
            elif path == '/api/reset':
                if self.state.get('is_training'):
                    raise ValueError('Cancel personal model fitting before resetting.')
                self.reset()
            elif path in ('/api/toggle_demo', '/api/demo'):
                if self.state.get('is_training'):
                    raise ValueError('Cancel personal fitting before switching source.')
                if self.replay_data is not None:
                    raise ValueError('This studio is replaying a recording. Restart without --replay-edf for live/demo input.')
                if self.rows or self.state['streaming']:
                    raise ValueError('Save or reset the current session before switching data source.')
                new_mode = 'live' if self.state['mode'] == 'demo' else 'demo'
                self.state.update(
                    mode=new_mode, demo_mode=new_mode == 'demo',
                    source='synthetic_demo' if new_mode == 'demo' else 'live',
                    connected=new_mode == 'demo', is_connected=new_mode == 'demo',
                    status='Synthetic demo.' if new_mode == 'demo' else 'Connect EEG to begin.',
                )
                if self.inlet:
                    self.inlet.close_stream()
                self.inlet = self.adapter = None
                self.load_decoder(new_mode == 'demo')
                self._clear_recording()
            elif path == '/api/set_threshold':
                if self.protocol or self.state.get('is_training'):
                    raise ValueError('Threshold is frozen during cue runs and model fitting.')
                val = body.get('val', body.get('threshold', 0.5))
                if isinstance(val, list):
                    val = val[0]
                val = float(val)
                if not np.isfinite(val) or not 0.01 <= val <= 0.99:
                    raise ValueError('Threshold must be between .01 and .99.')
                self.state['threshold'] = val
                self.state['calib_th'] = val
                self.config['threshold'] = val
                if self.decoder:
                    self.decoder.threshold = val
                    self.decoder.dwell_counter = 0
            elif path in ('/api/calibrate', '/api/validate'):
                if self.state.get('is_training'):
                    raise ValueError('A personal model is already being fitted.')
                if self.protocol:
                    raise ValueError('A cue run is already in progress.')
                if not self.state['streaming'] or self.decoder is None:
                    raise ValueError('Start a compatible stream first.')
                if path == '/api/validate' and (self.state['mode'] == 'demo' or self.state.get('demo_mode')):
                    raise ValueError('Synthetic demo cannot validate the NN.')
                if self.replay_data is not None:
                    raise ValueError('Recorded replay cannot run prospective cues. Restart without --replay-edf for your live self-test.')
                kind = 'calibration' if path == '/api/calibrate' else 'validation'
                if kind == 'validation' and self.contract.get('role') == 'personal_training_template':
                    raise ValueError('Run personal calibration first. The architecture template has no fitted classifier.')
                if kind == 'validation' and self.config.get('require_personal_calibration', False) and not self.state['personalized']:
                    raise ValueError('Run personal calibration first. Validation must use a frozen personal model.')
                personal = kind == 'calibration' and self.config.get('personalize_on_calibration', True) and self.state['mode'] != 'demo'
                if personal:
                    participant = body.get('participant', self.state['participant'])
                    if re.fullmatch(r'[A-Za-z0-9_-]{1,40}', participant) is None:
                        raise ValueError('Use a short pseudonymous participant ID: letters, digits, underscore or hyphen.')
                    if self.rows:
                        self.pause('new_personal_calibration')
                        self.save()
                        self.reset()
                    self.state.update(
                        participant=participant, participant_id=participant,
                        personalized=False, threshold=float(self.config.get('threshold', 0.5)), error='',
                        calib_acc=None, calibration_result=None, validation_result=None,
                    )
                    self.checkpoint = self.base_checkpoint
                    self.contract = load_contract(self.checkpoint)
                    self.state['model_path'] = str(self.checkpoint)
                    self.state['model_role'] = self.contract.get('role', 'personal_training_template')
                    self.load_decoder(False)
                    if self.decoder is None:
                        raise ValueError(self.state['error'])
                    self.state.update(streaming=True, is_streaming=True)
                    self.segment += 1
                    self.raw_segments.append(dict(segment=self.segment, raw_start=self.raw_sample_count))

                pairs = int(self.config.get('personal_calibration_pairs', 9) if personal
                            else self.config.get('calibration_pairs', 3) if kind == 'calibration'
                            else self.config.get('validation_pairs', 10))
                self.protocol = CueProtocol(
                    kind, self.clock(), self.decoder.window_samples / self.decoder.sfreq,
                    trials_per_class=pairs, duration=float(self.config.get('trial_duration_s', 8.0)),
                    seed=int(self.config.get('seed', 42)) + len(self.results),
                )
                self.state.update(
                    is_calibrating=True, protocol_kind=kind, total_trials=2 * pairs,
                    cue_total=2 * pairs, cue_remaining_s=float(self.config.get('trial_duration_s', 8.0)),
                    status=f'{kind.title()} in progress.',
                )
                self._send_rest_udp('cue_run_started')
                self.refresh_protocol(self.clock())
            elif path == '/api/export_figure':
                if self.state.get('is_training'):
                    raise ValueError('Calibration data are already saved. Wait for fitting or cancel before exporting.')
                self.pause('saved')
                return self.save()
            elif path == '/api/save':
                if self.state.get('is_training'):
                    raise ValueError('Calibration is already saved. Wait for fitting or stop it first.')
                self._abort('Saved and stopped.')
                return self.save()
            elif path == '/api/cue-presented':
                if self.protocol and hasattr(self.protocol, 'acknowledge'):
                    self.protocol.acknowledge(body.get('run_id'), body.get('trial'), self.clock())
            elif path == '/api/prediction-painted':
                return {}
            elif path == '/api/connect':
                self.state.update(mode='live', connected=True, is_connected=True, error='', status='Connected.')
            else:
                raise ValueError(f'Unknown operation: {path}')
            return self.snapshot()

    def refresh_protocol(self, now):
        if not self.protocol:
            return
        p = self.protocol
        p.advance(now)
        if p.done:
            self.events.extend(p.events)
            protocol_success = False
            personal = p.kind == 'calibration' and self.config.get('personalize_on_calibration', True) and self.state['source'] == 'live'
            try:
                if personal:
                    result = dict(kind='personal_eeg_collection', n_trials=len(p.events),
                                  warning='EEG collected. Personal NN fitting and tuning are pending.')
                else:
                    result = p.finish(self.state['threshold'])
                protocol_success = True
                result['run_id'] = p.run_id
                result['source'] = self.state['source']
                result['model_verified'] = self.contract.get('verified', False)
                result['scope'] = 'exploratory_self_test' if not self.contract.get('verified') else 'prospective_cue_test'
                self.results.append(result)
                if p.kind == 'calibration' and not personal:
                    self.state.update(threshold=result['threshold'], calib_th=result['threshold'],
                                      calib_acc=result['fit_balanced_accuracy'], calibration_result=result)
                    if self.decoder:
                        self.decoder.threshold = result['threshold']
                        self.decoder.dwell_counter = 0
                elif p.kind == 'validation':
                    self.state['validation_result'] = result
                    self.state['result'] = result
            except ValueError as exc:
                self.state['error'] = str(exc)
            self.protocol = None
            self.state.update(is_calibrating=False, calib_phase='idle', protocol_kind='',
                              cue='', cue_remaining_s=0., status='Cue run complete.')
            self.pause('cue_run_complete')
            saved = self.save()
            if protocol_success and personal:
                self.start_personal_training(saved['base'], p.run_id)
            return

        remaining = max(0., p.duration - (now - p.start))
        cue_label = 'action' if p.labels[p.index] else 'rest'
        self.state.update(
            calib_trial=p.index + 1, cue_index=p.index + 1,
            calib_phase=cue_label, cue=cue_label,
            calib_timer=remaining, cue_remaining_s=remaining,
            calib_progress=remaining / p.duration,
        )

    def ingest(self, raw, ts, now=None):
        if not self.state['streaming']:
            return
        now = now if now is not None else self.clock()
        raw = np.asarray(raw, dtype=np.float32)
        ts = np.asarray(ts, dtype=float)

        if self.started_at is None and len(ts):
            self.started_at = float(ts[0])

        self.raw.append(raw.copy())
        self.timestamps.append(ts.copy())
        self.times.append(ts.copy())
        self.segment_ids.append(np.full(len(ts), self.segment, np.int32))
        self.raw_sample_count += raw.shape[1]

        before = self.decoder.samples_seen if self.decoder else 0
        t0 = time.perf_counter()

        output_ready = self.state['personalized'] or not self.config.get('require_personal_calibration', False)
        protocol_output = self.protocol is None or (self.protocol.kind == 'validation' and self.config.get('validation_udp_output', True))
        emit = self.state['source'] == 'live' and protocol_output and output_ready

        outputs = self.decoder.process_chunk_all(raw, emit=False) if self.decoder else []
        latency = (time.perf_counter() - t0) * 1000
        self.state['latency_ms'] = latency

        for packet in outputs:
            offset = packet['sample_end'] - before - 1
            sample_time = float(ts[offset]) if offset < len(ts) else float(ts[-1])
            if self.protocol:
                self.protocol.add(packet, sample_time)
            row = dict(
                packet, lsl_time=sample_time, time_s=sample_time - (self.started_at or sample_time),
                segment=self.segment, participant_id=self.state['participant'],
                model_sha256=self.contract.get('sha256', ''),
                source=self.state['source'], latency_ms=latency,
                protocol=self.protocol.kind if self.protocol else '',
                protocol_run_id=getattr(self.protocol, 'run_id', None) if self.protocol else None,
                trial=self.protocol.index + 1 if self.protocol else None,
                cue=int(self.protocol.labels[self.protocol.index]) if self.protocol else None,
            )
            self.rows.append(row)
            self.predictions.append(row)

            p_action = float(packet['p_raw'])
            p_smooth = float(packet['p_smooth'])
            decision = int(packet['grasp_command'])
            flex = float(packet['continuous_flex'])
            quality = packet['quality']
            threshold = float(packet['threshold'])

            for key in ('p_raw', 'p_smooth', 'grasp_command', 'continuous_flex', 'quality'):
                self.state[key] = packet[key]
            self.state['step_count'] += 1
            self.state.update(
                raw_probabilities=[1.0 - p_action, p_action],
                ema_probabilities=[1.0 - p_smooth, p_smooth],
                decision='right_hand_grasp' if decision else 'rest',
                quality=quality,
            )

            self.hist['prob_history'].append(p_action)
            self.hist['ema_history'].append(p_smooth)
            self.history['time'].append(sample_time)
            self.history['rest'].append(1.0 - p_action)
            self.history['right_hand_grasp'].append(p_action)
            self.history['raw_rest'].append(1.0 - p_action)
            self.history['raw_right_hand_grasp'].append(p_action)
            self.history['ema_rest'].append(1.0 - p_smooth)
            self.history['ema_right_hand_grasp'].append(p_smooth)

            for ch, key in [('C3', 'eeg_c3'), ('Cz', 'eeg_cz'), ('C4', 'eeg_c4')]:
                idx = next((i for i, n in enumerate(self.decoder.channels) if n.casefold() == ch.casefold()), None)
                val = float(np.sqrt(np.mean(self.decoder.last_filtered[idx]**2)) * 1e6) if idx is not None else 0.
                self.hist[key].append(val)
                row[ch + '_rms_uv'] = val if idx is not None else None

            if emit or self.udp_port:
                udp_packet = dict(
                    timestamp=sample_time,
                    sample_end=packet['sample_end'],
                    p_raw=p_action,
                    p_smooth=p_smooth,
                    grasp_command=decision,
                    continuous_flex=flex,
                    state_label='ACTION' if decision else 'REST',
                    quality=quality,
                    threshold=threshold,
                )
                self._send_udp(udp_packet)

        if self.decoder and hasattr(self.decoder, 'last_chunk_filtered'):
            for ch, key in [('C3', 'wave_c3'), ('Cz', 'wave_cz'), ('C4', 'wave_c4')]:
                idx = next((i for i, n in enumerate(self.decoder.channels) if n.casefold() == ch.casefold()), None)
                if idx is not None and self.decoder.last_chunk_filtered.shape[1] > 0:
                    self.scope[key].extend((self.decoder.last_chunk_filtered[idx] * 1e6).tolist())
                    if key.replace('wave_', '').upper() in self.waves:
                        self.waves[key.replace('wave_', '').upper()].extend((self.decoder.last_chunk_filtered[idx] * 1e6).tolist())

    def save(self):
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        name = 'session_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base = self.reports / name

        pd.DataFrame(self.rows).to_csv(str(base) + '_predictions.csv', index=False)
        pd.DataFrame(self.events).to_csv(str(base) + '_events.csv', index=False)
        n_ch = len(self.contract.get('channels', [])) or 20
        data = np.concatenate(self.raw, axis=1) if self.raw else np.empty((n_ch, 0), np.float32)
        times = np.concatenate(self.timestamps) if self.timestamps else np.empty(0)
        channels = self.contract.get('channels', [f'ch{i}' for i in range(n_ch)])
        sfreq = float(self.contract.get('sfreq', 256.0))

        np.savez_compressed(str(base) + '_raw.npz', eeg_volts=data, lsl_timestamps=times,
                            channels=np.array(channels), sfreq=sfreq)
        # Modern eeg.npz format inside folder
        session_folder = self.reports / self.state['participant'] / self.session_id
        session_folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(session_folder / 'eeg.npz', raw=data, timestamps=times,
                            segments=np.concatenate(self.segment_ids) if self.segment_ids else np.empty(0, dtype=int),
                            channels=np.array(channels), sfreq=sfreq, unit='V')

        meta = dict(
            created_utc=datetime.now(timezone.utc).isoformat(), source=self.state['source'], config=self.config,
            participant_id=self.state['participant'], raw_segments=self.raw_segments,
            personal_calibration=self.state['calibration_result'],
            model_contract=self.contract,
            stream=self.adapter.metadata if self.adapter and self.state['mode'] != 'demo' else None,
            results=self.results, n_predictions=len(self.rows), n_raw_samples=data.shape[1],
            calibration_is_validation=False, error=self.state['error'],
        )
        Path(str(base) + '_metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False), encoding='utf-8')
        (session_folder / 'metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False), encoding='utf-8')
        (session_folder / 'events.json').write_text(json.dumps(self.events, indent=2, allow_nan=False), encoding='utf-8')
        (session_folder / 'predictions.json').write_text(json.dumps(self.predictions, indent=2, allow_nan=False), encoding='utf-8')

        fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        if self.rows:
            t = [r['time_s'] for r in self.rows]
            axes[0].plot(t, [r['p_raw'] for r in self.rows], label='p(action)')
            axes[0].plot(t, [r['p_smooth'] for r in self.rows], label='EMA')
            axes[0].plot(t, [r['threshold'] for r in self.rows], '--', label='Threshold at prediction')
            axes[0].legend()
            axes[1].step(t, [r['grasp_command'] for r in self.rows], where='post')
        axes[0].set_ylim(0, 1)
        axes[0].set_ylabel('Probability')
        axes[1].set_ylabel('Decision')
        axes[1].set_xlabel('Seconds from first recorded sample')
        fig.suptitle('SYNTHETIC DEMO - NOT VALIDATION' if self.state['mode'] == 'demo' else 'EEG cue session - see metadata for measured results')
        fig.tight_layout()
        fig.savefig(str(base) + '_plot.png', dpi=130)
        plt.close(fig)

        self.state['last_saved'] = str(base)
        return dict(status='saved', base=str(base), filename_png=name + '_plot.png',
                    filename_csv=name + '_predictions.csv', folder=str(self.reports))

    def start_personal_training(self, snapshot_base, run_id):
        self.training_cancel = threading.Event()
        cancel = self.training_cancel
        participant = self.state['participant']
        self.state.update(is_training=True, training_status='Preparing personal calibration EEG', calib_acc=None, error='')
        output = self.reports / 'personal_models' / f'{participant}_{run_id}.pt'

        def progress(row):
            with self.lock:
                self.state['training_status'] = f"Fitting personal NN, epoch {row['epoch']}, tuning score {row['validation_balanced_accuracy']:.1%} (not final validation)"

        def fit_personal_job():
            try:
                from train_personal_model import load_personal
                from train_lab_model import train_export
                if self.config.get('personal_initialization', 'scratch') == 'pretrained' and load_contract(self.base_checkpoint).get('role') == 'personal_training_template':
                    raise ValueError('Choose scratch initialization for an untrained architecture template.')
                X, y, groups, records, channels = load_personal(snapshot_base, self.base_checkpoint, run_ids=[run_id])
                result = train_export(
                    X, y, groups, records, output, channels,
                    epochs=int(self.config.get('personal_training_epochs', 60)), patience=12, batch_size=16,
                    initial_checkpoint=self.base_checkpoint if self.config.get('personal_initialization', 'scratch') == 'pretrained' else None,
                    freeze_backbone=self.config.get('personal_initialization', 'scratch') == 'pretrained' and not self.config.get('personal_unfreeze', False),
                    progress_callback=progress, stop_event=cancel,
                )
                with self.lock:
                    if cancel.is_set() or self.stop.is_set():
                        raise InterruptedError('Personal calibration cancelled.')
                    contract_path = Path(str(output) + '.json')
                    personal_contract = json.loads(contract_path.read_text(encoding='utf-8'))
                    personal_contract.update(
                        personal_model=True, participant_id=participant, calibration_run_id=run_id,
                        initialization=self.config.get('personal_initialization', 'scratch'),
                    )
                    contract_path.write_text(json.dumps(personal_contract, indent=2), encoding='utf-8')
                    self.checkpoint = output
                    self.contract = load_contract(output)
                    self.state['threshold'] = self.contract['threshold']
                    self.load_decoder(False)
                    if self.decoder is None:
                        raise ValueError(self.state['error'])
                    calibration = dict(
                        kind='personal_nn_calibration', run_id=run_id, participant_id=participant,
                        model_path=str(output), model_sha256=self.contract['sha256'],
                        threshold=self.contract['threshold'],
                        tuning_balanced_accuracy=result['validation_balanced_accuracy_at_threshold'],
                        internal_test_balanced_accuracy=result['test_balanced_accuracy'],
                        warning='Personal fitting and tuning completed. Record new validation cues to evaluate performance.',
                    )
                    self.results.append(calibration)
                    self.state.update(
                        personalized=True, model_ready=True, model_path=str(output),
                        calibration_result=calibration, model_role='personal',
                        calib_acc=calibration['tuning_balanced_accuracy'],
                        training_status='Personal model ready. Start Stream, then Validate on new cues.', error='',
                    )
                    self.save()
            except Exception as exc:
                with self.lock:
                    self.state.update(error=str(exc), training_status='Personal model fitting failed or was cancelled.', personalized=False)
            finally:
                with self.lock:
                    self.state['is_training'] = False
                    self.state['training'] = False

        self.training_thread = threading.Thread(target=fit_personal_job, daemon=True)
        self.training_thread.start()

    def worker(self):
        retry_at = 0.
        demo_next = 0.
        while not self.stop.is_set():
            now = self.clock()
            with self.lock:
                self.refresh_protocol(now)
                demo = self.state['mode'] == 'demo'
            if demo:
                with self.lock:
                    if self.state['streaming'] and self.decoder and now >= demo_next:
                        n = self.decoder.step_samples
                        end = now
                        ts = end - np.arange(n - 1, -1, -1) / self.decoder.sfreq
                        raw = self.rng.normal(0, 5e-6, (self.decoder.n_ch, n)).astype(np.float32)
                        self.ingest(raw, ts, now)
                        demo_next = now + n / self.decoder.sfreq
                self.stop.wait(0.01)
                continue
            if self.replay_data is not None:
                with self.lock:
                    if self.state['streaming'] and now >= self.replay_next:
                        start = self.replay_position
                        end = min(start + self.decoder.step_samples, self.replay_data.shape[1])
                        if start == end:
                            self.pause('recording_complete')
                        else:
                            if self.replay_anchor is None:
                                self.replay_anchor = now - (end - 1) / self.decoder.sfreq
                            ts = self.replay_anchor + np.arange(start, end) / self.decoder.sfreq
                            self.ingest(self.replay_data[:, start:end], ts, now)
                            self.replay_position = end
                            self.replay_next = max(self.replay_next, now) + (end - start) / self.decoder.sfreq
                            cue = next((r['trial_type'] for r in self.replay_events if r['sample'] <= end - 1 < r['sample'] + r['duration'] * self.decoder.sfreq), 'unlabeled')
                            self.state.update(recorded_cue=cue, replay_time_s=end / self.decoder.sfreq)
                self.stop.wait(0.01)
                continue
            if self.inlet is None:
                if now < retry_at:
                    self.stop.wait(0.05)
                    continue
                try:
                    inlet, adapter = connect(self.config, self.contract)
                    with self.lock:
                        self.inlet, self.adapter = inlet, adapter
                        self.state['stream_name'] = adapter.metadata['name']
                        self.last_data = self.clock()
                        self.state['connected'] = True
                        self.state['is_connected'] = True
                        if self.decoder:
                            self.state['error'] = ''
                except Exception as exc:
                    with self.lock:
                        self.state['connected'] = False
                        self.state['is_connected'] = False
                        self.state['error'] = str(exc)
                    retry_at = self.clock() + 2.
                    continue
            try:
                chunk, ts = self.inlet.pull_chunk(timeout=0.05, max_samples=64)
                now = self.clock()
                with self.lock:
                    if self.state['mode'] == 'demo':
                        continue
                    if chunk:
                        if not self.state['streaming'] and now - ts[-1] > self.adapter.max_age:
                            self.inlet.flush()
                            self.adapter.reset_timestamps()
                            self.state['connected'] = False
                            self.state['is_connected'] = False
                            self.last_data = now
                            continue
                        raw, ts = self.adapter.transform(chunk, ts, now)
                        self.last_data = now
                        self.state['connected'] = True
                        self.state['is_connected'] = True
                        self.state['sample_age_s'] = max(0., now - ts[-1])
                        self.ingest(raw, ts, now)
                    elif self.last_data is not None and now - self.last_data > 1.0:
                        raise ValueError('EEG stream timed out; commands stopped. Reconnect and start again.')
            except Exception as exc:
                with self.lock:
                    self.state['error'] = str(exc)
                    self.state['connected'] = False
                    self.state['is_connected'] = False
                    self.pause('stream_error')
                    if self.inlet:
                        self.inlet.close_stream()
                    self.inlet = None
                    self.adapter = None
                    retry_at = self.clock() + 1.
        with self.lock:
            self.pause('shutdown')
            if self.decoder:
                self.decoder.close()
            if self.inlet:
                self.inlet.close_stream()
            if self.udp_sock:
                try:
                    self.udp_sock.close()
                except OSError:
                    pass
                self.udp_sock = None


class Handler(BaseHTTPRequestHandler):
    studio = None

    def log_message(self, *args):
        pass

    def respond(self, status, value, content_type='application/json'):
        body = json.dumps(value, allow_nan=False).encode('utf-8') if content_type == 'application/json' else value.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type + '; charset=utf-8')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            html_path = self.studio.root / 'studio.html'
            if not html_path.exists():
                html_path = self.studio.root / 'threeclass_studio.html'
            self.respond(200, html_path.read_text(encoding='utf-8'), 'text/html')
        elif self.path == '/api/state':
            self.respond(200, self.studio.snapshot(seen=True))
        else:
            self.respond(404, dict(error='Not found'))

    def do_POST(self):
        origin = self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'):
            self.respond(403, dict(error='Local console origin required.'))
            return
        try:
            url = urlparse(self.path)
            length = int(self.headers.get('Content-Length', '0'))
            body = json.loads(self.rfile.read(length) or b'{}') if length > 0 else {}
            query = parse_qs(url.query)
            params = dict(body)
            for k, v in query.items():
                if k not in params:
                    params[k] = v[0] if len(v) == 1 else v
            result = self.studio.action(url.path, params)
            self.respond(200, result)
        except (ValueError, RuntimeError) as exc:
            with self.studio.lock:
                self.studio.state['error'] = str(exc)
            self.respond(409, dict(error=str(exc)))
        except Exception as exc:
            self.respond(500, dict(error=str(exc)))


def main(default_protocol=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'lab_config.json')
    p.add_argument('--port', type=int, default=8052)
    p.add_argument('--demo', action='store_true')
    p.add_argument('--no-browser', action='store_true')
    p.add_argument('--mode', choices=['lsl', 'calibrate', 'simulate'], default='lsl')
    p.add_argument('--ckpt')
    p.add_argument('--trials', type=int)
    p.add_argument('--trial-dur', type=float)
    p.add_argument('--udp-host')
    p.add_argument('--udp-port', type=int)
    p.add_argument('--threshold', type=float)
    p.add_argument('--lsl-stream')
    p.add_argument('--allow-unverified-model', action='store_true')
    p.add_argument('--replay-edf', type=Path, help='Paced recorded-data visualization.')

    args = p.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))

    for key, value in [('checkpoint', args.ckpt), ('calibration_pairs', args.trials),
                       ('trial_duration_s', args.trial_dur), ('udp_host', args.udp_host),
                       ('udp_port', args.udp_port), ('threshold', args.threshold),
                       ('stream_name', args.lsl_stream)]:
        if value is not None:
            config[key] = value

    if args.allow_unverified_model:
        config['allow_unverified_model'] = True
    if args.replay_edf:
        config['replay_edf'] = str(args.replay_edf.resolve())

    torch.set_num_threads(int(config.get('torch_threads', 1)))
    studio = Studio(config)

    if args.demo or args.mode == 'simulate':
        studio.action('/api/toggle_demo', {})
    if default_protocol or args.mode == 'calibrate':
        print('Open the studio, Start Stream, then select Calibrate. Calibration requires visible cues.', flush=True)

    handler = type('StudioHandler', (Handler,), {'studio': studio})
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler)
    worker = threading.Thread(target=studio.worker, daemon=True)
    worker.start()

    url = f'http://127.0.0.1:{args.port}'
    print(f'Binary BCI Studio: {url}\nReports: {studio.reports}\nModel: {studio.checkpoint}\n{studio.state["error"]}', flush=True)

    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        studio.training_cancel.set()
        studio.stop.set()
        worker.join(timeout=5.)
        server.server_close()
        if studio.training_thread:
            studio.training_thread.join(timeout=10.)
        if studio.rows:
            with studio.lock:
                studio.save()


if __name__ == '__main__':
    main()
