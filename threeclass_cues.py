"""Three-class cues whose usable EEG intervals follow display acknowledgement.

An acknowledgement records the server's receipt after the browser rendered a
cue. It is not a measured photon/display onset. Transition intervals remain
unlabeled and every scored window must fit inside one acknowledged cue.
"""
import math
import numbers
import uuid

import numpy as np

CLASSES = ['rest', 'volar_grasp', 'finger_extension']


def _finite_time(value, label='Cue clock'):
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{label} must be finite.') from exc
    if not math.isfinite(result):
        raise ValueError(f'{label} must be finite.')
    return result


class ThreeClassProtocol:
    """Balanced cue sequence with optional explicit browser-render ACKs.

    ``require_ack=False`` preserves immediate, clock-driven cues for offline
    simulations. Production callers should require acknowledgement. Calibration
    consists of balanced randomized blocks so whole later blocks can
    be reserved for tuning. Validation uses a shuffled balanced sequence.
    """

    def __init__(self, kind, now, per_class, duration=8., seed=42,
                 require_ack=False, ack_timeout_s=3., classes=CLASSES, window_s=4., settle_s=.5):
        now = _finite_time(now)
        duration = _finite_time(duration, 'Cue duration')
        ack_timeout_s = _finite_time(ack_timeout_s, 'Acknowledgement timeout')
        self.window_s = _finite_time(window_s, 'Decoder window')
        self.settle_s = _finite_time(settle_s, 'Cue settling time')
        if kind not in ('calibration', 'validation'):
            raise ValueError('Use calibration or validation cues.')
        if isinstance(per_class, bool) or not isinstance(per_class, numbers.Integral) or per_class < 1:
            raise ValueError('Cue count per class must be a positive integer.')
        if duration < 7. or ack_timeout_s <= 0:
            raise ValueError('Cues require at least seven seconds and a positive acknowledgement timeout.')
        if self.window_s <= 0 or self.settle_s < 0 or self.window_s+self.settle_s >= duration:
            raise ValueError('The decoder window and settling time must fit inside a cue.')
        if not isinstance(require_ack, bool):
            raise ValueError('require_ack must be a Boolean.')
        self.classes = list(classes)
        n_classes = len(self.classes)
        self.kind, self.duration = kind, duration
        self.require_ack, self.ack_timeout_s = require_ack, ack_timeout_s
        self.run_id = uuid.uuid4().hex
        rng = np.random.default_rng(seed)
        if kind == 'calibration':
            self.labels = np.concatenate([rng.permutation(n_classes) for _ in range(int(per_class))])
        else:
            self.labels = np.tile(np.arange(n_classes), int(per_class))
            rng.shuffle(self.labels)
        self.index, self.done, self.aborted = 0, False, False
        self.events, self.predictions = [], []
        self.awaiting_ack, self.ack_deadline = False, None
        self._active = None
        self._waiting_since = None
        self._begin_or_wait(now)

    def _begin_or_wait(self, now):
        if self.require_ack:
            self.awaiting_ack = True
            self._waiting_since = now
            self.ack_deadline = now+self.ack_timeout_s
        else:
            self._open_event(now)

    def _open_event(self, now):
        event = dict(trial=self.index+1, label=self.classes[int(self.labels[self.index])],
                     onset=float(now), end=None, complete=False,
                     run_id=self.run_id, kind=self.kind,
                     display_acknowledged=self.require_ack)
        if self.kind == 'calibration':
            event['calibration_block'] = self.index//len(self.classes)
        if self.require_ack:
            event['ack_received_at'] = float(now)
        self.events.append(event)
        self._active = event
        self.awaiting_ack, self.ack_deadline = False, None
        self._waiting_since = None

    def acknowledge(self, run_id, trial, now):
        """Start only the matching current cue; duplicate ACKs never reset it."""
        now = _finite_time(now)
        if (run_id != self.run_id or self.done or self.aborted or
                isinstance(trial, bool) or not isinstance(trial, numbers.Integral) or
                int(trial) != self.index+1):
            raise ValueError('Stale or unexpected cue acknowledgement.')
        if self.awaiting_ack:
            if now < self._waiting_since:
                raise ValueError('Cue clock moved backwards before acknowledgement.')
            if now > self.ack_deadline:
                raise ValueError('Cue display acknowledgement timed out.')
            self._open_event(now)
        elif self._active is None:
            raise ValueError('No active cue can accept an acknowledgement.')
        elif now < self._active['onset']:
            raise ValueError('Cue clock moved backwards before acknowledgement.')
        return dict(self._active)

    def advance(self, now):
        now = _finite_time(now)
        if self.done or self.aborted:
            return
        if self.awaiting_ack:
            # A worker can capture its clock just before a concurrent HTTP ACK
            # acquires the shared protocol lock. An earlier observation cannot
            # advance the newly created cue and must not abort a valid run.
            if now < self._waiting_since:
                return
            if now > self.ack_deadline:
                raise ValueError('Cue display acknowledgement timed out.')
            return
        if now < self._active['onset']:
            return
        if now-self._active['onset'] < self.duration:
            return
        # A delayed worker closes only its actual active cue. It never invents
        # intervening cues or backdates the next cue's acknowledged start.
        self._active.update(end=float(now), complete=True)
        self._active = None
        self.index += 1
        if self.index == len(self.labels):
            self.done = True
            self.awaiting_ack, self.ack_deadline = False, None
        else:
            self._begin_or_wait(now)

    def add(self, packet, sample_time):
        sample_time = _finite_time(sample_time, 'Sample timestamp')
        if self.done or self.aborted or self.awaiting_ack or self._active is None:
            return False
        onset = self._active['onset']
        if sample_time-self.window_s < onset+self.settle_s or sample_time > onset+self.duration:
            return False
        self.predictions.append(dict(packet, trial=self._active['trial'],
                                     timestamp=sample_time, truth=self._active['label']))
        return True

    def abort(self, now):
        """Abort without corrupting a completed cue during an unlabeled gap."""
        now = _finite_time(now)
        if self.done or self.aborted:
            return
        if self._active is not None:
            if now < self._active['onset']:
                raise ValueError('Cue clock moved backwards before abort.')
            self._active.update(end=float(now), complete=False)
        self._active = None
        self.awaiting_ack, self.ack_deadline = False, None
        self._waiting_since = None
        self.aborted = True
        # done means ALL requested cues completed, not merely terminal. Keeping
        # it false also prevents legacy scorers from accepting partial runs.
        self.done = False
