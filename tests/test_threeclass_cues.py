"""Display-acknowledged cue timing must not invent ground-truth intervals."""
import copy
import numpy as np
import pytest

from threeclass_cues import ThreeClassProtocol


def test_acknowledgement_controls_onset_and_repeated_ack_is_idempotent():
    p = ThreeClassProtocol('calibration', 100., 2, require_ack=True)
    assert p.awaiting_ack and not p.events and p.ack_deadline == 103.
    p.advance(100.5)
    assert not p.events
    first = p.acknowledge(p.run_id, 1, 100.7)
    assert first['onset'] == 100.7 and first['ack_received_at'] == 100.7
    assert first['display_acknowledged'] and not p.awaiting_ack
    assert p.acknowledge(p.run_id, 1, 101.2)['onset'] == 100.7
    assert len(p.events) == 1
    with pytest.raises(ValueError, match='Stale or unexpected'):
        p.acknowledge('old-run', 1, 101.2)
    with pytest.raises(ValueError, match='Stale or unexpected'):
        p.acknowledge(p.run_id, 2, 101.2)


def test_transition_gaps_are_unlabeled_and_windows_use_acknowledged_cue():
    p = ThreeClassProtocol('validation', 100., 1, require_ack=True)
    p.add(dict(class_id=0), 104.5)
    assert not p.predictions
    p.acknowledge(p.run_id, 1, 101.)
    p.add(dict(class_id=0), 105.49)
    p.add(dict(class_id=0), 105.5)
    assert len(p.predictions) == 1
    p.advance(109.)
    assert p.awaiting_ack and p.index == 1 and p.events[0]['complete']
    p.add(dict(class_id=0), 109.)
    assert len(p.predictions) == 1
    p.acknowledge(p.run_id, 2, 110.)
    assert p.events[0]['end'] == 109. and p.events[1]['onset'] == 110.
    p.add(dict(class_id=1), 114.49)
    p.add(dict(class_id=1), 114.5)
    assert len(p.predictions) == 2 and p.predictions[-1]['trial'] == 2
    with pytest.raises(ValueError, match='Stale or unexpected'):
        p.acknowledge(p.run_id, 1, 110.2)


def test_timeout_missing_ack_and_no_retroactive_cue_backlog():
    p = ThreeClassProtocol('validation', 100., 1, require_ack=True)
    with pytest.raises(ValueError, match='timed out'):
        p.advance(103.01)
    with pytest.raises(ValueError, match='timed out'):
        p.acknowledge(p.run_id, 1, 103.01)
    assert not p.events
    p = ThreeClassProtocol('validation', 100., 1, require_ack=True)
    p.acknowledge(p.run_id, 1, 100.1)
    p.advance(130.)
    assert p.index == 1 and p.awaiting_ack and not p.done
    assert len(p.events) == 1 and p.events[0]['end'] == 130.
    assert p.ack_deadline == 133.


def test_stale_worker_clock_cannot_abort_a_concurrently_acknowledged_cue():
    p = ThreeClassProtocol('validation', 100., 1, require_ack=True)
    p.advance(99.9)
    assert p.awaiting_ack and not p.events
    p.acknowledge(p.run_id, 1, 100.2)
    p.advance(100.1)
    assert p.events[0]['onset'] == 100.2 and not p.aborted and not p.events[0]['complete']


def test_abort_initial_wait_or_transition_preserves_completed_events():
    empty = ThreeClassProtocol('calibration', 100., 1, require_ack=True)
    empty.abort(100.5)
    assert empty.aborted and not empty.done and not empty.events and not empty.awaiting_ack
    empty.advance(200.)
    assert not empty.events
    with pytest.raises(ValueError, match='Stale or unexpected'):
        empty.acknowledge(empty.run_id, 1, 100.6)
    p = ThreeClassProtocol('calibration', 100., 1, require_ack=True)
    p.acknowledge(p.run_id, 1, 100.2)
    p.advance(108.2)
    completed = copy.deepcopy(p.events)
    p.abort(108.6)
    assert p.events == completed and p.events[-1]['complete']
    assert p.aborted and not p.done
    active = ThreeClassProtocol('validation', 100., 1, require_ack=True)
    active.acknowledge(active.run_id, 1, 100.2)
    active.abort(102.)
    assert active.events[-1]['end'] == 102. and not active.events[-1]['complete']
    assert active.aborted and not active.done


def test_balanced_calibration_blocks_and_completed_validation():
    p = ThreeClassProtocol('calibration', 100., 12, require_ack=True)
    assert all(set(p.labels[a:a+3]) == {0, 1, 2} for a in range(0, 36, 3))
    now = 100.
    for trial in range(1, 37):
        p.acknowledge(p.run_id, trial, now+.1)
        now += 8.1
        p.advance(now)
    assert p.done and not p.aborted and len(p.events) == 36
    assert [e['calibration_block'] for e in p.events] == [i//3 for i in range(36)]
    assert all(e['complete'] for e in p.events)
    finished = copy.deepcopy(p.events)
    p.abort(now+.1)
    assert p.done and p.events == finished
    with pytest.raises(ValueError, match='Stale or unexpected'):
        p.acknowledge(p.run_id, 36, now+.1)
    v = ThreeClassProtocol('validation', 0., 10)
    assert all(np.sum(v.labels == i) == 10 for i in range(3))


def test_offline_immediate_mode_retains_existing_clock_semantics():
    p = ThreeClassProtocol('validation', 100., 1)
    assert len(p.events) == 1 and p.events[0]['onset'] == 100. and not p.awaiting_ack
    p.add(dict(class_id=0), 104.25)
    p.add(dict(class_id=0), 104.5)
    assert len(p.predictions) == 1
    p.advance(130.)
    assert p.index == 1 and p.events[0]['end'] == 130. and p.events[1]['onset'] == 130.
    assert not p.done


@pytest.mark.parametrize('argument,value', [
    ('now', float('nan')), ('duration', float('inf')), ('duration', 6.),
    ('ack_timeout_s', 0.), ('ack_timeout_s', float('nan')),
    ('per_class', 0), ('per_class', 1.5), ('per_class', True), ('require_ack', 'yes')])
def test_invalid_controls_fail_before_any_cue(argument, value):
    args = dict(kind='calibration', now=100., per_class=12)
    args[argument] = value
    with pytest.raises(ValueError):
        ThreeClassProtocol(**args)
