"""Supervisor rejects ambiguous acquisition before opening an inlet."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import pylsl
import run_x24


def test_duplicate_outlets_are_observed_before_opening_inlet(monkeypatch):
    calls = []

    def resolve(prop, value, *, minimum, timeout):
        calls.append((prop, value, minimum, timeout))
        # Model liblsl's early return once its minimum has been reached.
        return [object(), object()][:minimum]

    inlet = Mock(side_effect=AssertionError('Ambiguous stream must not be opened'))
    monkeypatch.setattr(pylsl, 'resolve_byprop', resolve)
    monkeypatch.setattr(pylsl, 'StreamInlet', inlet)
    with pytest.raises(RuntimeError, match='Duplicate X24 outlets'):
        run_x24.probe_stream({}, {}, SimpleNamespace(poll=lambda: None))
    assert calls == [('source_id', run_x24.SOURCE_ID, 2, 1.)]
    inlet.assert_not_called()


def test_dead_bridge_does_not_attempt_network_discovery(monkeypatch):
    resolve = Mock(side_effect=AssertionError('Dead bridge must fail first'))
    monkeypatch.setattr(pylsl, 'resolve_byprop', resolve)
    with pytest.raises(RuntimeError, match='X24 bridge exited'):
        run_x24.probe_stream({}, {}, SimpleNamespace(poll=lambda: 7, returncode=7))
    resolve.assert_not_called()
