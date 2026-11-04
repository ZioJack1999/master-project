"""Discovery must target the configured identity and fail on ambiguity."""
import sys
from types import SimpleNamespace
import pytest
from lsl_acquisition import select_stream
from threeclass_acquisition import discover_eeg_streams


def stream(name, source, kind='EEG'):
    return SimpleNamespace(name=lambda:name, source_id=lambda:source, type=lambda:kind)


def test_selected_identity_is_queried_even_when_another_eeg_outlet_responds_first(monkeypatch):
    unrelated, intended = stream('other', 'other_id'), stream('X24', 'x24_id')
    calls = []
    def resolve(prop, value, **options):
        calls.append((prop, value, options))
        return [intended] if prop == 'source_id' and value == 'x24_id' else [unrelated]
    monkeypatch.setitem(sys.modules, 'pylsl', SimpleNamespace(resolve_byprop=resolve,
        resolve_streams=lambda **_: [unrelated, intended]))
    c = dict(source_id='x24_id', stream_name='X24')
    assert select_stream(discover_eeg_streams(c), c) is intended
    assert calls == [('source_id', 'x24_id', dict(minimum=2, timeout=1.))]


def test_name_selection_duplicate_identity_and_non_eeg_are_checked(monkeypatch):
    wanted = stream('X24', 'one')
    results = [wanted]
    calls = []
    def resolve(prop, value, **options):
        calls.append(prop)
        return results
    monkeypatch.setitem(sys.modules, 'pylsl', SimpleNamespace(resolve_byprop=resolve,
        resolve_streams=lambda **_: results))
    c = dict(stream_name='X24')
    assert select_stream(discover_eeg_streams(c), c) is wanted and calls == ['name']
    results.append(stream('X24', 'two'))
    with pytest.raises(ValueError, match='found 2'):
        select_stream(discover_eeg_streams(c), c)
    results[:] = [stream('X24', 'one', 'Markers')]
    with pytest.raises(ValueError, match='found 0'):
        select_stream(discover_eeg_streams(c), c)


def test_without_identity_collect_all_before_deciding_uniqueness(monkeypatch):
    calls = []
    streams = [stream('one', 'a'), stream('two', 'b')]
    def all_streams(**kw):
        calls.append(kw)
        return streams
    monkeypatch.setitem(sys.modules, 'pylsl', SimpleNamespace(resolve_byprop=lambda *a,**kw:[],
        resolve_streams=all_streams))
    with pytest.raises(ValueError, match='found 2'):
        select_stream(discover_eeg_streams({}), {})
    assert calls == [dict(wait_time=1.)]
