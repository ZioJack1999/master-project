"""Standalone package closure, integrity and fresh-stream preflight checks."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile
import numpy as np
import pytest
from package_threeclass_lab import build
from threeclass_preflight import inspect
from verify_threeclass_bundle import verify, REFERENCE, REQUIRED_FILES

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT/'threeclass_config.json').read_text())


def test_stock_and_custom_montage_require_personal_calibration_without_relabeling():
    stock = inspect(config(), ROOT)
    custom_config = json.loads((ROOT/'threeclass_config.custom20.example.json').read_text())
    custom = inspect(custom_config, ROOT)
    assert stock['configuration_valid'] and stock['reference_compatible']
    assert custom['configuration_valid'] and not custom['reference_compatible']
    assert custom['acquisition_contract']['channels'] == custom_config['channels']
    assert 'incompatible' in custom['initial_model_status']
    assert stock['personal_calibration_required'] and custom['personal_calibration_required']
    assert not stock['acquisition_ready'] and not custom['acquisition_ready']


@pytest.mark.parametrize('field,value', [('sfreq', 250), ('window_s', 2.),
    ('stride_s', .5), ('settle_s', float('nan')), ('max_gap_s', 0),
    ('channels', ['C3']*20)])
def test_preflight_rejects_unsupported_geometry_and_missing_sites(field, value):
    c = config()
    c[field] = value
    result = inspect(c, ROOT)
    assert not result['configuration_valid'] and result['errors']


def fake_lsl(monkeypatch, now=100., stale=False, duplicate=False):
    import lsl_acquisition
    c = config()
    streams = [SimpleNamespace(name=lambda:'X24', source_id=lambda:'lab_x24', type=lambda:'EEG')]
    if duplicate:
        streams.append(SimpleNamespace(name=lambda:'X24-copy', source_id=lambda:'other', type=lambda:'EEG'))
    closed = []
    meta = dict(name='X24', source_id='lab_x24', type='EEG', sfreq=256,
                channel_count=26, labels=['prefix'+str(i) for i in range(6)]+c['channels'],
                units=['']*6+['uV']*20)
    class Inlet:
        def __init__(self, stream, **kwargs): self.stream=stream
        def info(self, **kwargs): return self.stream
        def open_stream(self, **kwargs): pass
        def pull_chunk(self, **kwargs):
            last = now-5 if stale else now
            return np.ones((64, 26)).tolist(), (last-np.arange(63,-1,-1)/256).tolist()
        def close_stream(self): closed.append(self.stream)
    monkeypatch.setitem(sys.modules, 'pylsl', SimpleNamespace(StreamInlet=Inlet,
        local_clock=lambda:now, proc_clocksync=1, resolve_byprop=lambda *a,**kw:streams,
        resolve_streams=lambda **kw:streams))
    monkeypatch.setattr(lsl_acquisition, 'stream_metadata', lambda info:copy.deepcopy(meta))
    return closed


def test_preflight_checks_real_adapter_field_mapping_units_and_fresh_timestamps(monkeypatch):
    closed = fake_lsl(monkeypatch)
    result = inspect(config(), ROOT, inspect_lsl=True)
    assert result['acquisition_ready']
    assert result['streams'][0]['compatible_metadata']
    assert result['streams'][0]['fresh_samples_checked']
    assert result['streams'][0]['fresh_samples'] == 64 and len(closed) == 1


def test_preflight_does_not_call_stale_or_ambiguous_stream_ready(monkeypatch):
    closed = fake_lsl(monkeypatch, stale=True)
    stale = inspect(config(), ROOT, inspect_lsl=True)
    assert not stale['acquisition_ready'] and len(closed) == 1
    assert 'Stale' in stale['streams'][0]['error']
    closed = fake_lsl(monkeypatch, duplicate=True)
    ambiguous = inspect(config(), ROOT, inspect_lsl=True)
    assert not ambiguous['acquisition_ready'] and len(closed) == 2
    assert any('found 2' in s for s in ambiguous['errors'])


def test_threeclass_zip_is_standalone_and_detects_changed_files(tmp_path):
    output = tmp_path/'threeclass-test.zip'
    build(output, root=ROOT)
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        assert all(name.startswith('BCI_ThreeClass_Lab/') for name in names)
        assert not any('bci_studio_gui' in n or 'eegnet_' in n or 'experiments/' in n or
                       'session_reports/' in n or n.endswith(('.edf','.npz','.npy','.tsv')) for n in names)
        archive.extractall(tmp_path/'extracted')
    extracted = tmp_path/'extracted/BCI_ThreeClass_Lab'
    result = verify(extracted)
    assert result['threeclass_package_valid']
    assert all((extracted/f).is_file() for f in REQUIRED_FILES)
    # Git/Windows text newline conversion is accepted while model bytes stay exact.
    py = extracted/'threeclass_preflight.py'
    py.write_bytes(py.read_bytes().replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))
    assert verify(extracted)['threeclass_package_valid']
    py.write_text('tampered', encoding='utf-8')
    assert not verify(extracted)['threeclass_package_valid']
    with pytest.raises(ValueError, match='already exists'):
        build(output, root=ROOT)
    model = extracted/REFERENCE
    original = model.read_bytes()
    model.write_bytes(original+b'changed')
    result = verify(extracted)
    assert any(p['file'] == REFERENCE for p in result['problems'])
