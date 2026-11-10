"""Software-only integration checks for the expanded X24 lab ZIPs."""
from collections import Counter
import json
import os
import subprocess
import sys
from zipfile import ZipFile

import numpy as np
import pytest

from multiclass_lab_bundle import CLASS_SETS, FILES, MANIFEST, policy, verify_bundle
from package_multiclass_lab import build
from threeclass_studio import ThreeClassStudio


@pytest.mark.parametrize('count', CLASS_SETS)
def test_package_and_personal_validation(count, tmp_path):
    output = build(tmp_path/'lab.zip', count)
    assert build(tmp_path/'repeat.zip', count).read_bytes() == output.read_bytes()
    with ZipFile(output) as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {
            f'BCI_{count}Class_X24_Lab/{name}' for name in (*FILES, MANIFEST)
        }
        archive.extractall(tmp_path)
    root = tmp_path/f'BCI_{count}Class_X24_Lab'
    config = verify_bundle(root)
    if os.name == 'nt':
        command = ['cmd.exe', '/d', '/c', 'Launch_MULTICLASS_X24.bat', '--verify-only']
    else:
        command = [sys.executable, 'run_x24.py', '--task', 'multiclass', '--verify-only']
    completed = subprocess.run(command, cwd=root, capture_output=True, text=True,
                               env=dict(os.environ, X24_DECODER_PYTHON=sys.executable), check=True)
    assert json.loads(completed.stdout)['classes'] == CLASS_SETS[count]
    assert config['calibration_cues_per_class'] == 12
    assert config['validation_cues_per_class'] == 10
    assert config['policy'] == policy(count)

    # Use shorter synthetic runs through the normal actions and the real fitter.
    config.update(calibration_cues_per_class=8, validation_cues_per_class=1,
                  cue_duration_s=7., epochs=1, ensemble_seeds=[42], threads=1)
    now = [100.]
    studio = ThreeClassStudio(config, root=root, clock=lambda: now[0])
    assert studio.snapshot()['classes'] == CLASS_SETS[count]
    assert studio.state['model_role'] == 'untrained' and studio.reference_checkpoint is None
    studio.state['connected'] = True
    with pytest.raises(ValueError, match='before validation'):
        studio.action('/api/validate')

    rng = np.random.default_rng(42)
    t = np.arange(7*256)/256

    def collect(path):
        studio.action(path)
        protocol = studio.protocol
        while studio.protocol is protocol:
            studio.snapshot(seen=True)
            studio.action('/api/cue-presented', dict(run_id=protocol.run_id, trial=protocol.index+1))
            label = int(protocol.labels[protocol.index])
            raw = rng.normal(0, 2e-6, (20, len(t))).astype(np.float32)
            raw[label] += 12e-6*np.sin(2*np.pi*11*t)
            studio.ingest(raw, now[0]+t)
            now[0] += 7.
            studio.snapshot(seen=True)
            studio._refresh(now[0])
        return protocol

    calibration = collect('/api/calibrate')
    studio.training_thread.join(timeout=60)
    assert studio.state['model_ready'], studio.state['error']
    assert studio.checkpoint.name == 'personal_multiclass.pt'
    assert studio.decoder.classes == CLASS_SETS[count]
    assert studio.decoder.policy.config == policy(count)
    fit = json.loads((studio.checkpoint.parent/'training.json').read_text())
    assert fit['split_details']['reserved_tuning_block_ids'] == [6, 7]
    assert set(fit['train_cues']).isdisjoint(fit['tuning_cues'])
    for block in range(8):
        assert Counter(e['label'] for e in calibration.events if e['calibration_block'] == block) == dict.fromkeys(CLASS_SETS[count], 1)

    selected = studio.state['model_choices'][0]['model_id']
    studio.action('/api/load-model', dict(model_id=selected))
    checkpoint_hash = studio.state['model_sha256']
    validation = collect('/api/validate')
    assert validation.run_id != calibration.run_id
    result = studio.state['result']
    assert result['frozen']['model_sha256'] == checkpoint_hash
    assert result['frozen']['policy'] == policy(count)
    metrics = result['metrics']
    assert metrics['n_cues'] == count
    assert metrics['n_scored'] + metrics['n_rejected'] == count
    assert np.array(metrics['confusion_matrix']).shape == (count, count+1)
    assert metrics['confusion_columns'] == CLASS_SETS[count]+['rejected']
    assert all(len(row['probabilities_raw']) == count for row in studio.predictions)
    assert (studio.reports/'self').is_dir()


def test_changed_bundle_file_is_refused(tmp_path):
    output = build(tmp_path/'lab.zip', 5)
    with ZipFile(output) as archive:
        archive.extractall(tmp_path)
    root = tmp_path/'BCI_5Class_X24_Lab'
    (root/'multiclass_config.json').write_text('{}')
    with pytest.raises(ValueError, match='file changed: multiclass_config.json'):
        verify_bundle(root)
