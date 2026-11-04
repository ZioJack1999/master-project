"""Saved models must match the wearer, physical sites and calibration provenance."""
import hashlib
import json
import os
from pathlib import Path
import pytest
from threeclass_model_store import list_personal_models, validate_personal_model

CHANNELS = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz',
            'C4','T4','T5','P3','Pz','P4','T6','O1','POz','O2']
CLASSES = ['rest', 'volar_grasp', 'finger_extension']


def write_model(root, name='model_run1', participant='self', training_score=.9, tuning_score=.4):
    directory = root/participant/name
    directory.mkdir(parents=True)
    path = directory/'personal_threeclass.pt'
    # Deliberately not a Torch checkpoint: the metadata layer must never try
    # to unpickle or execute it.
    path.write_bytes(b'opaque test checkpoint '+name.encode())
    contract = dict(schema_version=2, task_mode='multiclass', role='personal_threeclass_model',
        arch='feature_ensemble', classes=CLASSES, participant=participant, channels=CHANNELS,
        sfreq=256, n_times=1024, normalization='feature_train_only',
        filter=dict(mode='causal', fmin=8., fmax=30., order=4), verified=True,
        initialization='scratch', calibration_run_id=name.removeprefix('model_'),
        artifact_limit_v=1e-4, min_signal_std_v=1e-9,
        training_cues=['train1', 'train2', 'train3'], tuning_cues=['tune1', 'tune2', 'tune3'],
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    sidecar = Path(str(path)+'.json')
    sidecar.write_text(json.dumps(contract))
    diagnostics = dict(calibration_only=True, final_validation_performed=False,
        train_cues=contract['training_cues'], tuning_cues=contract['tuning_cues'],
        train_cues_per_class={c:1 for c in CLASSES}, tuning_cues_per_class={c:1 for c in CLASSES},
        config=dict(participant=participant),
        training=dict(cue_trials=dict(balanced_accuracy=training_score)),
        tuning=dict(cue_trials=dict(balanced_accuracy=tuning_score)))
    (directory/'training.json').write_text(json.dumps(diagnostics))
    return path, contract


def mutate_contract(path, **updates):
    sidecar = Path(str(path)+'.json')
    contract = json.loads(sidecar.read_text())
    contract.update(updates)
    sidecar.write_text(json.dumps(contract))


def test_discovery_orders_by_version_time_not_tuning_score_and_never_loads_weights(tmp_path):
    older, _ = write_model(tmp_path, name='model_older', tuning_score=.99)
    newer, _ = write_model(tmp_path, name='model_newer', tuning_score=.35)
    os.utime(older, ns=(100, 100))
    os.utime(newer, ns=(200, 200))
    listed = list_personal_models(tmp_path, 'self', CHANNELS)
    assert [m['model_id'] for m in listed['models']] == [
        'model_newer/personal_threeclass.pt', 'model_older/personal_threeclass.pt']
    assert listed['rejected'] == []
    chosen = listed['models'][0]
    checked = validate_personal_model(chosen['model_id'], 'self', CHANNELS, tmp_path)
    assert checked['checkpoint'] == str(newer.resolve())
    assert checked['training_cues'] == checked['tuning_cues'] == 3
    assert checked['diagnostics']['tuning_cue_balanced_accuracy'] == .35
    # No model deserializer is imported; these opaque bytes would fail Torch.
    assert older.read_bytes().startswith(b'opaque test checkpoint')


@pytest.mark.parametrize('change,match', [
    (dict(participant='other'), 'different participant'),
    (dict(classes=['rest', 'action']), 'three-class'),
    (dict(role='recorded_within_person_reference'), 'personal three-class'),
    (dict(normalization='per_window'), 'preprocessing'),
    (dict(sfreq=128), 'preprocessing'),
    (dict(verified=False), 'provenance'),
    (dict(initialization='pretrained'), 'provenance'),
    (dict(calibration_run_id=''), 'calibration run ID'),
    (dict(training_cues=['train1', 'train1']), 'unique'),
    (dict(tuning_cues=['train1']), 'overlap'),
    (dict(sha256='0'*64), 'SHA256 differs'),
])
def test_wrong_identity_geometry_provenance_or_bytes_are_explicit_rejections(tmp_path, change, match):
    path, _ = write_model(tmp_path)
    mutate_contract(path, **change)
    with pytest.raises(ValueError, match=match):
        validate_personal_model(path, 'self', CHANNELS, tmp_path)
    listed = list_personal_models(tmp_path, 'self', CHANNELS)
    assert not listed['models'] and len(listed['rejected']) == 1
    assert match in listed['rejected'][0]['reason']


def test_physical_montage_order_cannot_be_changed_by_matching_channel_count(tmp_path):
    path, _ = write_model(tmp_path)
    swapped = CHANNELS.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(ValueError, match='sites/order'):
        validate_personal_model(path, 'self', swapped, tmp_path)
    different = CHANNELS.copy(); different[0] = 'F1'
    with pytest.raises(ValueError, match='sites/order'):
        validate_personal_model(path, 'self', different, tmp_path)
    checked = validate_personal_model(path, 'self', [c.lower() for c in CHANNELS], tmp_path)
    assert checked['channels'] == CHANNELS
    with pytest.raises(ValueError, match='unique'):
        list_personal_models(tmp_path, 'self', ['C3']*20)


def test_incomplete_or_corrupt_exports_remain_unloadable_with_reasons(tmp_path):
    path, _ = write_model(tmp_path)
    path.write_bytes(b'changed model bytes')
    incomplete = tmp_path/'self/model_incomplete'
    incomplete.mkdir()
    listed = list_personal_models(tmp_path, 'self', CHANNELS)
    assert not listed['models'] and len(listed['rejected']) == 2
    assert any('missing' in r['reason'] for r in listed['rejected'])
    assert any('SHA256' in r['reason'] for r in listed['rejected'])
    assert list_personal_models(tmp_path, 'new_person', CHANNELS) == dict(models=[], rejected=[])


def test_corrupt_or_mismatched_calibration_diagnostics_cannot_masquerade_as_valid(tmp_path):
    path, _ = write_model(tmp_path)
    diagnostics = path.parent/'training.json'
    diagnostics.write_text('{invalid JSON')
    with pytest.raises(ValueError, match='diagnostics.*valid JSON'):
        validate_personal_model(path, 'self', CHANNELS, tmp_path)
    diagnostics.write_text(json.dumps(dict(calibration_only=True, final_validation_performed=False,
        train_cues=['other1'], tuning_cues=['other2'])))
    with pytest.raises(ValueError, match='disagree'):
        validate_personal_model(path, 'self', CHANNELS, tmp_path)
    diagnostics.unlink()
    # Contract provenance and checkpoint hash suffice if optional diagnostics
    # were intentionally archived elsewhere. No tuning score is invented.
    assert validate_personal_model(path, 'self', CHANNELS, tmp_path)['diagnostics'] is None


def test_paths_cannot_escape_the_selected_person(tmp_path):
    self_path, _ = write_model(tmp_path)
    other_path, _ = write_model(tmp_path, participant='other')
    with pytest.raises(ValueError, match='leaves'):
        validate_personal_model(other_path, 'self', CHANNELS, tmp_path)
    with pytest.raises(ValueError, match='leaves'):
        validate_personal_model('../other/model_run1/personal_threeclass.pt', 'self', CHANNELS, tmp_path)
    for participant in ['../other', 'a/b', '', 'a'*41]:
        with pytest.raises(ValueError, match='pseudonymous'):
            list_personal_models(tmp_path, participant, CHANNELS)
    with pytest.raises(ValueError, match='model_'):
        validate_personal_model(self_path.parent/'renamed.pt', 'self', CHANNELS, tmp_path)
    with pytest.raises(ValueError, match='path cannot be resolved'):
        validate_personal_model(None, 'self', CHANNELS, tmp_path)


def test_symlinked_contract_or_participant_cannot_redirect_model_identity(tmp_path):
    path, _ = write_model(tmp_path)
    external = tmp_path/'outside.json'
    external.write_text(Path(str(path)+'.json').read_text())
    sidecar = Path(str(path)+'.json')
    sidecar.unlink()
    try:
        sidecar.symlink_to(external)
    except (OSError, NotImplementedError):
        pytest.skip('Creating symlinks is unavailable in this Windows environment.')
    with pytest.raises(ValueError, match='contract must stay'):
        validate_personal_model(path, 'self', CHANNELS, tmp_path)
    alias = tmp_path/'alias'
    alias.symlink_to(tmp_path/'self', target_is_directory=True)
    with pytest.raises(ValueError, match='own reports'):
        list_personal_models(tmp_path, 'alias', CHANNELS)
