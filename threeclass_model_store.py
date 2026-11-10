"""Verify and list a wearer's saved multiclass models without loading weights.

The caller explicitly selects a returned model ID, validates it again, and only
then passes the verified checkpoint to the decoder. Filesystem modification
time orders the list; calibration accuracy never chooses a model automatically.
"""
import hashlib
import json
import math
from pathlib import Path
import re


CLASSES = ['rest', 'volar_grasp', 'finger_extension']
def model_identity(classes):
    mode = 'threeclass' if list(classes) == CLASSES else 'multiclass'
    return f'personal_{mode}.pt', f'personal_{mode}_model'


FILTER = dict(mode='causal', fmin=8., fmax=30., order=4)


def _participant_directory(reports_dir, participant):
    if not isinstance(participant, str) or re.fullmatch(r'[A-Za-z0-9_-]{1,40}', participant) is None:
        raise ValueError('Use a short pseudonymous participant ID: letters, digits, underscore or hyphen.')
    try:
        reports = Path(reports_dir).resolve()
        expected = reports/participant
        directory = expected.resolve()
    except (OSError, RuntimeError, TypeError) as exc:
        raise ValueError('Participant model folder cannot be resolved.') from exc
    if directory != expected or not directory.is_relative_to(reports):
        raise ValueError('Participant model folder must stay inside its own reports directory.')
    return directory


def _channels(channels):
    if (not isinstance(channels, (list, tuple)) or len(channels) != 20
            or any(not isinstance(c, str) or not c.strip() or c != c.strip() for c in channels)):
        raise ValueError('Select twenty nonempty physical electrode names.')
    normalized = [c.strip().casefold() for c in channels]
    if len(set(normalized)) != 20:
        raise ValueError('Physical electrode names must be unique.')
    return normalized


def _json_object(path, description):
    try:
        value = json.loads(path.read_text(encoding='utf-8-sig'))
    except (OSError, ValueError, UnicodeError, RecursionError) as exc:
        raise ValueError(f'{description} is missing or is not valid JSON.') from exc
    if not isinstance(value, dict):
        raise ValueError(f'{description} must be a JSON object.')
    return value


def _cue_ids(value, description):
    if (not isinstance(value, list) or not value
            or any(not isinstance(c, str) or not c.strip() for c in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'{description} must contain unique, nonempty cue IDs.')
    return set(value)


def _calibration_score(diagnostics, key):
    result = diagnostics.get(key)
    if result is None:
        return None
    if not isinstance(result, dict) or not isinstance(result.get('cue_trials'), dict):
        raise ValueError('Calibration diagnostics have invalid training/tuning metrics.')
    value = result['cue_trials'].get('balanced_accuracy')
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError('Calibration cue balanced accuracy must be finite and between zero and one.')
    return float(value)


def _diagnostics(path, directory, train, tuning, participant, classes):
    if not path.exists() and not path.is_symlink():
        return None
    try:
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError('Calibration diagnostics path cannot be resolved.') from exc
    if resolved.parent != directory:
        raise ValueError('Calibration diagnostics must stay in the selected model folder.')
    data = _json_object(resolved, 'Calibration diagnostics')
    if data.get('calibration_only') is not True or data.get('final_validation_performed') is not False:
        raise ValueError('Expected calibration-only diagnostics, separate from fresh validation.')
    if _cue_ids(data.get('train_cues'), 'Diagnostic training cues') != train:
        raise ValueError('Diagnostic training cues disagree with the model contract.')
    if _cue_ids(data.get('tuning_cues'), 'Diagnostic tuning cues') != tuning:
        raise ValueError('Diagnostic tuning cues disagree with the model contract.')
    config = data.get('config', {})
    if not isinstance(config, dict) or config.get('participant', participant) != participant:
        raise ValueError('Calibration diagnostics belong to a different participant.')
    for key, expected in [('train_cues_per_class', len(train)), ('tuning_cues_per_class', len(tuning))]:
        counts = data.get(key)
        if counts is not None:
            if (not isinstance(counts, dict) or set(counts) != set(classes)
                    or any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts.values())
                    or sum(counts.values()) != expected):
                raise ValueError('Calibration per-class cue counts disagree with the model contract.')
    return dict(training_cue_balanced_accuracy=_calibration_score(data, 'training'),
                tuning_cue_balanced_accuracy=_calibration_score(data, 'tuning'),
                scope='Calibration training/tuning only; not fresh validation accuracy.')


def validate_personal_model(path, participant, channels, reports_dir, classes=CLASSES):
    """Verify one model and return metadata; never deserialize checkpoint bytes.

    ``path`` is either an absolute checkpoint or a ``model_id`` returned by
    :func:`list_personal_models`, relative to ``reports_dir/participant``.
    Case-insensitive electrode spelling is accepted; order and sites must match.
    """
    directory = _participant_directory(reports_dir, participant)
    expected_channels = _channels(channels)
    try:
        supplied = Path(path)
        checkpoint = (supplied if supplied.is_absolute() else directory/supplied).resolve()
    except (OSError, RuntimeError, TypeError) as exc:
        raise ValueError('Personal model path cannot be resolved.') from exc
    if not checkpoint.is_relative_to(directory):
        raise ValueError('Personal model path leaves the selected participant folder.')
    filename, role = model_identity(classes)
    relative = checkpoint.relative_to(directory)
    if (len(relative.parts) != 2 or not relative.parts[0].startswith('model_')
            or relative.parts[1] != filename):
        raise ValueError(f'Select a saved model_*/{filename} checkpoint.')
    if not checkpoint.is_file():
        raise ValueError('Personal checkpoint file is missing.')
    contract_path = Path(str(checkpoint)+'.json')
    try:
        resolved_contract = contract_path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError('Personal model contract cannot be resolved.') from exc
    if resolved_contract.parent != checkpoint.parent:
        raise ValueError('Personal model contract must stay in the selected model folder.')
    contract = _json_object(resolved_contract, 'Personal model contract')
    if (contract.get('schema_version') != 2 or contract.get('task_mode') != 'multiclass'
            or contract.get('role') != role
            or contract.get('arch') != 'feature_ensemble'
            or contract.get('classes') != list(classes)):
        task = 'three-class' if list(classes) == CLASSES else f'{len(classes)}-class'
        raise ValueError(f'Expected a personal {task} feature-ensemble model contract for the selected classes.')
    if contract.get('participant') != participant:
        raise ValueError('Personal model belongs to a different participant.')
    if _channels(contract.get('channels')) != expected_channels:
        raise ValueError('Personal model electrode sites/order differ from the selected physical montage.')
    if (contract.get('sfreq') != 256 or contract.get('n_times') != 1024
            or contract.get('normalization') != 'feature_train_only' or contract.get('filter') != FILTER):
        raise ValueError('Personal model preprocessing differs from the supported 256 Hz, four-second pipeline.')
    if contract.get('verified') is not True or contract.get('initialization') != 'scratch':
        raise ValueError('Personal model lacks documented scratch calibration provenance.')
    run = contract.get('calibration_run_id')
    if not isinstance(run, str) or not run.strip():
        raise ValueError('Personal model lacks its calibration run ID.')
    train = _cue_ids(contract.get('training_cues'), 'Training cues')
    tuning = _cue_ids(contract.get('tuning_cues'), 'Tuning cues')
    if train & tuning:
        raise ValueError('Personal model training and tuning cues overlap.')
    for key in ['artifact_limit_v', 'min_signal_std_v']:
        value = contract.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('Personal model has invalid signal-quality limits.')
    expected_hash = contract.get('sha256')
    if not isinstance(expected_hash, str) or re.fullmatch(r'[0-9a-f]{64}', expected_hash) is None:
        raise ValueError('Personal model contract lacks a valid checkpoint SHA256.')
    try:
        with checkpoint.open('rb') as handle:
            actual_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        stat = checkpoint.stat()
    except OSError as exc:
        raise ValueError('Personal checkpoint cannot be read.') from exc
    if actual_hash != expected_hash:
        raise ValueError('Personal checkpoint SHA256 differs from its contract.')
    diagnostics = _diagnostics(checkpoint.parent/'training.json', checkpoint.parent,
                               train, tuning, participant, classes)
    model_id = relative.as_posix()
    return dict(model_id=model_id, checkpoint=str(checkpoint), contract=str(resolved_contract),
                participant=participant, channels=contract['channels'], classes=list(classes),
                sha256=actual_hash, calibration_run_id=run, training_cues=len(train),
                tuning_cues=len(tuning), modified_time_ns=stat.st_mtime_ns,
                bytes=stat.st_size, diagnostics=diagnostics,
                scope='Saved personal calibration model. Collect new frozen-model validation cues.')


def list_personal_models(reports_dir, participant, channels, classes=CLASSES):
    """List compatible versions newest-first, retaining explicit rejection reasons."""
    directory = _participant_directory(reports_dir, participant)
    _channels(channels)
    models, rejected, seen = [], [], set()
    if not directory.exists():
        return dict(models=models, rejected=rejected)
    if not directory.is_dir():
        raise ValueError('Participant model location is not a directory.')
    filename, _ = model_identity(classes)
    for folder in sorted(directory.glob('model_*')):
        checkpoint = folder/filename
        try:
            candidate = validate_personal_model(checkpoint, participant, channels, reports_dir, classes)
            if candidate['checkpoint'] not in seen:
                models.append(candidate)
                seen.add(candidate['checkpoint'])
        except (OSError, ValueError, RuntimeError) as exc:
            rejected.append(dict(model_id=(Path(folder.name)/filename).as_posix(), reason=str(exc)))
    models.sort(key=lambda candidate:(-candidate['modified_time_ns'], candidate['model_id']))
    return dict(models=models, rejected=rejected)
