"""Check three-class config, model metadata and optional fresh LSL samples."""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from verify_threeclass_bundle import CLASSES

ROOT = Path(__file__).resolve().parent


def inspect(config, root=ROOT, inspect_lsl=False):
    root = Path(root).resolve()
    result = dict(configuration_valid=False, acquisition_ready=False,
                  reference_compatible=False, personal_calibration_required=True,
                  packages={}, streams=[], errors=[],
                  readiness_scope='Configuration/model metadata only until an explicit fresh LSL probe succeeds.')
    for package in ['numpy', 'scipy', 'torch', 'pandas', 'matplotlib', 'mne', 'scikit-learn', 'pylsl']:
        try:
            result['packages'][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result['packages'][package] = 'MISSING'
    try:
        channels = config['channels']
        if (not isinstance(channels, list) or len(channels) != 20
                or any(not isinstance(c, str) or not c.strip() for c in channels)
                or len({c.strip().casefold() for c in channels}) != 20):
            raise ValueError('Choose 20 unique, verified physical EEG sites in model order.')
        geometry = [float(config.get(k, d)) for k, d in
                    [('sfreq', 256), ('window_s', 4.), ('stride_s', .25), ('settle_s', .5)]]
        if (not all(math.isfinite(v) for v in geometry) or geometry[0] != 256
                or geometry[1] != 4. or geometry[2] != .25 or geometry[3] < .5):
            raise ValueError('Three-class acquisition requires 256 Hz, 4-second windows, .25-second stride and at least .5-second settling.')
        for key, default in [('max_sample_age_s', 1.), ('max_gap_s', .1)]:
            value = float(config.get(key, default))
            if not math.isfinite(value) or value <= 0:
                raise ValueError(key+' must be finite and positive.')
        model = (root/config['checkpoint']).resolve()
        contract = json.loads(Path(str(model)+'.json').read_text(encoding='utf-8-sig'))
        actual = hashlib.sha256(model.read_bytes()).hexdigest()
        if contract.get('sha256') != actual:
            raise ValueError('Selected three-class checkpoint hash differs from its contract.')
        if (contract.get('task_mode') != 'multiclass' or contract.get('schema_version') != 2
                or contract.get('classes') != CLASSES or contract.get('sfreq') != 256
                or contract.get('n_times') != 1024
                or contract.get('filter') != dict(mode='causal', fmin=8., fmax=30., order=4)):
            raise ValueError('Selected model must use the three declared classes and supported causal preprocessing.')
        result['reference_compatible'] = [c.casefold() for c in channels] == [c.casefold() for c in contract['channels']]
        result['model'] = dict(path=str(model), sha256=actual, classes=CLASSES,
                              role=contract.get('role', 'unspecified'), channels=contract['channels'],
                              historical_accuracy_only=contract.get('role') == 'recorded_within_person_reference')
        result['acquisition_contract'] = dict(channels=channels, sfreq=256, n_times=1024)
        result['configuration_valid'] = True
        result['initial_model_status'] = ('recorded_reference_for_display_personal_calibration_required'
            if result['reference_compatible'] else
            'initial_reference_incompatible_personal_calibration_required')
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result['errors'].append(str(exc))
        return result
    if inspect_lsl:
        try:
            from pylsl import StreamInlet, local_clock, proc_clocksync
            from lsl_acquisition import StreamAdapter, stream_metadata
            from threeclass_acquisition import discover_eeg_streams
            streams = discover_eeg_streams(config)
            selected = [s for s in streams
                        if (not config.get('stream_name') or s.name() == config['stream_name'])
                        and (not config.get('source_id') or s.source_id() == config['source_id'])]
            if len(selected) != 1:
                result['errors'].append(f'Expected one selected EEG stream; found {len(selected)}. Set stream_name/source_id.')
            for stream in streams:
                chosen = len(selected) == 1 and stream == selected[0]
                inlet = StreamInlet(stream, max_buflen=1, processing_flags=proc_clocksync, recover=False)
                try:
                    meta = stream_metadata(inlet.info(timeout=2.))
                    item = {k:v for k, v in meta.items() if k != 'xml'}
                    item.update(selected=chosen, compatible_metadata=False, fresh_samples_checked=False)
                    try:
                        adapter = StreamAdapter(meta, result['acquisition_contract'], config)
                        item['compatible_metadata'] = True
                        if chosen:
                            inlet.open_stream(timeout=2.)
                            chunk, timestamps = inlet.pull_chunk(timeout=1., max_samples=64)
                            if not timestamps:
                                raise ValueError('Selected stream produced no fresh samples during the probe.')
                            transformed, ts = adapter.transform(chunk, timestamps, local_clock())
                            item.update(fresh_samples_checked=True, fresh_samples=int(transformed.shape[1]),
                                        sample_age_s=float(local_clock()-ts[-1]))
                            result['acquisition_ready'] = True
                    except Exception as exc:
                        item['error'] = str(exc)
                    result['streams'].append(item)
                finally:
                    inlet.close_stream()
        except Exception as exc:
            result['errors'].append('LSL inspection: '+str(exc))
    result['readiness_scope'] = ('Configuration/model and a short selected-stream metadata/timestamp/unit probe only. '
                                 'Fresh samples do not establish clean EEG or accuracy; personal calibration and new-cue validation are required.')
    if not inspect_lsl:
        result['readiness_scope'] = ('Configuration and checkpoint metadata only; no stream or fresh EEG was inspected. '
                                     'Personal calibration and new-cue validation are required.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT/'threeclass_config.json')
    p.add_argument('--inspect-lsl', action='store_true')
    args = p.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding='utf-8-sig'))
        result = inspect(config, ROOT, args.inspect_lsl)
    except (OSError, ValueError, KeyError) as exc:
        result = dict(configuration_valid=False, errors=[str(exc)])
    print(json.dumps(result, indent=2))
    ok = result['configuration_valid'] and (not args.inspect_lsl or result.get('acquisition_ready', False))
    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
