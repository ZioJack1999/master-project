"""Verify the separate three-class package without opening EEG or a GUI."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CLASSES = ['rest', 'volar_grasp', 'finger_extension']
REFERENCE = 'decoder_and_models/multiclass_covariance_sub08_v2.pt'
REFERENCE_SHA256 = '0f7ac12e985e2f659cc47d2d258cb998025b7dc1722e9cf48fb11599f0d724fb'
REQUIRED_FILES = [
    'threeclass_studio.py', 'threeclass_personal.py', 'studio.html',
    'threeclass_cues.py', 'threeclass_model_store.py', 'threeclass_decisions.py',
    'threeclass_acquisition.py',
    'threeclass_preflight.py', 'verify_threeclass_bundle.py',
    'threeclass_config.json', 'threeclass_config.custom20.example.json',
    'Launch_THREECLASS_X24.bat', 'run_x24.py', 'x24_lsl_bridge.py',
    'athena_acquisition.py', 'Start_X24_Bridge.bat',
    'multiclass_decoder.py', 'multiclass_features.py', 'multiclass_metrics.py',
    'model_utils.py', 'networks.py', 'temporal_networks.py', 'realtime_bci_decoder.py',
    'create_personal_template.py', 'lsl_acquisition.py', 'requirements.txt',
    REFERENCE, REFERENCE+'.json',
]


def reference_contract(root=ROOT):
    root = Path(root).resolve()
    model = root/REFERENCE
    contract = json.loads(Path(str(model)+'.json').read_text(encoding='utf-8-sig'))
    actual = hashlib.sha256(model.read_bytes()).hexdigest()
    if actual != REFERENCE_SHA256 or contract.get('sha256') != actual:
        raise ValueError('Recorded three-class reference checkpoint hash mismatch.')
    if (contract.get('task_mode') != 'multiclass' or contract.get('schema_version') != 2
            or contract.get('classes') != CLASSES or contract.get('sfreq') != 256
            or contract.get('n_times') != 1024):
        raise ValueError('Expected the three-class, 256 Hz, four-second reference contract.')
    return contract


def verify(root=ROOT):
    root = Path(root).resolve()
    problems, variants = [], []
    manifest_path = root/'THREECLASS_BUNDLE_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8-sig')) if manifest_path.exists() else None
    files = manifest['sha256'] if manifest else {name:None for name in REQUIRED_FILES}
    if manifest:
        for name in REQUIRED_FILES:
            if name not in files:
                problems.append(dict(file=name, issue='absent from bundle manifest'))
    for name, expected in files.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Bundle manifest path leaves its folder.')
        if not path.is_file():
            problems.append(dict(file=name, issue='missing'))
            continue
        if expected:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                lf = manifest.get('text_lf_sha256', {}).get(name)
                if lf and hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() == lf:
                    variants.append(name)
                else:
                    problems.append(dict(file=name, issue='changed'))
    try:
        reference_contract(root)
    except (OSError, ValueError, KeyError) as exc:
        problems.append(dict(file=REFERENCE, issue=str(exc)))
    return dict(threeclass_package_valid=not problems, files_checked=len(files),
                scope='Complete extracted bundle content hashes plus fixed reference checkpoint.' if manifest
                else 'Source checkout: required files and fixed reference checkpoint only; no bundle manifest present.',
                line_ending_variants=variants, problems=problems,
                accuracy_scope='Integrity only. Historical participant08 reference is not personal live validation.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    try:
        result = verify(args.root)
    except (OSError, ValueError, KeyError) as exc:
        result = dict(threeclass_package_valid=False, error=str(exc))
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['threeclass_package_valid'] else 1)


if __name__ == '__main__':
    main()
