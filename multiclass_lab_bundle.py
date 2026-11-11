"""Verify the expanded live lab distribution without starting acquisition."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FIVE = ['rest', 'volar_grasp', 'finger_extension', 'wrist_flexion', 'wrist_extension']
CLASS_SETS = {
    5: FIVE,
    7: FIVE + ['wrist_pronation', 'wrist_supination'],
    9: FIVE[:3] + ['opponens_grasp'] + FIVE[3:] + ['wrist_pronation', 'wrist_supination', 'pinch_grasp'],
}
MANIFEST = 'MULTICLASS_BUNDLE_MANIFEST.json'
FILES = (
    'README.md', 'Launch_MULTICLASS_X24.bat', 'requirements.txt',
    'multiclass_config.json', 'multiclass_lab_bundle.py', 'run_x24.py',
    'x24_lsl_bridge.py', 'athena_acquisition.py', 'lsl_acquisition.py',
    'threeclass_studio.py', 'studio.html', 'threeclass_acquisition.py',
    'threeclass_cues.py', 'threeclass_decisions.py', 'threeclass_model_store.py',
    'threeclass_personal.py', 'multiclass_decoder.py', 'multiclass_features.py',
    'multiclass_metrics.py', 'model_utils.py', 'networks.py', 'realtime_bci_decoder.py',
    'temporal_networks.py', 'dual_band_filter.py',
    'virtual_hand.html', 'static/js/three.min.js', 'static/js/virtual_hand.js',
)


def policy(n_classes):
    return dict(ema_alpha=1., dwell_updates=1, min_confidence=1.5/n_classes, min_margin=.05)


def expanded_config(base, n_classes):
    config = dict(base, classes=CLASS_SETS[n_classes], policy=policy(n_classes),
                  reports_dir=f'session_reports/{n_classes}class')
    config.pop('checkpoint', None)
    return config


def verify_bundle(root=ROOT):
    root = Path(root).resolve()
    hashes = json.loads((root/MANIFEST).read_text(encoding='utf-8'))['sha256']
    if set(hashes) != set(FILES):
        raise ValueError('The multiclass manifest does not match the required lab files.')
    for name, expected in hashes.items():
        data = (root/name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Multiclass lab file changed: {name}. Extract a fresh copy.')
    config = json.loads((root/'multiclass_config.json').read_text(encoding='utf-8'))
    count = len(config['classes'])
    if (config['classes'] != CLASS_SETS.get(count) or 'checkpoint' in config
            or config['reports_dir'] != f'session_reports/{count}class'
            or config['policy'] != policy(count)):
        raise ValueError('Invalid expanded lab class profile or policy.')
    return config
