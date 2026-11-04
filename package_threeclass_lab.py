"""Build a standalone three-class lab ZIP without binary models or recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from verify_threeclass_bundle import REQUIRED_FILES, REFERENCE_SHA256, verify

ROOT = Path(__file__).resolve().parent


def build(output, root=ROOT):
    root = Path(root).resolve()
    result = verify(root)
    if not result['threeclass_package_valid']:
        raise ValueError('Three-class package is incomplete or changed: '+json.dumps(result['problems']))
    files = {name:(root/name).read_bytes() for name in REQUIRED_FILES}
    files['README.md'] = (root/'THREECLASS_LAB_README.md').read_bytes()
    manifest = dict(schema_version=1, task_mode='threeclass', classes=['rest', 'volar_grasp', 'finger_extension'],
                    reference_sha256=REFERENCE_SHA256,
                    scope='Three-class display/acquisition/personal calibration and frozen validation; no binary models, raw EEG or actuator output.',
                    sha256={name:hashlib.sha256(data).hexdigest() for name, data in files.items()},
                    text_lf_sha256={name:hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest()
                                    for name, data in files.items() if not name.endswith('.pt')})
    files['THREECLASS_BUNDLE_MANIFEST.json'] = (json.dumps(manifest, indent=2)+'\n').encode('utf-8')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Bundle already exists; keep it and choose a new output filename.')
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr('BCI_ThreeClass_Lab/'+name, data)
    info = dict(path=str(output.resolve()), files=len(files),
                sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    print(json.dumps(info, indent=2))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'lab_packages/BCI_ThreeClass_Lab_2026-11-05_r2.zip')
    build(p.parse_args().output)
