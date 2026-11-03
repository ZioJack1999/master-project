"""Build a binary-only lab ZIP, excluding research models and all recordings."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from verify_binary_release import verify

ROOT = Path(__file__).resolve().parent


def build(output):
    result = verify(ROOT)
    if not result['binary_release_unchanged']:
        raise ValueError('Binary content differs from the frozen release.')
    manifest = json.loads((ROOT/'binary_release_manifest.json').read_text())
    files = {name:(ROOT/name).read_bytes() for name in manifest['sha256']}
    for name in ['verify_binary_release.py', 'binary_release_manifest.json',
                 'Launch_BINARY_Lab.bat', 'README_LAB_GUIDE.md', 'THURSDAY_BINARY_VALIDATION.md']:
        files[name] = (ROOT/name).read_bytes()
    files['README.md'] = (ROOT/'BINARY_LAB_README.md').read_bytes()
    info = dict(binary_reference_commit=manifest['reference_commit'],
                scope='Binary lab application only; no multiclass research, raw EEG or personal weights.',
                sha256={name:hashlib.sha256(data).hexdigest() for name, data in files.items()})
    files['BINARY_BUNDLE_MANIFEST.json'] = (json.dumps(info, indent=2)+'\n').encode()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Bundle already exists; retain it and choose a new output filename.')
    with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr('BCI_Binary_Lab/'+name, data)
    print(json.dumps(dict(path=str(output.resolve()), files=len(files),
                         sha256=hashlib.sha256(output.read_bytes()).hexdigest()), indent=2))
    return output


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'lab_packages/BCI_Binary_Lab_2026-11-05.zip')
    build(p.parse_args().output)
