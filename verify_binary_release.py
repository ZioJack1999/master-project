"""Check the frozen binary files without starting acquisition or training."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify(root=ROOT, manifest=None):
    root = Path(root).resolve()
    manifest = manifest or json.loads((root/'binary_release_manifest.json').read_text())
    problems = []
    line_ending_variants = []
    for name, expected in manifest['sha256'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('Manifest path leaves release folder.')
        if not path.is_file():
            problems.append(dict(file=name, issue='missing'))
        else:
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                canonical = manifest.get('text_lf_sha256', {}).get(name)
                if canonical and hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() == canonical:
                    line_ending_variants.append(name)
                else:
                    problems.append(dict(file=name, issue='changed'))
    # The selected model contract must also remain explicitly binary.
    config = json.loads((root/'lab_config.json').read_text(encoding='utf-8-sig'))
    checkpoint = (root/config['checkpoint']).resolve()
    if not checkpoint.is_relative_to(root):
        raise ValueError('Default checkpoint leaves release folder.')
    contract = json.loads(Path(str(checkpoint)+'.json').read_text())
    if contract['classes'] != ['rest', 'action']:
        problems.append(dict(file=config['checkpoint'], issue='not binary'))
    return dict(binary_release_unchanged=not problems, files_checked=len(manifest['sha256']),
                reference_commit=manifest['reference_commit'], problems=problems,
                line_ending_variants=line_ending_variants,
                scope='Content integrity; text LF/CRLF equivalents accepted, model bytes exact. Live X24 acquisition and new-cue accuracy require lab validation.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=ROOT)
    args = p.parse_args()
    result = verify(args.root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['binary_release_unchanged'] else 1)


if __name__ == '__main__':
    main()
