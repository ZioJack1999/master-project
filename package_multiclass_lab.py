"""Build standalone 5-, 7- and 9-class X24 lab archives from public source."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from multiclass_lab_bundle import CLASS_SETS, FILES, MANIFEST, ROOT, policy


def build(output, n_classes, root=ROOT):
    root, output = Path(root), Path(output)
    config = json.loads((root/'threeclass_config.json').read_text(encoding='utf-8'))
    del config['checkpoint']
    config.update(classes=CLASS_SETS[n_classes], policy=policy(n_classes),
                  reports_dir=f'session_reports/{n_classes}class')
    files = {}
    for name in FILES:
        if name == 'multiclass_config.json':
            files[name] = (json.dumps(config, indent=2)+'\n').encode()
        else:
            source = 'MULTICLASS_LAB_README.md' if name == 'README.md' else name
            files[name] = (root/source).read_bytes().replace(b'\r\n', b'\n')
        if name.endswith('.bat'):
            files[name] = files[name].replace(b'\n', b'\r\n')
    hashes = {name:hashlib.sha256(data).hexdigest() for name, data in files.items()}
    files[MANIFEST] = (json.dumps(dict(classes=CLASS_SETS[n_classes], sha256=hashes), indent=2)+'\n').encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x') as archive:
        for name, data in files.items():
            entry = ZipInfo(f'BCI_{n_classes}Class_X24_Lab/{name}')
            archive.writestr(entry, data, compress_type=ZIP_DEFLATED, compresslevel=9)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'downloads')
    parser.add_argument('--version', required=True, help='Filename version such as 2026-11-10.')
    args = parser.parse_args()
    for n_classes in CLASS_SETS:
        output = args.output_dir/f'BCI_{n_classes}Class_X24_Lab_{args.version}.zip'
        print(build(output, n_classes))
