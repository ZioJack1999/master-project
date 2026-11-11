"""Launch the BCI dashboards, X24 supervisors, and standalone hand viewer."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
MODES = {
    'x24': ('BCI Studio with X24 bridge', 'Launch_THREECLASS_X24.bat', ('--profile-selector',)),
    'studio': ('BCI Studio (existing LSL source)', 'unified_studio.py', ()),
    'demo': ('Synthetic demo', 'unified_studio.py', ('--demo',)),
    'hand': ('3D hand viewer', 'virtual_hand_viewer.py', ()),
}


def select_mode():
    entries = [
        (mode, label) for mode, (label, script, _) in MODES.items()
        if os.name == 'nt' or script.endswith('.py')
    ]
    print('BCI Studio\n')
    choices = {}
    for index, (mode, label) in enumerate(entries, 1):
        choices[str(index)] = mode
        print(f'  {index}. {label}')
    print('  0. Exit')
    while True:
        selection = input('\nSelect a mode: ').strip()
        if selection == '0':
            return None
        if selection in choices:
            return choices[selection]
        print('Enter a number from the menu.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    direct_modes = [mode for mode, (_, script, _) in MODES.items() if script.endswith('.py')]
    parser.add_argument('mode', nargs='?', choices=direct_modes, help='Omit to open the launch menu.')
    parser.add_argument('options', nargs=argparse.REMAINDER, help='Arguments forwarded to the selected app.')
    args = parser.parse_args()
    try:
        mode = args.mode if args.mode else select_mode()
        if mode is None:
            return 0
        _, script, defaults = MODES[mode]
        if script.endswith('.bat'):
            return subprocess.call(['cmd.exe', '/d', '/c', script, *defaults], cwd=ROOT)
        return subprocess.call([sys.executable, script, *defaults, *args.options], cwd=ROOT)
    except (EOFError, KeyboardInterrupt):
        return 0
    except OSError as exc:
        parser.exit(1, f'Could not start BCI Studio: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
