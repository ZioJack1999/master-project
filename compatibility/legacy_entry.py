"""Copied beside legacy model files to forward old script paths to the root."""
from pathlib import Path
import runpy
import sys

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
target=Path(__file__).name
if target=='bci_live_gui.py': target='bci_studio_gui.py'
if __name__=='__main__':
    runpy.run_path(str(root/target),run_name='__main__')
else:
    globals().update(runpy.run_path(str(root/target)))
