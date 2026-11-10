# X24 motor-imagery lab

For a new coding session, read [AGENTS.md](AGENTS.md) and [TODO.md](TODO.md) first. They contain the current project rules, evidence, next tasks and end-of-session handoff requirements.

Live motor-imagery apps for the B-Alert X24/ESU. Acquisition runs in **32-bit mastersp** and sends EEG over LSL to the existing **64-bit PyTorch** process. No backend rewrite or new conda environment is needed.

| App | Launch from this folder | Local GUI | Personal calibration / new validation |
|---|---|---|---|
| Rest / imagined right-hand grasp | `Launch_BINARY_X24.bat` | http://127.0.0.1:8052 | 18 calibration cues / 20 new validation cues |
| Rest / Volar Grasp / Finger Extension | `Launch_THREECLASS_X24.bat` | http://127.0.0.1:8051 | 36 calibration cues / 30 new validation cues |

Connect the ESU, power on the X24 and run one launcher. It verifies model integrity, starts the 32-bit bridge, checks fresh labeled 256 Hz EEG and opens the GUI. Keep the launcher running. Closing the browser tab alone does not stop acquisition; Ctrl+C in the launcher stops both processes. Finish and save one experiment before switching apps.

Binary and three-class live transport, waveforms, probabilities and saving have been verified on this PC through the shared bridge. The three-class test saved 11,848 samples per channel at 256 Hz and correctly retained its recorded-reference identity. Both apps require personal calibration and fresh validation before any claim about accuracy on a new wearer.

- [Binary lab procedure](README_LAB_GUIDE.md)
- [Three-class procedure](THREECLASS_LAB_README.md)
- [Hardware/runtime details and measured connection checks](X24_CONNECTION_STATUS.md)

## Expanded personal-validation ZIPs

| Package | Calibration / new validation | Local GUI port |
|---|---|---|
| [5-class ZIP](downloads/BCI_5Class_X24_Lab_2026-11-10.zip) | 60 / 50 cues | 8055 |
| [7-class ZIP](downloads/BCI_7Class_X24_Lab_2026-11-10.zip) | 84 / 70 cues | 8057 |
| [9-class ZIP](downloads/BCI_9Class_X24_Lab_2026-11-10.zip) | 108 / 90 cues | 8059 |

Extract each package to its own writable folder and run `Launch_MULTICLASS_X24.bat` after following the [expanded lab setup and procedure](MULTICLASS_LAB_README.md). Each package starts without weights and fits from personal calibration. The expanded apps have software checks. Live X24 testing and prospective personal accuracy are pending.

## Runtime setup

This PC already has the vendor SDK and the two required interpreters:

- Acquisition: `%USERPROFILE%\miniconda3\envs\mastersp\python.exe` (32-bit Python 3.9).
- Decoder: `%USERPROFILE%\miniconda3\python.exe` (existing 64-bit conda base).
- Vendor SDK: `C:\ABM\B-Alert\SDK` (x86). Vendor software/drivers are licensed and are not in this repository.

The launchers set process-local DLL paths. Neither conda environment is modified. The small x86 LSL dependency lives under ignored `.runtime/x24-lsl32`. For a fresh checkout on this PC, restore it from PowerShell in the project folder:

```powershell
& "$env:USERPROFILE\miniconda3\envs\mastersp\python.exe" -m pip install --no-deps --target .runtime/x24-lsl32 pylsl==1.17.6
```

Install `requirements.txt` only with a compatible **64-bit** decoder Python when preparing another PC. Never install the decoder requirements into 32-bit mastersp. `requirements-tested.txt` preserves historical audit versions; this PC's actual decoder versions are recorded in private preflight reports.

The three-class launcher selects CUDA when available and shows the model device in the GUI. On this PC the launcher uses PyTorch 2.10.0 with CUDA 12.6 from `.runtime/torch-cu126`. To restore that runtime with this PC's 64-bit Python 3.14, run:

```powershell
& "$env:USERPROFILE\miniconda3\python.exe" -m pip install --no-deps --target .runtime/torch-cu126 --index-url https://download.pytorch.org/whl/cu126 torch==2.10.0+cu126
.\Launch_THREECLASS_X24.bat --device cuda
```

`--device cuda` requires GPU availability at startup. Signal filtering and covariance preprocessing run on the CPU, and the neural models run on the selected device.

## Checks

```powershell
.\Check_X24_Mastersp.bat
.\Launch_BINARY_X24.bat --check-only
.\Launch_THREECLASS_X24.bat --check-only
```

Only one bridge should own the ESU. `Start_X24_Bridge.bat` starts the sender alone for troubleshooting; stop it before using a combined launcher.

The retained suite covers binary/three-class inference, calibration provenance, fresh validation, saved-model identity, LSL transport and bridge timing. After installing `requirements-dev.txt` into a development decoder runtime:

```powershell
python -m pytest -q
```

Cleanup validation on this PC: **112 passed, one Windows symlink test skipped**. These are software checks, not biological accuracy measurements.

## Data and repository scope

Private EEG, cues, predictions, plots, personal models and logs stay under ignored `session_reports/`. The three-class reference checkpoint is historical participant data, explicitly labeled as a reference; binary templates are untrained. Keep the complete saved session together.

The active repo contains the live apps, their runtime/model dependencies, the shared X24 bridge, operator docs, tests and the expanded lab ZIPs. The older offline experiments, historical result dumps, unused models and earlier distributions are in Git history. Local recovery copies are under ignored `backups/`.

The binary verifier checks **21 retained original core/model hashes**. Retired files are listed in its manifest; removing an obsolete launcher or historical weight is not presented as changing the retained decoder. The three-class verifier checks runtime closure and its fixed reference checkpoint. Neither check establishes signal quality or personal accuracy.
