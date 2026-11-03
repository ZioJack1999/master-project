# Binary lab package — Thursday November 5, 2026

This package contains only the **Rest versus imagined right-hand grasp** application, its binary model/template references, configs and lab instructions. Multiclass experiments and recordings are excluded. The binary software and weights preserve the November 3 reference; live X24 acquisition and new-cue accuracy still need lab validation.

1. Extract the complete ZIP into a writable folder on the lab PC. Do not run inside the ZIP.
2. Run `Setup_BCI_Studio.bat` with 64-bit Python 3.11 installed. Setup downloads Python dependencies; licensed vendor acquisition/SDK and the LSL bridge are separate lab prerequisites.
3. Follow [Thursday's checklist](THURSDAY_BINARY_VALIDATION.md) and [lab procedure](README_LAB_GUIDE.md). Match the physical cap, reference, stream labels, units and 256 Hz rate. Put local settings in `lab_config.local.json`.
4. Run `Launch_BINARY_Lab.bat --config lab_config.local.json`, or omit `--config` if using the unchanged stock20 defaults. The launcher checks the binary files and starts the existing studio.
5. Collect personal calibration, then two separate new validation blocks with frozen model/settings. Save all private session files and results before leaving.

`verify_binary_release.py` checks 36 protected files, including the GUI HTML, allowing text line-ending conversion while requiring exact model bytes. It does not verify hardware, clean EEG, the Python environment or accuracy. Templates have random weights until personal calibration finishes.

Private reports save into `session_reports`. This package implements virtual-hand output only, with no FES integration. Keep the working lab's acquisition installation intact. More-class results/code remain available separately in the [full project](https://github.com/ZioJack1999/master-project).
