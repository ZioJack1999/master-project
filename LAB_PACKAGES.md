# Two separate lab packages — Thursday November 5, 2026

Download and extract each ZIP into its own writable folder. Each package has its own model/config files, launcher, calibration and saved-session location. Finish and save one experiment before opening the other; the licensed acquisition software and LSL bridge can remain running if the lab setup supports it. Keep only the intended EEG outlet active for the frozen binary validation; revision-2 three-class acquisition explicitly resolves the configured source ID/name when multiple outlets exist.

| Package | Classes | Setup / launcher | Local page | Calibration / fresh validation |
|---|---|---|---|---|
| [Binary ZIP](downloads/BCI_Binary_Lab_2026-11-05.zip) | Rest, imagined right-hand grasp | `Setup_BCI_Studio.bat` / `Launch_BINARY_Lab.bat` | `http://127.0.0.1:8050` | 18 calibration cues; then 20 NEW validation cues per block |
| [Three-class ZIP · revision 2](downloads/BCI_ThreeClass_Lab_2026-11-05_r2.zip) | Rest, Volar Grasp, Finger Extension | `Setup_THREECLASS_Lab.bat` / `Launch_THREECLASS_Lab.bat` | `http://127.0.0.1:8051` | 36 calibration cues; then 30 NEW validation cues per block |

The binary package preserves the existing workflow and its reference/template weights. The three-class package adds its own live acquisition, cue display, personal covariance-NN fitting and frozen validation interface. Its supplied participant08 model is marked as a recorded reference. The historical 81.6% score does not establish accuracy on a new wearer; fresh personal calibration is required before validation.

The latest [three-class revision](THREECLASS_R2_IMPROVEMENTS.md) confirms each cue after browser rendering, reserves later calibration blocks for tuning, reports confidence rejections, identifies artifact/flat channels across all sites and lets you explicitly reload a verified saved personal model. Earlier ZIPs remain available for reproducibility; use revision 2 for the new three-class checks.

Use 64-bit Python 3.11. Setup downloads Python dependencies. Neither ZIP contains proprietary vendor acquisition/SDK installers or an acquisition-to-LSL bridge; these must be available on the lab PC. Verify the physical montage, reference, 256 Hz stream, channel mapping, units and fresh timestamps before recording. The stock20 and EEGMI custom20 profiles represent different physical electrode arrangements; select the matching profile and keep local settings in the package's `*.local.json` config.

Read each package's `README.md`. Start with binary calibration and two new frozen validation blocks, save all files, close its console/launcher, then test the three-class package separately. The three-class setup uses its own `%USERPROFILE%\.bci_threeclass\venv` environment. Its output is display and recording only; it sends no actuator commands.

During three-class cues keep the display visible. Stop/cancel, a hidden cue tab, display timeout or EEG interruption aborts an incomplete run. Primary cue accuracy includes rejected cues as errors. Retain quality/exclusion counts and model hashes with the EEG/events/predictions. Synthetic demo mode cannot calibrate or validate a personal model.

Both packages passed software checks on this computer. Actual X24/ESU connectivity, installed vendor compatibility and personal live accuracy remain for the lab session.
