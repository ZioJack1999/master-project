# Working X24 connection on this lab PC

Verified November 5, 2026: ESU/X24 â†’ 32-bit mastersp â†’ LSL â†’ existing 64-bit conda Python â†’ original binary GUI.

## Start

Run **Launch_BINARY_X24.bat from this project folder**. It starts the bridge, checks fresh samples and opens http://127.0.0.1:8052. Close other applications that own the ESU. Keep the launch process running; Ctrl+C stops the GUI and bridge. Closing the browser tab alone does not stop acquisition.

The old extracted BCI_binary folder is ignored and is not a supported entry point. Use the launchers beside this document. Historical experiments and distributions are recoverable in Git history and local ignored backups.

Leave Demo Mode OFF. Enter a pseudonymous participant ID, click Start Stream and inspect the signal. Verify physical cap sites/reference and electrode quality. Click Personal calibration (18 Ã— 8s), follow the REST/imagined right-hand grasp cues, and wait for Personal model ready. Start Stream again and Validate for 20 NEW cues. Save Session & Stop afterwards. Initial template probabilities have no learned meaning.

## Verified results

- Athena detects the first-generation X24 10-20 and its ABM ESU.
- SDK labels include all 20 stock20 EEG sites; auxiliary and timestamp fields are excluded by name.
- The 32-bit bridge delivers 20 channels in volts to the 64-bit inlet at a measured 256 Hz. SDK microvolt scaling was cross-checked against vendor EDF physical scaling: near-unit gain after startup.
- The GUI showed HARDWARE CONNECTED and live C3/Cz/C4 waveforms with Demo OFF.
- A live signal test saved 8,098 samples/channel over 31.63 seconds and 119 decoder steps. Median rate: 256 Hz. Maximum timestamp interval: about 3.914 ms. This was a connection test, not an accuracy test.
- All 21 retained binary core/model fingerprints pass verify_binary_release.py; obsolete release files were retired during cleanup without changing retained reference hashes. The unfinished NumPy rewrite was removed; original EEGNet, calibration and validation code are unchanged.
- Six bridge timing/mapping tests pass in both architectures. Fresh-stream preflight and physical stop/restart checks pass.
- A dedicated 65-second bridge run and a subsequent 60-second GUI freshness check passed. The original EEGNet also completed a synthetic software-only fit/save/reload/inference smoke check; no synthetic model was selected in the live GUI.

Signal quality checks sometimes passed and sometimes detected artifacts. Physical placement/reference, clean biological EEG and prospective personal accuracy still require the wearer/lab checks. SDK sample counters are anchored once to packet arrival in the LSL clock; hardware transport and display latency have not been measured.

One earlier restart hit the timestamp freshness guard and stopped the bridge; the later checks above passed with the same guards enabled. If this recurs, save any available session data, stop the launcher and investigate receiver/USB timing before restarting. An interrupted cue run is not a completed validation block.

The shared supervisor also starts the verified three-class GUI via Launch_THREECLASS_X24.bat at http://127.0.0.1:8051. Its physical X24 test displayed live waveforms and three finite class probabilities summing to one on clean updates. All 20 sites were included in quality diagnostics; ten monitored updates passed the software quality check. Saving produced 11,848 samples/channel over 46.28 seconds, 170 prediction updates, median 256 Hz and a maximum timestamp gap of about 3.913 ms. An independent saved-file check confirmed finite volts, live provenance and the correct three class names. The model remained explicitly labeled as the historical participant-08 reference, with personal validation disabled until calibration. This verifies transport and software behavior, not personal classification accuracy.

## Processes and dependencies

| Role | Runtime |
|---|---|
| ESU acquisition | C:\Users\TDT\miniconda3\envs\mastersp\python.exe â€” existing 32-bit Python 3.9.12 |
| Athena SDK | C:\ABM\B-Alert\SDK â€” installed x86 SDK |
| Local LSL sender | .runtime/x24-lsl32 â€” pylsl 1.17.6 with 32-bit liblsl 1.16.2 |
| Decoder/GUI | C:\Users\TDT\miniconda3\python.exe â€” existing 64-bit conda base, PyTorch 2.15.0 CPU |

Neither conda environment was modified. No new conda environment is retained. The small pylsl dependency is installed only in the ignored project-local .runtime folder; PATH/PYTHONPATH overrides affect only the child process. The inherited ABMSDK points to an older Box copy, so the bridge explicitly selects the installed SDK.

To restore the bridge dependency in another checkout on this PC, run from that project folder:

```powershell
& "$env:USERPROFILE\miniconda3\envs\mastersp\python.exe" -m pip install --no-deps --target .runtime/x24-lsl32 pylsl==1.17.6
```

This dependency must include a 32-bit LSL DLL. Existing 64-bit decoder dependency versions differ from requirements-tested.txt; this PC's real transport, GUI and save checks passed, while personal biological performance remains untested.

## Diagnostics and private files

- Launch_BINARY_X24.bat --check-only: verify fresh EEG through both architectures, then stop without the GUI.
- Start_X24_Bridge.bat: bridge alone; stop it before using the combined launcher.
- Check_X24_Mastersp.bat: headset detection; --sample-seconds 5 adds a short SDK transport check.
- session_reports/x24_bridge/: bridge logs, fresh-stream reports, run configs and vendor recordings. Configs select source abm_x24_mastersp and stream BAlert_X24.
- session_reports/session_*: GUI raw EEG, cues, predictions, metadata and plots. Personal weights go under session_reports/personal_models/.

Recordings/logs are ignored by Git. Installed pyEDFlib rejected a short vendor EDF as noncompliant; use the GUI's tested raw NPZ/session metadata as primary saved data for this route. B-Alert Live requests license activation, while direct SDK RAW acquisition succeeds.

The standalone bridge's --diagnostic flag enables native fault tracing. Under tracing, this legacy SDK emits a handled Windows first-chance exception during its stop call; stopping is acknowledged and the process exits successfully. GetRawData storage is copied before the next call and remains SDK-owned: calling FreeBuffer on that particular return buffer corrupts this installed build's heap.
