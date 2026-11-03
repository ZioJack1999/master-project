# Thursday November 5: binary lab validation

**Primary task: validate Rest versus imagined right-hand grasp on yourself.**
The binary GUI, decoder, training code, settings and reference/template weights are frozen at the November 3 binary release. Multiclass experiments live under `experiments/multiclass_v3/`; they do not start the GUI, change binary weights or send UDP. All accuracy results below refer to their named task; a multiclass score is never binary validation evidence.

## Before going to the lab

- Copy/clone the complete release and install it with `Setup_BCI_Studio.bat`. Test setup on the lab PC before recording if possible. An internet-free setup requires preparing matching Python wheels separately; the repository does not contain vendor installers or all dependencies.
- Run `python verify_binary_release.py`. Expected: `binary_release_unchanged: true`. This verifies code/config contents and exact model bytes, accepting Git's LF/CRLF text conversion. It does not verify hardware/accuracy or freeze the Python environment; `requirements-tested.txt` records the tested versions.
- Bring `README_LAB_GUIDE.md`, this checklist and a writable destination for private session reports. Note vendor software/SDK/driver versions and the actual physical cap/reference arrangement at the lab.
- Budget roughly 60–90 minutes for first connection, signal preparation, calibration and at least two validation blocks. Each default calibration takes 144 seconds plus fitting; each validation block takes 160 seconds. Electrode/bridge troubleshooting can take longer.

## Connect and check

1. Ask the lab operator to establish clean EEG in the existing licensed X24 acquisition software first. Verify the X24/ESU pairing, electrodes/reference and impedance with that software. The studio does not measure impedance.
2. Establish a matching acquisition-to-LSL bridge. The lab guide lists candidate vendor/legacy routes; no live route has been verified here. A 32-bit SDK must run in its own compatible process, separate from the 64-bit studio.
3. Choose the profile matching the actual electrode sites. Default `lab_config.json` is stock20; `lab_config.eegmi_custom20.example.json` is a different physical layout. Make local edits in ignored `lab_config.local.json`, preserving the supplied profile.
4. Run `python lab_preflight.py --config lab_config.local.json --inspect-lsl` with the studio Python. Check the selected stream/source, every required site, 256 Hz, units and timestamps. Metadata compatibility alone does not establish fresh or clean EEG. If using an unchanged default config, omit `--config`.
5. Launch `Launch_BINARY_Lab.bat --config lab_config.local.json`. The wrapper checks the binary release, then calls the existing binary launcher. The same configuration is used for calibration and testing. No multiclass models belong in this config.

## Calibrate and validate

1. Enter a pseudonymous ID, Start Stream and inspect the signal. Rest naturally; during action cues imagine grasping without moving your fingers, jaw or shoulders. Record movement/artifact problems in your notes.
2. Run **Personal calibration**: 18 randomized 8-second cues, training from scratch. Wait for **Personal model ready**. Initial random-template probabilities have no learned meaning. Keep raw calibration EEG and fitting results.
3. Before validation, record model hash, threshold, montage, preprocessing and the intended pass criterion. Example pilot criterion: **at least 70% cue balanced accuracy, both classes detected and permutation p < .05**, with quality/exclusion counts reported. With only 20 cues uncertainty remains substantial; retain results even if they fail.
4. Start Stream → **Validate**: 20 NEW cues, with model/threshold fixed. Do not alter settings after seeing scores and call the same data a new test. Save Session & Stop; keep all EEG, events, predictions and metadata together.
5. Run a second new 20-cue validation block with the same model/settings. Keep the blocks separate and report each; report any combined summary as an additional result. If you recalibrate, start a new model version and use new test cues.
6. If feasible, collect another day/cap placement later. Successful same-day blocks establish a pilot, not across-day reliability or multiclass readiness.

## Keep these tracks separate

| Track | Thursday role | Model/output | Evidence still needed |
|---|---|---|---|
| Binary lab | Primary | Rest/action; existing personal GUI | Physical acquisition and frozen new-cue self-test |
| Multiclass | Separate offline research | 3/5/9 gestures; separate folders/config/results | Better validation, new independent recordings and live GUI integration |
| Cross-person transfer | Exploratory multiclass only | Person-disjoint sources; adaptation labeled separately | More participants and untouched prospective tests |

Inspect saved local files before leaving. Copy reports to private storage; recordings and personal weights are excluded from Git. Check Box upload completion before relying on the cloud copy.

The studio controls a virtual hand only. It implements no FES integration. Establish any Unity/UDP receiver separately with `udp_monitor.py`; do not let transport troubleshooting replace saving the EEG validation.
