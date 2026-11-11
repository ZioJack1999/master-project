# Current project handoff

Last updated: November 11, 2026 (America/New_York).

## Objective

Complete personal motor-imagery validation before November 26, 2026. Target at least 70% balanced accuracy on new cues, with leakage, overfitting and underfitting checks. The user reports the system accepts binary and is trying three-class. Prospective personal accuracy remains unreported.

## Current baseline

Reviewed GitHub default branch `master` through `93ce5d0`, including the 3D hand, the dedicated three-class live launcher and the synthetic demo controls.

- Direct API acquisition is implemented: X24/ESU -> x86 Athena SDK in existing 32-bit `mastersp` -> LSL -> existing 64-bit decoder. A separate licensed acquisition GUI is not required for this route.
- `Launch_BINARY_X24.bat`: binary GUI at http://127.0.0.1:8052. Personal calibration 18 x 8 seconds; validation 20 NEW cues.
- `Launch_THREECLASS_X24.bat`: three-class GUI at http://127.0.0.1:8051. Personal calibration 36 x 8 seconds; validation 30 NEW cues.
- `Launch_THREECLASS_Studio.bat`: three-class dashboard and 3D hand on port 8051 with Demo off. Use **Connect EEG** with an available compatible LSL source. The combined X24 launcher above starts and supervises the bridge.
- `Launch_BINARY_Demo.bat` and `Launch_THREECLASS_Demo.bat` open the dashboard and viewer in synthetic mode. Click **Start Stream** to begin.
- The three [expanded ZIPs](README.md#expanded-personal-validation-zips) each use `Launch_MULTICLASS_X24.bat` in their own extracted folder. Calibration/validation counts are 60/50, 84/70 and 108/90 cues for 5, 7 and 9 classes. The 7-class set adds wrist pronation and supination to the historical 5-class set. The historical 9-class set adds opponens and pinch grasp. Use the lab's installed x86 SDK and separate 32-bit LSL runtime as described in the package README.
- Run one combined launcher at a time. Leave Demo OFF. Enter a pseudonymous participant ID, Start Stream, inspect quality, calibrate, then validate on fresh cues. Save before stopping. Ctrl+C in the launcher stops acquisition and GUI.
- Earlier ZIPs, launchers and offline research remain in Git history and ignored local backups. Use the current launchers and expanded ZIPs above.
- Existing `RESUME_NOTES.md` describes November 4 and is superseded for current operation by this file, `README.md` and `X24_CONNECTION_STATUS.md`.

## Evidence and limits

The November 5 committed `X24_CONNECTION_STATUS.md` reports real binary and three-class hardware transport, GUI traces/probabilities and saved-file checks on the lab PC. Binary saved 8,098 samples/channel; three-class saved 11,848, at measured 256 Hz. This November 6 session reviewed those records and source code; it did not repeat hardware tests or inspect the private lab recordings.

The three-class checkpoint remained explicitly the historical participant-08 reference with personal validation blocked until calibration. Binary templates are untrained. Transport success is not personal classification accuracy. Electrode placement/reference, biological signal quality and prospective performance still require checking. SDK timestamps are anchored to packet arrival; transport/display latency is unmeasured. A previous freshness-stop incident is documented; do not disable guards to avoid it.

The November 5 lab cleanup recorded 112 passing tests, one Windows symlink skip and six bridge tests in both architectures. The November 11 software checks are below. Historical study test sets have been exposed; do not reuse them for fresh confirmation.

## Next actions, in order

- [x] Search the accessible source, Box EEGMI_Personal and two Desktop distribution roots for saved GUI-session metadata. November 6: only two old November 3 three-class session records were found, both with empty results and no cue events; one is demo, the other marked live but still a recorded reference. Neither is completed personal validation. No November 5 lab recording was found in these roots.
- [ ] Obtain the lab PC's complete relevant `session_reports/` subfolders (raw NPZ, metadata, cue events, predictions, personal checkpoint/contract/training diagnostics and bridge preflight/logs). Preserve their relationships and privacy. Local historical EEGMI datasets are not substitutes for these new lab sessions.
- [ ] Complete the three-class lab trial. Verify montage/reference and clean signal, fit a personal model, freeze it, collect a fresh validation block and save the complete session. Repeat independently in each expanded package, one X24 bridge at a time. If binary acceptance was a transport or format check, collect a separate fresh binary validation block as needed.
- [ ] Analyze completed validation: cue-level balanced accuracy, per-class recall, confusion matrix including rejected cues, accepted coverage, timing gaps, artifacts and train/tuning gaps. State sample sizes and uncertainty. Keep binary and three-class results separate.
- [ ] If performance is below target, investigate labels/cue timing, physical quality, units/montage, rejected windows and calibration adequacy before changing the NN. Select changes using calibration training/tuning only; reserve new validation data for confirmation.
- [ ] Version, test and document any necessary changes; update current operator instructions and this handoff. Publish only code/non-identifying evidence; keep recordings and personal models private.

Cross-subject generalization remains deferred.

## November 11 setup review

Confirmed the live work launcher and both demo launchers on `origin/master`. The viewer identifies synthetic decisions and leaves confidence blank while the stream is idle. The README describes the live launch and the manual demo start.

The full suite at `267f84a` passed 190 tests, with two skips for unavailable CUDA and Windows symlink creation. After the viewer follow-up, all 16 targeted viewer and studio tests passed. The binary verifier passed 19 retained hashes and the three-class verifier checked 28 required files including the fixed reference. Batch smoke checks confirmed the live studio opened with Demo off, disconnected and idle. Both demos opened idle, then produced finite probabilities after Start Stream. Zen rendered the hand in a WebGL framebuffer check and showed the synthetic source and blank idle confidence. The lab X24 and wearer accuracy still require the live trial.

GitHub delivery of the setup was verified through `93ce5d0`. The Desktop live launcher matches Git, and the three launchers and viewer HTML were copied and checksum-verified in local Box `EEGMI_Personal`. Box cloud sync is unconfirmed. All temporary browser and studio processes used for testing exited.

Next action: use the combined three-class X24 launcher on the lab PC, verify the incoming signal and complete personal calibration and the fresh validation block.

## November 10 session

Prepared the experimental 5-, 7- and 9-class X24 ZIPs for GitHub `master` in the commit titled `Package 5-, 7- and 9-class X24 validation labs`. Each starts without a checkpoint, fits from fresh personal calibration, reserves later whole cue blocks for tuning, and saves a separate session with frozen-model validation, rejection-aware confusion matrix, and per-class recall. The confidence floor is fixed at 1.5 divided by the class count with a 0.05 runner-up margin. There are no EEG recordings, private models or vendor SDK files in the ZIPs.

Software validation covered repeatable ZIP generation, hash verification through all three extracted batch launchers, and synthetic cue collection, actual fitting, saved-model reload, frozen validation and saving for all three profiles. Full Python suite: 118 passed and one Windows symlink test skipped. Binary and three-class integrity checks passed 21 and 27 files respectively. Hardware and wearer accuracy for the expanded sets still require a live lab session.

Delivery includes the three ZIPs in `downloads/` and checksum-verified copies in the local Box and Desktop download folders. Box cloud sync is unconfirmed. No GUI, bridge, training or acquisition process was left running. The next action is to finish three-class validation, then extract and test each expanded package with the lab X24.

## November 6 session

Request: "If there's something you can do, do it." Continued the new handoff priorities after documentation commit `89a5074`.

Completed: fetched origin (no newer changes), searched accessible saved-session metadata including ignored session folders, and established the missing lab-data boundary above. Fixed the X24 supervisor duplicate-outlet discovery: `minimum=1` could return before observing a second matching source; `minimum=2` with the existing one-second timeout permits the ambiguity check to run. Added regression tests for duplicate refusal before opening an inlet and failure before discovery when the child bridge has exited. No model/core changes, environment modifications, training or hardware acquisition were performed.

Validation performed here: full suite 112 passed / one Windows symlink skip in 65.02 seconds (collection began before adding the two supervisor tests); targeted bridge/supervisor run 8 passed, including the two new regression cases. Binary verifier passed all 21 retained fingerprints; three-class verifier passed all 27 required runtime/reference files. No physical repeat test was performed. An accidentally duplicated full-suite process was stopped; one full run completed. No need to repeat unrelated fitting studies.

Delivery: previous documentation commit `89a5074` is pushed. The current change is committed under `Reject duplicate X24 outlets during supervisor preflight` and pushed to origin/master; consult Git history/final response if interrupted. Current handoff and a separate `x24_supervisor_update/` copy of the changed supervisor/test and application instructions are delivered locally to Box and both Desktop distributions. Their older complete runtimes are not upgraded by this incremental delivery; use the current Git checkout to run the supported X24 launchers. Box cloud upload is unconfirmed.

Processes: this session started no GUI, bridge, training or acquisition processes.

Next concrete action: obtain the complete lab-PC session files listed above, then establish whether personal calibration plus a complete unseen validation block has actually been performed. If none exists, collect it in the lab using the current launchers before claiming accuracy.
