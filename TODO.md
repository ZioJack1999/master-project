# Current project handoff

Last updated: November 10, 2026 (America/New_York).

## Objective

Complete personal motor-imagery validation before November 26, 2026. Target at least 70% balanced accuracy on new cues, with leakage, overfitting and underfitting checks. The user reports the system accepts binary and is trying three-class. Prospective personal accuracy remains unreported.

## Current baseline

Fetched and reviewed GitHub `master` through `481a908`, including the verified hardware-transport handoff and duplicate-outlet guard.

- Direct API acquisition is implemented: X24/ESU -> x86 Athena SDK in existing 32-bit `mastersp` -> LSL -> existing 64-bit decoder. A separate licensed acquisition GUI is not required for this route.
- `Launch_BINARY_X24.bat`: binary GUI at http://127.0.0.1:8052. Personal calibration 18 x 8 seconds; validation 20 NEW cues.
- `Launch_THREECLASS_X24.bat`: three-class GUI at http://127.0.0.1:8051. Personal calibration 36 x 8 seconds; validation 30 NEW cues.
- The three [expanded ZIPs](README.md#expanded-personal-validation-zips) each use `Launch_MULTICLASS_X24.bat` in their own extracted folder. Calibration/validation counts are 60/50, 84/70 and 108/90 cues for 5, 7 and 9 classes. The 7-class set adds wrist pronation and supination to the historical 5-class set. The historical 9-class set adds opponens and pinch grasp. Use the lab's installed x86 SDK and separate 32-bit LSL runtime as described in the package README.
- Run one combined launcher at a time. Leave Demo OFF. Enter a pseudonymous participant ID, Start Stream, inspect quality, calibrate, then validate on fresh cues. Save before stopping. Ctrl+C in the launcher stops acquisition and GUI.
- Earlier ZIPs, launchers and offline research remain in Git history and ignored local backups. Use the current launchers and expanded ZIPs above.
- Existing `RESUME_NOTES.md` describes November 4 and is superseded for current operation by this file, `README.md` and `X24_CONNECTION_STATUS.md`.

## Evidence and limits

The November 5 committed `X24_CONNECTION_STATUS.md` reports real binary and three-class hardware transport, GUI traces/probabilities and saved-file checks on the lab PC. Binary saved 8,098 samples/channel; three-class saved 11,848, at measured 256 Hz. This November 6 session reviewed those records and source code; it did not repeat hardware tests or inspect the private lab recordings.

The three-class checkpoint remained explicitly the historical participant-08 reference with personal validation blocked until calibration. Binary templates are untrained. Transport success is not personal classification accuracy. Electrode placement/reference, biological signal quality and prospective performance still require checking. SDK timestamps are anchored to packet arrival; transport/display latency is unmeasured. A previous freshness-stop incident is documented; do not disable guards to avoid it.

Repository README records 112 passing tests and one Windows symlink skip from the lab cleanup session, plus six bridge tests in both architectures. These are prior-session results, not checks rerun today. Historical study test sets have been exposed; do not reuse them for fresh confirmation.

## Next actions, in order

- [x] Search the accessible source, Box EEGMI_Personal and two Desktop distribution roots for saved GUI-session metadata. November 6: only two old November 3 three-class session records were found, both with empty results and no cue events; one is demo, the other marked live but still a recorded reference. Neither is completed personal validation. No November 5 lab recording was found in these roots.
- [ ] Obtain the lab PC's complete relevant `session_reports/` subfolders (raw NPZ, metadata, cue events, predictions, personal checkpoint/contract/training diagnostics and bridge preflight/logs). Preserve their relationships and privacy. Local historical EEGMI datasets are not substitutes for these new lab sessions.
- [ ] Complete the three-class lab trial. Verify montage/reference and clean signal, fit a personal model, freeze it, collect a fresh validation block and save the complete session. Repeat independently in each expanded package, one X24 bridge at a time. If binary acceptance was a transport or format check, collect a separate fresh binary validation block as needed.
- [ ] Analyze completed validation: cue-level balanced accuracy, per-class recall, confusion matrix including rejected cues, accepted coverage, timing gaps, artifacts and train/tuning gaps. State sample sizes and uncertainty. Keep binary and three-class results separate.
- [ ] If performance is below target, investigate labels/cue timing, physical quality, units/montage, rejected windows and calibration adequacy before changing the NN. Select changes using calibration training/tuning only; reserve new validation data for confirmation.
- [ ] Version, test and document any necessary changes; update current operator instructions and this handoff. Publish only code/non-identifying evidence; keep recordings and personal models private.

Cross-subject generalization remains deferred.

## November 10 latest session

Prepared the experimental 5-, 7- and 9-class X24 ZIPs for GitHub `master` in the commit titled `Package 5-, 7- and 9-class X24 validation labs`. Each starts without a checkpoint, fits from fresh personal calibration, reserves later whole cue blocks for tuning, and saves a separate session with frozen-model validation, rejection-aware confusion matrix, and per-class recall. The confidence floor is fixed at 1.5 divided by the class count with a 0.05 runner-up margin. There are no EEG recordings, private models or vendor SDK files in the ZIPs.

Software validation covered repeatable ZIP generation, hash verification through all three extracted batch launchers, and synthetic cue collection, actual fitting, saved-model reload, frozen validation and saving for all three profiles. Full Python suite: 118 passed and one Windows symlink test skipped. Binary and three-class integrity checks passed 21 and 27 files respectively. Hardware and wearer accuracy for the expanded sets still require a live lab session.

Delivery includes the three ZIPs in `downloads/` and checksum-verified copies in the local Box and Desktop download folders. Box cloud sync is unconfirmed. No GUI, bridge, training or acquisition process was left running. The next action is to finish three-class validation, then extract and test each expanded package with the lab X24.

## November 6 latest session

Request: "If there's something you can do, do it." Continued the new handoff priorities after documentation commit `89a5074`.

Completed: fetched origin (no newer changes), searched accessible saved-session metadata including ignored session folders, and established the missing lab-data boundary above. Fixed the X24 supervisor duplicate-outlet discovery: `minimum=1` could return before observing a second matching source; `minimum=2` with the existing one-second timeout permits the ambiguity check to run. Added regression tests for duplicate refusal before opening an inlet and failure before discovery when the child bridge has exited. No model/core changes, environment modifications, training or hardware acquisition were performed.

Validation performed here: full suite 112 passed / one Windows symlink skip in 65.02 seconds (collection began before adding the two supervisor tests); targeted bridge/supervisor run 8 passed, including the two new regression cases. Binary verifier passed all 21 retained fingerprints; three-class verifier passed all 27 required runtime/reference files. No physical repeat test was performed. An accidentally duplicated full-suite process was stopped; one full run completed. No need to repeat unrelated fitting studies.

Delivery: previous documentation commit `89a5074` is pushed. The current change is committed under `Reject duplicate X24 outlets during supervisor preflight` and pushed to origin/master; consult Git history/final response if interrupted. Current handoff and a separate `x24_supervisor_update/` copy of the changed supervisor/test and application instructions are delivered locally to Box and both Desktop distributions. Their older complete runtimes are not upgraded by this incremental delivery; use the current Git checkout to run the supported X24 launchers. Box cloud upload is unconfirmed.

Processes: this session started no GUI, bridge, training or acquisition processes.

Next concrete action: obtain the complete lab-PC session files listed above, then establish whether personal calibration plus a complete unseen validation block has actually been performed. If none exists, collect it in the lab using the current launchers before claiming accuracy.
