# X24 lab guide

Open `Launch_Studio.bat` and choose the X24 bridge. The shared dashboard opens on port 8051 with Three-class selected. To use the binary fallback, stop streaming, wait for fitting to finish, and choose Binary from **Class profile**. The X24 bridge stays active during the switch. The batch files below open a fixed profile. See [connection details](X24_CONNECTION_STATUS.md) for SDK paths, checks and known limitations.

| Profile | Launcher | GUI |
|---|---|---|
| Three-class | `Launch_THREECLASS_X24.bat` | http://127.0.0.1:8051 |
| Binary | `Launch_BINARY_X24.bat` | http://127.0.0.1:8052 |

## Prepare either profile

1. Connect the ESU, power the X24 and close competing acquisition programs.
2. Verify the physical cap/reference and electrode quality with the lab's vendor tools. The studio does not measure impedance or change the reference.
3. Launch the app with Demo off. Enter a pseudonymous participant ID, apply it with Connect/Reconnect EEG when prompted, then click Start Stream.
4. Inspect the source, timestamps, live C3/Cz/C4 traces and quality status. Fix artifacts, flat channels and connection errors before calibration. Review all 20 sites in the three-class app.

The supported bridge profile is the first-generation X24 10-20 at 256 Hz. It reads the SDK's actual channel names and maps these 20 EEG sites into model order:

Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, POz, O2.

SDK labels do not prove physical electrode placement. Do not rename electrodes to make a different cap fit. The alternate EEGMI custom20 example/template is retained for explicitly matching acquisition, not as a label override for this stock bridge.

## Three-class calibration and validation

The classes are **Rest, Volar Grasp, Finger Extension**. Live transport, all-site quality display, three-class probabilities and session saving were verified on the lab X24. A saved 46.28-second signal check contains 20 channels and 11,848 samples per channel at 256 Hz. Wearer calibration and prospective personal accuracy still need a lab trial.

1. Run personal calibration for 12 eight-second cues per class, **36 cues / 288 seconds**, plus fitting.
2. Follow the visible imagery cues without moving your fingers, jaw or shoulders. Keep the browser visible for cue display acknowledgement. Stop/Cancel or loss of display or EEG freshness aborts an incomplete run.
3. Wait for Personal model ready. Review clean-cue counts and tuning/coverage diagnostics. Poor data may require a complete new calibration.
4. Start a frozen validation block of **30 new cues / 240 seconds**. Use the same model and policy for the block. If either changes after review, collect new cues to evaluate it.
5. Save the session under `session_reports/threeclass/`. Keep raw EEG, events, predictions, model hashes, quality/rejection counts and metrics together.

The supplied participant-08 covariance-NN checkpoint is a recorded reference. The GUI requires a personal fit before fresh validation. Calibration partitions whole cues before signal-quality exclusions, fits preprocessing only on training cues and reserves later cues for tuning. Validation uses a frozen saved model and policy. Rejected cues count as errors.

### Reusing a personal model

Enter the matching participant ID and explicitly select a verified saved personal model. Identity, channel order, preprocessing, provenance and model bytes are checked. Collect fresh validation cues for the current cap placement and session.

The saved-model menu shows each decoder's window length and classifier. A two-second model can produce its first cue-eligible window about two seconds earlier than a four-second model. Both update every 250 ms. During validation, the current-window display waits for EEG from the active cue. The final result shows the confidence-gated vote and a secondary comparison that averages clean window probabilities per cue.

During streaming, the timing line reports EEG sample to prediction, decoder chunk duration and the most recent sample-to-paint acknowledgement. The last measurement includes browser scheduling and the acknowledgement's trip back to the server. The timestamps are saved with the predictions.

### Three-class checks

`Launch_THREECLASS_X24.bat --check-only` verifies the reference and runtime, then checks for fresh labeled 256 Hz EEG through both architectures. Use `threeclass_preflight.py --config <run config> --inspect-lsl` for acquisition details when the bridge is running. The private run config is under `session_reports/x24_bridge/`.

## Binary calibration and validation

1. Click Personal calibration: 18 randomized 8-second REST/imagined right-hand grasp cues. During action cues imagine the grasp while keeping fingers, jaw and shoulders still.
2. Wait for Personal model ready. The initial template is untrained, and its probabilities have no learned meaning. Initial hand predictions are withheld.
3. Start Stream again, then Validate for 20 NEW 8-second cues. Keep the model and threshold fixed throughout this block.
4. Save Session & Stop. Keep EEG, events, predictions, model identity, quality/exclusion counts and plots together. A second new block with the same frozen model is useful; keep each result separate.

Calibration/tuning scores are not validation. Personal accuracy and across-day reliability remain unestablished until the appropriate new data are collected. Recalibration requires new validation cues.

The initial template is loaded after a fresh launch. Run personal calibration again. The binary GUI sends virtual hand output over UDP.

## Saving and restart

Private files go to `session_reports/`, including personal models. Save and use Ctrl+C in the launcher to stop both processes before switching profiles or restarting. If a timestamp or connection error interrupts a cue run, preserve the data, fix the connection and start a new run.
