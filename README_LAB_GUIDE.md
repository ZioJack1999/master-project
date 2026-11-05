# Binary X24 lab procedure

Use `Launch_BINARY_X24.bat` from this folder. It starts 32-bit mastersp acquisition and the existing 64-bit PyTorch GUI at http://127.0.0.1:8052. See [connection details](X24_CONNECTION_STATUS.md) for SDK paths, checks and known limitations.

## Prepare and check

1. Connect the ESU, power the X24 and close competing acquisition programs.
2. Verify the physical cap/reference and electrode quality with the lab's vendor tools. The studio does not measure impedance or change the reference.
3. Launch the binary app. Leave Demo Mode OFF. Enter a pseudonymous participant ID and click Start Stream.
4. Inspect the live C3/Cz/C4 traces and quality status. Fix artifacts, flat channels and connection errors before calibration.

The supported bridge profile is the first-generation X24 10-20 at 256 Hz. It reads the SDK's actual channel names and maps these 20 EEG sites into model order:

Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, POz, O2.

SDK labels do not prove physical electrode placement. Do not rename electrodes to make a different cap fit. The alternate EEGMI custom20 example/template is retained for explicitly matching acquisition, not as a label override for this stock bridge.

## Calibrate and validate

1. Click Personal calibration: 18 randomized 8-second REST/imagined right-hand grasp cues. During action cues imagine the grasp while keeping fingers, jaw and shoulders still.
2. Wait for Personal model ready. The initial template is untrained, and its probabilities have no learned meaning. Initial hand predictions are withheld.
3. Start Stream again, then Validate for 20 NEW 8-second cues. Keep the model and threshold fixed throughout this block.
4. Save Session & Stop. Keep EEG, events, predictions, model identity, quality/exclusion counts and plots together. A second new block with the same frozen model is useful; keep each result separate.

Calibration/tuning scores are not validation. Personal accuracy and across-day reliability remain unestablished until the appropriate new data are collected. Recalibration requires new validation cues.

The initial template is loaded after a fresh launch; run personal calibration again. The binary GUI implements a virtual hand/UDP output, not FES.

## Saving and restart

Private files go to `session_reports/`, including personal models. Save before shutting down. Closing only the browser does not stop the backend. Ctrl+C in the launcher stops both processes; then power-cycle/reconnect as needed and run the launcher again. If a timestamp/connection error interrupts a cue run, preserve the data and start a new run after fixing the connection.
