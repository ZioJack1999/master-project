# Three-class X24 lab procedure

Classes: **Rest, Volar Grasp, Finger Extension**. Run `Launch_THREECLASS_X24.bat` from this folder. The launcher uses the same 32-bit mastersp/Athena â†’ LSL â†’ 64-bit PyTorch route as binary and opens http://127.0.0.1:8051.

Finish/save the binary session and stop its launcher before switching. The three-class application records and displays predictions; it sends no actuator commands.

Live transport, all-site quality display, three-class probabilities and session saving were verified on the physical X24. A saved 46.28-second signal check contains 20 channels and 11,848 samples/channel at 256 Hz. No wearer calibration or personal accuracy test has been performed.

## Live test and personal calibration

1. Connect the ESU, power on the X24 and verify electrodes/reference and the physical montage. Launch the app and leave Demo OFF.
2. Check the hardware/source, timestamps and channel quality for all 20 sites. The stock bridge requires the physical X24 10-20 layout; do not substitute electrode names.
3. Enter a pseudonymous participant ID, click Connect/Reconnect EEG for that ID, then Start stream and run personal calibration: 12 eight-second cues per class, **36 cues / 288 seconds**, plus fitting.
4. Follow the visible imagery cues without moving your fingers, jaw or shoulders. Keep the browser visible; cue timing requires display acknowledgement. Stop/Cancel or lost display/EEG freshness aborts an incomplete run.
5. Wait for Personal model ready. Review clean-cue counts and tuning/coverage diagnostics. Poor data may require a complete new calibration.
6. Start a frozen validation block: **30 NEW cues / 240 seconds**. Do not change model/policy after seeing results and count the same data as a new test.
7. Save the session. Keep raw EEG, events, predictions, model hashes, quality/rejection counts and metrics together under `session_reports/threeclass/`.

The supplied participant-08 covariance-NN checkpoint is a **recorded reference**, not a personal model for this wearer. Its historical performance does not predict your accuracy. The GUI requires a personal fit before fresh validation.

Calibration partitions whole cues before signal-quality exclusions, fits preprocessing only on training cues, and reserves later cues for tuning. Validation uses a frozen saved model and policy; rejected cues count as errors. Those safeguards do not establish biological performance without new wearer data.

## Reusing a personal model

Enter the matching participant ID and explicitly select a verified saved personal model. Identity, channel order, preprocessing, provenance and model bytes are checked. A loaded model still needs new validation cues for the current cap placement/session. A new participant cannot inherit another person's fitted model.

The saved-model menu shows each decoder's window length and classifier. A two-second model can produce its first cue-eligible window about two seconds earlier than a four-second model. Both update every 250 ms. During validation, the current-window display waits for EEG from the active cue. The final result shows the usual confidence-gated vote and a secondary comparison that averages clean window probabilities per cue.

During streaming, the timing line reports EEG sample to prediction, decoder chunk duration, and the most recent sample-to-paint acknowledgement. The last measurement includes browser scheduling and the acknowledgement's trip back to the server. The timestamps are saved with the predictions.

## Checks and limitations

`Launch_THREECLASS_X24.bat --check-only` verifies the reference/runtime plus fresh labeled 256 Hz EEG through both architectures. `threeclass_preflight.py --config <run config> --inspect-lsl` provides acquisition details when the bridge is already running. The selected private run config is under `session_reports/x24_bridge/`.

The retained software tests cover cue acknowledgement, gaps/abort behavior, all-site quality, personal fitting, saved-model reuse and live LSL transport. Hardware transport/display latency and prospective personal accuracy require lab validation. See [shared connection notes](X24_CONNECTION_STATUS.md).

Keep the launcher running during use. Save before stopping it with Ctrl+C. Closing the browser tab alone leaves acquisition active.
