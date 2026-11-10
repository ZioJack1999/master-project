# Expanded X24 motor imagery lab

Each ZIP runs one class set. Extract it to its own writable folder and use the same X24/ESU, stock20 physical montage and Windows runtime as the binary and three-class lab.

| Package | Imagery cues, in addition to Rest | Calibration / fresh validation | Local port |
|---|---|---|---|
| 5-class | Volar Grasp, Finger Extension, Wrist Flexion, Wrist Extension | 60 cues (8 min) / 50 cues (6 min 40 s) | 8055 |
| 7-class | 5-class set plus Wrist Pronation and Wrist Supination | 84 cues (11 min 12 s) / 70 cues (9 min 20 s) | 8057 |
| 9-class | 7-class set plus Opponens Grasp and Pinch Grasp | 108 cues (14 min 24 s) / 90 cues (12 min) | 8059 |

Times are cue durations only. Display transitions, fitting and setup add time. The 5- and 9-class sets retain the earlier research definitions. The 7-class set is a new intermediate profile. Each uses 12 eight-second calibration cues and 10 new validation cues per class. These are experimental display-and-recording workflows for evaluating personal decoding.

## Set up and run

1. Finish and save the previous experiment, then press Ctrl+C in its launcher window. The ESU can be owned by only one bridge at a time.
2. Use the installed 32-bit mastersp environment at `%USERPROFILE%\miniconda3\envs\mastersp` and x86 Athena SDK at `C:\ABM\B-Alert\SDK`. The decoder uses the existing 64-bit conda base Python at `%USERPROFILE%\miniconda3\python.exe`. If the decoder interpreter is elsewhere, set `X24_DECODER_PYTHON` to its full path before launching. The ZIP contains source files. It depends on the lab's installed SDK, drivers and Python libraries.
3. From the newly extracted folder, restore the local 32-bit LSL runtime once. If the working binary/three-class checkout already has `.runtime/x24-lsl32`, copy that entire folder to the matching path in this package. Otherwise use PowerShell from the extracted folder:

   ```powershell
   & "$env:USERPROFILE\miniconda3\envs\mastersp\python.exe" -m pip install --no-deps --target .runtime/x24-lsl32 pylsl==1.17.6
   ```

4. Run `Launch_MULTICLASS_X24.bat`. Use `Launch_MULTICLASS_X24.bat --verify-only` to check the extracted files without opening the device. Use `--check-only` for a short live bridge and fresh-stream check. With the normal launcher, keep its window open until you have saved and stopped.
5. Leave Demo off. Apply your pseudonymous participant ID, connect EEG, then select Start Stream. Confirm electrode placement, reference, signal quality and units as in the other X24 workflows.
6. Click **Personal calibration**. Keep the cue tab visible. Imagine each displayed movement with the same hand, keeping your actual hand, jaw and shoulders still. During Rest, relax and remain still. The console trains from scratch and reserves the later calibration blocks for tuning. It requires at least eight clean cues per class overall, including four for training and two in the reserved tuning blocks. The package starts without weights.
7. Once the personal model is ready, click **Fresh validation**. A frozen checkpoint and decision policy are used for this new cue block. Review balanced accuracy, per-class recall, the confusion matrix and rejection coverage. Rejected cues count as errors. Save the complete session and stop before switching packages.

Saved EEG, cue events, predictions and models are private under the package's `session_reports/<count>class/` folder. Reopening a package lets you explicitly load a matching personal model. Collect a new calibration if the cap or reference changes.

The confidence floor is 1.5 divided by the class count, with a fixed 0.05 lead over the runner-up. A cue requires a unique winning vote, at least four accepted updates and acceptance on at least half its eligible updates. These preset rules have software coverage. Prospective accuracy and live X24 performance for the expanded profiles remain to be measured.
