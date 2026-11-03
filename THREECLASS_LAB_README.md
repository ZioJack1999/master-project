# Three-class lab package — Rest, Volar Grasp, Finger Extension

Use this package separately from the binary Rest/action ZIP. Its launcher opens the **three-class console at http://127.0.0.1:8051/**. It records EEG, presents cues, fits your personal three-class model and evaluates new cues. It displays predictions and implements no actuator or FES output. Run one acquisition consumer at a time if the lab bridge/device requires exclusive ownership.

The supplied participant-08 reference achieved **81.6% cue balanced accuracy** and **77.3% continuous-replay cue balanced accuracy** on that participant's recorded data. Volar Grasp recall was 53.8%. These figures are historical evidence, not your accuracy. The live X24/ESU connection, new personal calibration and three-class performance on you require lab validation. The console labels the reference explicitly and requires a fitted personal model before fresh validation.

## Install and connect

1. Extract the complete three-class ZIP into its own writable folder. Keep the binary package in a different folder. Do not run from inside the ZIP.
2. Install 64-bit Python 3.11, then run `Setup_THREECLASS_Lab.bat`. Setup downloads Python requirements; vendor acquisition/SDK/driver software and the appropriate acquisition-to-LSL bridge are separate prerequisites. This package does not contain the vendor installer.
3. Establish clean EEG in the lab's existing licensed acquisition software first. Verify the physical electrodes and reference, impedance, X24/ESU pairing, stream timestamps and numeric units. A 32-bit vendor SDK needs its own matching process; the 64-bit model process communicates through LSL.
4. Copy the appropriate supplied profile to `threeclass_config.local.json` and make lab-specific edits there. Select the EEG stream by name and preferably source ID. Use the producer's verified channel labels and units; if metadata are absent, supply the full verified field order, including non-EEG prefix fields. Set an input-unit override only after confirming whether values are V, mV or µV. The required sample rate is **256 Hz**, with no implicit resampling.
5. Run the following in PowerShell from the extracted folder. The probe checks compatible metadata and fresh samples; it does not establish clean EEG or decoding accuracy. Then run `Launch_THREECLASS_Lab.bat --config threeclass_config.local.json`. If using the unchanged default profile, omit `--config` from both commands.

```powershell
& "$env:USERPROFILE\.bci_threeclass\venv\Scripts\python.exe" threeclass_preflight.py --config threeclass_config.local.json --inspect-lsl
```

The default stock20 physical sites are Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, POz, O2. The recorded reference requires this physical set. `threeclass_config.custom20.example.json` supports fresh personal fitting with the separate EEGMI-style custom20 arrangement. Verify the profile against the actual cap. Do not rename different electrode sites or substitute missing channels to match a model.

## Personal calibration

1. Enter a pseudonymous participant ID and choose **Connect EEG**. Inspect the actual C3/Cz/C4 EEG traces and signal-quality status. Select **Start stream** to preview live signal. The default reference probabilities describe a model trained for another participant.
2. When signal quality is satisfactory, choose **Personal calibration**. The default is **12 cues per class × 8 seconds: 36 randomized cues, 4.8 minutes**, plus fitting time. Follow the visible cue. During Volar Grasp imagine the feeling of closing your hand around an object; during Finger Extension imagine straightening your fingers. Keep your actual fingers, jaw and shoulders still.
3. The application saves raw EEG and cue events. It fits only on your completed calibration cues, splitting whole independent cue trials into training and tuning sets. Causal filtering, accepted-window checks and feature normalization are recorded with the model. Overlapping windows from one cue remain in the same partition; they do not become new independent trials.
4. Wait for **Personal three-class model**. Training/tuning accuracy is calibration evidence, not final validation. If fitting fails because too few independent clean cues remain, inspect signal quality and collect a new calibration run. Stop/cancel must leave an incomplete run unvalidated.

EEG windows are **4 seconds**, with a **0.25-second stride**: 93.75% overlap. Each restart needs a full warm-up window. The decoder retains causal filter state across contiguous samples and resets after gaps. The cue scoring also allows a settling margin; early windows that span cue boundaries do not count as clean evidence for the next class. Cue times use the LSL clock; browser/display latency has not been physically measured.

## Fresh validation

1. After fitting, choose **Fresh validation**. The default is **10 new cues per class × 8 seconds: 30 cues, 4 minutes**. Personal model, normalization, offsets and decision settings remain frozen during the block.
2. Follow every cue and remain still. **Uncertain**, low-quality and warm-up decisions are distinct from Rest; do not treat a rejected cue as correctly recognized Rest.
3. After completion, inspect cue balanced accuracy, overall cue accuracy, per-class recall, scored/rejected cue counts and coverage. **The primary accuracy includes every completed cue; rejected cues count as errors.** The confusion table includes a separate Rejected column. Any accepted-cue score is secondary. A high score on a small accepted subset is insufficient if many cues were rejected. Keep the full result and exclusions.
4. Choose **Save session** and retain raw EEG, timestamps, event labels, predictions, model hash and validation metadata together. Model/calibration files and EEG are private; do not publish them to Git.
5. Collect a second new validation block with the same frozen model/settings. Report each block separately. If you change settings or recalibrate after observing a result, use new cues for the next validation and keep the earlier result as development evidence.

Define success before the first validation. The project's ≥70% cue balanced-accuracy target is a pilot target, not a certified threshold. Also review all three recalls, uncertainty and exclusion rates. Thirty cues provide limited precision; another day or cap placement is needed to examine stability. Never infer your performance from the supplied historical 81.6% reference score.

## Reading the console

- **Recorded participant-08 reference**: useful for software checks and observing outputs; fresh personal validation is unavailable until your calibration finishes.
- **Personal three-class model**: a model fitted on the entered participant's completed calibration. Its hash identifies the exact weights used in predictions and validation.
- **Current probabilities / history**: class probabilities from accepted EEG windows, with raw and EMA values shown separately. Probabilities are confidence estimates, not measured hand force or angles.
- **Overlapping raw/EMA curves**: the fixed three-class policy uses EMA alpha 1, so raw and EMA probabilities coincide. This preserves the selected recorded-model policy; it is not evidence of artificial probability clipping.
- **Synthetic demo**: clearly labeled noise visualization. It cannot provide calibration or validation evidence. Return to live EEG to record a real session.
- **Stop / cancel**: pauses decoding and aborts an incomplete protocol. The displayed decision becomes paused/uncertain instead of retaining a stale class. Save before resetting the view.
- **Keep the cue display visible**: hiding the tab during calibration or validation aborts the active run. Closing/disconnecting the cue display also triggers a timeout. Start a new visible cue run after interruption.

The binary ZIP has its own launcher, settings and session folder. Keep both extracted folders, calibration models and validation results separate. Verify your local saved files before leaving the lab and confirm Box upload completion before relying on its cloud copy.
