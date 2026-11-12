# Restart handoff for BCI Studio

November 11, 2026 · America/New_York

## Paste into a new chat

> Continue the `master-project` checkout on GitHub’s default branch, `master`. Read `RESTART_HANDOFF.md`, `AGENTS.md`, `TODO.md`, `README.md`, and `X24_CONNECTION_STATUS.md`. Check the working tree and fetch the latest `origin/master` before editing. Prioritize the live three-class X24 workflow. If the lab run has happened, start with the saved calibration and validation evidence. Otherwise, help me prepare the lab run using `README_LAB_GUIDE.md`.

## Current priority

Run three-class personal motor-imagery calibration and prospective validation with the B-Alert X24. The classes are **Rest, Volar Grasp, and Finger Extension**. The target is at least 70% cue-level balanced accuracy on genuinely new data by November 26, 2026. Personal accuracy is still unverified. Binary is the fallback and has its own calibration and validation. The 5-, 7-, and 9-class profiles and ZIPs remain experimental pending a decision on lab use.

## What changed

- `4bd5476` consolidated six overlapping batch launchers, duplicate dashboards, and separate lab guides. `Launch_Studio.bat` is the shared entry point. **Class profile** defaults to Three-class. Switching profiles requires stopping the stream and finishing fitting. The previous recording is saved, the next profile starts idle, and the X24 bridge stays up. Models and recordings stay separate by profile.
- `b36ced3` made the Three-class synthetic hand preview repeatable. After **Start stream**, it cycles through Rest, Volar Grasp, and Finger Extension for two seconds each. Extension first appears at four seconds. The dashboard and hand label the scripted pose. Binary has only Rest and Right-Hand Grasp.
- `6610228` opens the cue dashboard first in live mode. Open the hand with **3D Virtual Hand Window** when needed. A visible dashboard or valid cue acknowledgement keeps the run active. The viewer waits for a prediction from the current cue during validation, shows accepted decoder output, and rests on rejection. The launcher reports busy ports and missing x86 dependencies before acquisition. The binary UDP output also follows the live, trained-model, and protocol guards.
- The Desktop **BCI Studio** shortcut points to the shared launcher. Older Desktop deliveries and lab notes were archived. App shortcuts were restored to the Desktop and the temporary Apps folder was removed. The local Box mirror of the workflow fixes was checksum-verified. Box cloud sync is unconfirmed.

## Verified evidence and open checks

- November 11 software review at `6610228` passed **193 tests**, with two skips for unavailable CUDA and Windows symlink creation. Both X24 launchers passed `--verify-only`. The integrity checks passed for 28 three-class files and 19 binary hashes.
- The X24 supervisor received 264 fresh samples from an isolated 20-channel software LSL sender at 256 Hz and stopped the child process. A full software rehearsal acknowledged 36 calibration cues, fitted on 27 with 9 held out for tuning, reloaded the saved model, and saved 30 fresh validation cues with the model hash frozen. Zen showed Finger Extension and returned the hand to Rest after Stop.
- The November 5 lab record documents physical 20-channel X24 transport at 256 Hz and session saving with the recorded reference checkpoint. The latest workflow fixes still need the physical lab check. The home workstation lacks the x86 `mastersp` interpreter, the local 32-bit LSL DLL, and the Athena SDK DLL. The lab PC uses the existing 32-bit Python 3.9 acquisition runtime and a separate 64-bit decoder. See [X24 connection details](X24_CONNECTION_STATUS.md).

## Next steps

1. On the lab PC, close any earlier Studio or demo launcher. Open **BCI Studio** or run `Launch_Studio.bat` from the current checkout and choose **1 · BCI Studio with X24 bridge**. The shared dashboard opens on port 8051. Keep the cue dashboard visible during a run.
2. Leave Demo off. Enter a pseudonymous participant ID, click **Connect EEG**, then **Start stream**. Verify fresh labeled 256 Hz EEG, the physical montage and reference, and all-site signal quality before starting the 36 eight-second calibration cues.
3. Once the personal model is ready, click **Fresh validation** to collect 30 new eight-second cues with the model and policy frozen. Save the session before stopping the launcher with Ctrl+C. Follow the [lab guide](README_LAB_GUIDE.md) for the supported procedure.
4. Keep the complete private `session_reports/threeclass/` and relevant `session_reports/x24_bridge/` evidence together. Review balanced accuracy, per-class recall, rejection-aware confusion matrix, accepted coverage, signal quality, and training/tuning gaps. Reserve new whole cues for each fresh test. Keep EEG, personal models, participant identifiers, and logs out of Git.

Use [TODO.md](TODO.md) for the active task list and [README.md](README.md) for the current launch and software verification commands.
