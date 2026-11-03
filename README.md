# Personal motor-imagery BCI Studio

Fit an EEGNet to your own **REST versus imagined right-hand grasp** calibration, then evaluate the frozen model on new cue trials. Windows are 2 seconds, stride .25 seconds (87.5% overlap), with continuous causal 8–30 Hz filtering.

The software pipeline has integration tests. Actual B-Alert X24 driver compatibility and accuracy on a new wearer still require the lab session. Read [the audit](DEBUG_AUDIT.md), [lab procedure](README_LAB_GUIDE.md) and separate [multiclass plan](MULTICLASS_PLAN.md).

## Start

Use 64-bit Python 3.11 (tested):

```powershell
git clone https://github.com/ZioJack1999/master-project.git
cd master-project
.\Setup_BCI_Studio.bat
.\Launch_BCI_3D_Studio.bat
```

Setup creates `%USERPROFILE%\.bci_studio\venv`. Launcher prefers a project `.venv` if present and opens `http://127.0.0.1:8050` from the correct directory.

Verify physical electrode sites, reference, LSL labels, units and 256 Hz rate first. Default `lab_config.json` uses an **untrained template** for BCIFEST's `STANDARD_MONTAGE`. That name does not prove your cap uses it. The EEGMI custom layout has `lab_config.eegmi_custom20.example.json`. Never rename physical sites to fit weights. See the lab guide for both lists and acquisition routes.

Start the matching acquisition-to-LSL bridge, then inspect:

```powershell
& "$env:USERPROFILE\.bci_studio\venv\Scripts\python.exe" lab_preflight.py --inspect-lsl
```

Enter a pseudonymous ID → Start Stream → Personal calibration. The default collects 18 randomized 8-second cues (144 seconds), trains from scratch and fits a threshold. Then Start Stream → Validate for **20 new cues**, with weights/threshold frozen during the run. This is a short pilot calibration; more data may be needed. Tuning is not final validation. Reset clears recording buffers and retains the fitted model; Personal calibration starts a new fit for the entered person.

## Recorded-data evidence

Real EEGMI subject 08: sessions 1–4 train, 5 tune, 6 test; custom 20 sites. Corrected fixed-normalization EEGNet achieved **96.57% held-out-session balanced accuracy on raw NN window predictions**. Continuous live-engine replay of session 6/run 1 at saved threshold .87, with EMA/dwell, gave **89.72% window balanced accuracy and 95% cue-trial balanced accuracy (19/20)**. These are recorded-data results on a known participant, not a live result on you. [Definitions and limits](DEBUG_AUDIT.md#measured-results).

Legacy weights are retained with unverified contracts. The default template has random weights: its probabilities have no learned meaning and cannot emit hand predictions. The trained subject-08 checkpoint is a reference for its montage, not a model for an arbitrary wearer.

## Demo, replay and tests

```powershell
.\Launch_BCI_3D_Studio.bat --demo
.\Launch_BCI_3D_Studio.bat --config lab_config.recorded_reference.example.json --replay-edf "C:\path\to\sub-08_ses-06_task-MI_run-01_eeg.edf"
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Demo uses stationary synthetic noise. Replay paces actual EDF samples at their recorded rate. Both disable hand predictions and prospective validation. The scope displays every filtered sample in the last 2 seconds; probability traces distinguish raw, EMA and threshold. Synthetic tests establish software behavior only.

Session reports save EEG volts, LSL timestamps, segment boundaries, cue/run IDs, participant ID, prediction model hashes/thresholds/quality, metrics and plots. Reports and personal weights remain private and excluded from Git. Save Session pauses acquisition; shutdown saves recorded data.

`train_lab_model.py` trains recorded data with whole-group splits; [the audit](DEBUG_AUDIT.md#reproduction) gives reproduction commands. `train_personal_model.py` fits saved live calibration. `replay_recording.py` uses the exact live engine and saved threshold. `udp_monitor.py` checks transport. `integrations/HandController.cs` is a reference Unity receiver needing a Unity build and Animator check. `requirements-tested.txt` records audit package versions.
