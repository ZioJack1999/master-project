# B-Alert X24 lab procedure

Person-specific REST versus imagined right-hand grasp with the pictured X24 and ESU. Physical hardware and the lab's installed software have not been tested in this audit.

## Acquisition routes

Confirm EEG in the installed licensed vendor acquisition first. ABM documents [B-Alert Live and SDK integration](https://www.advancedbrainmonitoring.com/products/b-alert-mobile). An older X24 may need its matching legacy SDK; preserve a working lab installation.

| Installed route | Studio connection | Remaining check |
|---|---|---|
| B-Alert Live / Athena SDK | Separate matching B-Alert LSL application | SDK version, bridge build, labels, units, timestamps |
| Legacy SDK / lab abmbci | Existing acquisition plus validated LSL outlet | Source uses 32-bit Python 3.9/DLLs; no hardware test here |
| Vendor EDF export only | Offline replay/training | EDF export alone is not a live stream |
| Existing BCI2000 / FieldTrip acquisition | Independently verified LSL output | Additional adapter route, not implemented/tested here |

[LSL App-BAlert](https://github.com/labstreaminglayer/App-BAlert) documents Athena/x86 dependencies and a legacy v1.0 tag; its README says hardware testing is incomplete. [BCI2000 B-Alert documentation](https://www.bci2000.org/mediawiki/index.php/Contributions%3AB-Alert) describes X24 acquisition at 256 Hz. These are candidate interfaces, not proof that every compatible program exports LSL. Studio accepts `type=EEG` LSL; other interfaces need a bridge.

Keep vendor acquisition in its required process architecture. A 64-bit PyTorch process cannot load 32-bit vendor DLLs. LSL connects separate processes. The older BCIFEST GUI hardware manager also initializes FES: do not launch it merely to inspect EEG. This studio implements virtual-hand UDP only, no stimulator integration.

ESU participates in vendor acquisition/synchronization; it does not classify EEG. Its driver pairing and marker path need a lab check.

## Physical montage

| Profile | Sites in model order |
|---|---|
| lab_config.json (`x24_stock20`) | Fp1, Fp2, F7, F3, Fz, F4, F8, T3, C3, Cz, C4, T4, T5, P3, Pz, P4, T6, O1, POz, O2 |
| lab_config.eegmi_custom20.example.json | F3, F1, Fz, F2, F4, C3, C1, Cz, C2, C4, CPz, P3, P1, Pz, P2, P4, POz, O1, Oz, O2 |

The first is BCIFEST's `STANDARD_MONTAGE` and matches the set of sites in two inspected BCIFEST EDFs; file order differs and is reordered by name. The second is EEGMI metadata. The X24 case alone establishes neither cap layout. Verify strip/cap sites and reference with the lab operator. Establish physical equivalence before supplying aliases such as T7/T8. Never substitute missing sites.

Both personal profiles use random architecture weights and learn from your calibration. Regenerate with `create_personal_template.py` if necessary. Do not choose pretrained initialization with a random template. Prepare/check electrodes using manufacturer instructions and the lab protocol. Studio does not measure impedance, change hardware reference, or resample.

## Preflight

1. Start matching vendor acquisition and EEG LSL bridge; close conflicting exclusive device owners.
2. Run `lab_preflight.py --config YOUR_CONFIG --inspect-lsl` with studio Python. `ready` means compatible metadata/checkpoint only, not fresh signal or accuracy.
3. Set `stream_name` and preferably `source_id`. Ambiguous streams and rates other than 256 Hz are rejected.
4. Prefer producer labels/units. Otherwise set `channel_labels` to the **full verified stream field order**, including six prefix fields in a 30-field ABM outlet. Only named EEG sites are selected. Set `input_unit` to `uV`, `mV` or `V` after confirming SDK numeric scale; the adapter converts to volts. A plausible graph is not proof of units.
5. Launch with `Launch_BCI_3D_Studio.bat --config YOUR_CONFIG`. Missing/duplicate sites, nonfinite samples, stale/backward timestamps and gaps stop decoding. No implicit resampling.

Inspect all electrodes in vendor software. Studio shows C3/Cz/C4 only. Automatic checks reject filtered peaks above 100 µV and near-flat windows, but cannot identify every eye/muscle artifact.

## Personal calibration and validation

1. Enter a pseudonymous ID and Start Stream. Template probabilities are meaningless; hand predictions are withheld.
2. Personal calibration: 18 randomized 8-second cues, 144 seconds plus fitting. Imagine the feeling of grasping; keep actual fingers, jaw and shoulders still. Physical movement/EMG separation is not proof of imagery decoding.
3. Raw EEG is saved. **Whole cue trials** are split for training, tuning and a small internal test. Default NN training starts from scratch. Early stopping/threshold fitting use tuning only. Aborted/failed fitting leaves personalization unavailable; improve the signal and repeat.
4. After Personal model ready, Start Stream → Validate: 20 NEW 8-second cues, 160 seconds. Model/threshold stay frozen during the run. Scoring uses actual EMA/dwell decisions with one majority vote per cue after full-window settling. Stop/Cancel aborts an incomplete run.
5. Save Session & Stop; keep NPZ/events/predictions/metadata together. Tuning and fresh validation are separate. Changing settings after observing validation makes that run development data; collect new cues for another test.
6. Repeat frozen testing in another block, ideally another day/cap placement. Define success beforehand. Example project pilot target: ≥70% trial balanced accuracy, both classes detected and permutation p<.05. This is not a certified standard; 20 trials have substantial uncertainty.

Eighteen cues are a pilot minimum, not a guarantee of enough NN data. If new-cue scores stay near chance, collect more balanced independent trials, inspect labels/artifacts, and compare bandpower/CSP before increasing NN complexity. Insufficient clean predictions fail the run; retain exclusion rates, not only successful runs.

Cue times use the studio/LSL clock. Browser/display latency is unmeasured. For precise event-related timing add a measured display marker/photodiode/ESU path. A .5-second settling margin reduces boundary mixing but does not establish millisecond accuracy.

## Prediction and virtual hand

512-sample (2 s) windows, 64-sample (.25 s) stride, **87.5% overlap**. Restart requires a full window. Causal filter state persists across chunks. Raw probability → EMA (alpha .3) → threshold → two-update dwell are separate stages. Flexion is a confidence/animation parameter, not measured grasp force or angle.

UDP defaults to localhost:5005. Calibration, demo and replay withhold predictions. After fitting, live validation/free decoding can send virtual-hand packets. Pause, artifacts, gaps and shutdown neutralize commands; the reference Unity receiver times out after .75 s. Test `udp_monitor.py` first. The C# receiver still needs a lab Unity build/Animator check.

| Symptom | Check |
|---|---|
| No stream | Bridge, EEG type, process architecture, exclusive ownership, selected source |
| Missing sites/units | Physical cap and SDK mapping; never invent labels or scales |
| Stale timestamps/gaps | LSL clocks, buffering/network; restart after reconnect |
| Flat probabilities | Untrained versus fitted model, hash, units, quality and actual EEG |
| Fit good, validation poor | Small sample, artifacts, timing, reference/cap changes, overfitting; retain both results |
| Offline good, live lower/slower | EMA/dwell/saved threshold alter deployment decisions; compare matching metrics |

Physical X24/ESU acquisition, Unity build and live self-test remain for the first lab session.
