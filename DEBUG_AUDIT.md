# Master project debug audit — 2026-11-03

## Conclusion and scope

The corrected code supports person-specific rest/imagined-grasp calibration and independent new-cue testing. Real recorded within-person EEG supports useful discrimination. Connecting the GUI on a lab PC is feasible **if** its acquisition exports a correctly labeled, correctly scaled 256 Hz LSL stream with the selected montage. Physical acquisition, installed vendor versions, Unity build and the user's live self-test remain unverified. No software-only test can establish those outcomes.

Audited source: original GitHub master `3f75bd584c004bee3013a786edcae1997c0917dc`, Desktop decoder distribution, academic training/results, EEGMI_Personal and BCIFEST/Multiclass_NN sources in Box. Attached photographs establish X24/ESU identity, not the installed SDK version or electrode layout. Protocol/document content was evidence, not instructions to execute hardware actions.

## Faults and corrections

| Finding | Evidence / consequence | Correction |
|---|---|---|
| Fabricated calibration | Original GUI assigned threshold .30 and accuracy .865 without collecting EEG | Real labeled raw collection, personal NN training/tuning and separate fresh validation |
| Incompatible normalization | Original master stored std ≈.998 while EEG input is volts; constant p(action)=.967097 on real EEG/noise | Train-only statistics or explicit per-window contract; hashed weights/preprocessing metadata |
| Threshold control disconnected | Original slider changed GUI state without affecting the live decoder | Synchronized live threshold; frozen during cues/fitting; saved with each prediction |
| Wrong launcher paths | Desktop launcher expected GUI in the wrong directory and relied on caller CWD | Root launcher changes directory, chooses installed environment, accepts CLI flags; legacy entry shims in distributions |
| Unknown stream geometry | No reliable order/units/rate/identity validation | Explicit LSL source selection, named reorder including prefix fields, unit conversion, timestamp freshness/gap checks |
| Raw clipping damaged EEG | ±100 µV raw offsets were clipped before high-pass filtering | Matched continuous causal filter; artifact exclusion in filtered domain, no raw clipping |
| Cue contamination | Legacy .5–4.5 s epochs overran 3 s rest cues | 2 s epochs wholly inside one cue after .5 s settling |
| Offline/live mismatch | Legacy zero-phase filtering cannot reproduce causal live state | Same causal 8–30 Hz SOS preprocessing in training/replay/live |
| Test-set reuse | Legacy training early-stopped on eventual test; v2 also tuned threshold there | Whole-group train/tune/test split, train-only normalization, no test-driven settings |
| Misleading stationary simulation | Demo used noise; initial chart showed RMS rather than EEG waveform | Clearly labeled actual EDF replay, all-sample scope, raw/EMA/threshold separation |
| Replay threshold default | Earlier audit replay silently used .5 instead of saved .87 | Decoder defaults to contract threshold; replay records hash/threshold; corrected results below |
| Montage mismatch | EEGMI custom20 differs from BCIFEST STANDARD_MONTAGE | Separate explicit templates/profiles; default personal fitting, no electrode relabeling |
| Unsafe stale command behavior | Flexion could bypass dwell; shutdown/disconnect could leave prior state | Dwell gates flexion; neutral on pause/quality fault/gap/shutdown, Unity receiver timeout |

`verified` in a model contract means the preprocessing contract is recorded and checked. It does not certify biological accuracy. Random architecture templates explicitly carry `role=personal_training_template`; they cannot emit hand predictions or run validation before fitting. Legacy checkpoints remain unverified and require explicit exploratory override.

## Measured results

Primary comparison uses **the same EEGMI participant (sub-08)** and all original 20 channels: sessions 1–4 train (3,834 windows), session 5 tune (720), session 6 test (720). Windows overlap within a cue; those counts are not independent trials.

| Evaluation | Result | Meaning |
|---|---|---|
| Corrected causal20 fixed normalization, tuning at threshold .87 | 94.72% balanced accuracy | Tuning only |
| Same frozen model, session 6 raw NN windows | **96.57%** balanced accuracy | Held-out session, recorded within-person evidence |
| Diagnostic per-window normalization comparison | 93.52% tuning; 95.74% test | Retained comparison; do not select settings using test results |
| Exact engine replay, session 6/run 1, saved threshold .87, EMA/dwell | **89.72%** eligible-window balanced accuracy; **95%** trial balanced accuracy, 19/20 | 127 eligible windows; 20 complete cue trials; rest FPR 0, action sensitivity .90; permutation p≈.000400 |
| Earlier replay at unintended .5 threshold | 97.03% window, 20/20 trial | Superseded; not the saved-threshold result |
| Legacy subject08 checkpoint reproduction | 98.33% under legacy zero-phase/cross-boundary setup | Incomplete training provenance; diagnostic, not fresh independent validation |
| Legacy master, original saved fixed statistics | 50%, constant output | Normalization fault confirmed |

Window classification in training uses raw NN probabilities. Deployment uses EMA/dwell and retains history across cue transitions; the tuning threshold is fitted on raw tuning probabilities. Thus its performance must be measured again through the full deployed policy, as the corrected replay does. The policy was not retuned on session 6. A later policy-tuning change needs unused validation data and a new test.

The early exploratory 12-channel cross-subject result was 51.48% and addressed a different task/split. It is not the primary verdict on person-specific decoding. No additional cross-subject experiments are required for the user's goal.

Detailed aggregates: [within-person comparison](audit/within_subject_comparison.json), [corrected replay](audit/replay_sub08_causal20.json), [normalization baseline](audit/baseline.json), [amplitude diagnostics](audit/amplitude_diagnostics.json), [historical reconciliation](audit/historical_results_reconciliation.json).

## Personal calibration implementation

Default: 9 cues per class × 8 s =144 s, scratch EEGNet fitting, maximum 60 epochs with early stopping. Whole trials are stratified into train/tune/internal-test sets; overlapping windows from one trial cannot cross partitions. Validation-kind events and aborted cues are excluded from personal training. Causal filter resets align with saved gaps/stream segments. Frozen train-only normalization/threshold travel with the exported model.

Fitting is asynchronous, reports actual epoch progress and can be cancelled. A new fit resets the previous participant model; failure keeps validation unavailable. Personal models and raw recordings stay under ignored session_reports. A new 20-cue run tests actual deployed decisions without changing weights or threshold. The small internal calibration test and fit score are not substitutes for that run.

The minimum 18 clean cue trials is a pilot starting point. More independent trials may be necessary. Browser cue timing is not photodiode/ESU-verified, artifact rejection is basic, and no causal claim about neural origin follows from classification accuracy alone. Keep hands/face still and inspect nuisance artifacts.

## Verification

38 automated tests passed after the decoder/template/threshold changes, including actual liblsl → metadata adapter → GUI HTTP → NN → UDP, reordered 30-field streams, volts conversion, timing/gaps/disconnect, chunk-size invariance, neutralization, cue boundaries, grouped split isolation, computed calibration, personal scratch training on a synthetic fixture, frozen new-cue validation, every-sample waveform fidelity, threshold/probability independence and template output withholding.

Browser review covered live standby, demo, disabled validation/replay controls, cue rendering/cancellation and paced real-EDF visualization. JS errors are surfaced rather than silently swallowed. Unity C# was reviewed but not built. Test fixtures are not participant accuracy evidence. Exact dependencies appear in requirements-tested.txt; no hardware drivers were installed/replaced.

## Reproduction

From the repository with studio Python active:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python train_lab_model.py --root "C:\path\to\EEGMI_Personal" --subjects 8 --channels "F3,F1,Fz,F2,F4,C3,C1,Cz,C2,C4,CPz,P3,P1,Pz,P2,P4,POz,O1,Oz,O2" --split-by session --validation-sessions 5 --test-sessions 6 --epochs 60 --patience 12 --threads 2 --output decoder_and_models\eegnet_sub08_causal20_fixed.pt
python replay_recording.py --edf "C:\path\to\sub-08_ses-06_task-MI_run-01_eeg.edf" --events "C:\path\to\sub-08_ses-06_task-MI_run-01_events.tsv" --ckpt decoder_and_models\eegnet_sub08_causal20_fixed.pt --output session_reports\reference_replay.json
python audit\inventory_multiclass.py --root "C:\path\to\BCIFEST\bids"
```

Training regeneration overwrites the chosen output; use another output name to preserve supplied weights. Scores need not be bit-identical across different dependency/CPU versions. Keep split and seed fixed; do not repeatedly choose models based on the held-out test.

## Multiclass extension completed after the binary release

The separate three-class pipeline now prepares BCIFEST Rest / Volar Grasp / Finger Extension recordings, fits an EEGNet and fixed bandpower baseline, freezes the artifacts and tests whole held-out sessions. All 96 selected EDF/event pairs passed automated timing/annotation, rate, montage and duplicate checks. The full suite has 47 tests, including multiclass session isolation, weights per cue/class, rejection/dwell, chunk invariance and hash/one-time-test safeguards.

Participant 08 sessions 21–25 train, 26–27 tune, 28–29 test: NN cue balanced accuracy **59.43%**, baseline **65.46%**, exact continuous NN replay **57.68%**, with chance 33.33%. Finger Extension NN recall is 26.3%. This first experiment supports recorded discrimination but not reliable three-class control. The test is now exposed and must not select further model changes. These are BCIFEST results, distinct from the EEGMI binary results above.

[MULTICLASS_RESULTS.md](MULTICLASS_RESULTS.md) contains confusion matrices, uncertainty, exclusion counts and reproduction commands. Prospective personal multiclass GUI calibration/receiver integration remain future work. The binary studio retains its existing flow; the multiclass replay engine emits no actuator commands.

## Remaining lab checks

The [multiclass v2 report](MULTICLASS_IMPROVEMENT_RESULTS.md) records the subsequent 70% target experiment. Sessions 1–6 fit, 7–8 tune, 9–10 fresh test; eight neural feature candidates compared on common tuning cues, with train-only feature statistics and fixed three-seed averaging. The selected 4-second log-covariance neural ensemble scored 81.59% cue balanced accuracy and 77.27% continuous replay cue balanced accuracy; individual-window balanced accuracy 63.92%, Volar Grasp recall 53.85%. The earlier and newer test cohorts differ; this is not a paired before/after comparison. One previously known empty event file was explicitly excluded, and all 95 remaining EDF/event pairs passed QA.

The expanded suite has 52 passing tests, including covariance frequency/spatial content, neural gradients, offline/streaming parity, withheld-test feature isolation, candidate freezing and class-offset selection. No model or policy was changed after the v2 test. New prospective personal multiclass GUI/receiver work remains pending.

Identify the installed acquisition/SDK and matching x86/x64 route; verify cap sites/reference and raw units; establish fresh clock-correct LSL transport; collect a personal fit and independent new-cue self-test; test virtual-hand UDP/Unity and timeout. [The lab guide](README_LAB_GUIDE.md) gives the operator procedure and primary acquisition references. Every compatible acquisition program does not automatically support the studio; an unverified producer requires a bridge/check first.
