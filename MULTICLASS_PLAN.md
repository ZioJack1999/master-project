# Separate multiclass NN plan — 2026-11-03

This is the requested next stage after binary validation. Multiclass training/control is **not implemented** in the binary studio. The primary objective remains within-person learning and testing, with calibration for each wearer.

## Classes established from Box

BCIFEST random multiclass task and Multiclass_NN configuration define:

| Protocol ID | Label |
|---|---|
| 1 | Opponens Grasp |
| 2 | Finger Extension |
| 3 | Volar Grasp |
| 4 | Wrist Flexion |
| 5 | Wrist Extension |
| 6 | Wrist Pronation |
| 7 | Rest |
| 8 | Pinch Grasp |
| 9 | Wrist Supination |

These are same-hand/wrist actions, not assumed left/right/feet tasks. Generic EEGMI action or BCIFEST binary Move cannot silently become a specific grasp. EEGMI and BCIFEST participant IDs/montages are separate; do not assume sub-08 refers to the same person across datasets.

Evidence read: `Multiclass_NN/config.py`, its notebooks, `BCIFEST_Personal/exp-bcifest-UM/src/exp/tasks/multiclass/task_olrandmcmi_train.py`, `src/gui/protocol_engine.py`, binary task source, and MI Training/MI Testing forms. Forms reference notebooks and stages rather than proving every saved session completed them. The original task samples 6–15 s durations; newer GUI defaults differ. Use actual events for labels/timing.

## Recorded-data inventory

Scoped scan of `BCIFEST/bids/sub-*/ses-*/eeg/*task-olrandmcmitrain*_events.tsv`: **410 files, 404 readable, six empty/unparseable**. Corresponding EDF paths exist for all 404 readable tables; existence alone is not a signal-quality check. Two representative sub-08 EDF headers were inspected: 256 Hz, 20 EEG + ECG/AUX fields, with BCIFEST montage and a different field order from model order. Event columns are onset, duration, trial_type; derive sample indices from onset and EDF rate.

| Dataset participant | Readable run tables | Empty tables among readable | Main implication |
|---|---:|---:|---|
| sub-08 | 191 across 19 sessions | 0 | Best initial inventory; all nine classes occur |
| sub-04 | 65 across 5 sessions | 45 | Fewer usable trials; many empty runs |
| sub-07 | 140 across 10 sessions | 140 | No class labels in these tables; inspect EDF annotations before any recovery |
| sub-00 | 7 | 1 | Unexpected Peace label; test/eligibility status unresolved |
| sub-99 | 1 | 0 | Six cues, missing classes; eligibility unresolved |

Sub-08 has 43 cues shorter than 2.5 s; two inspected approximately 60 s runs end with shortened final cues. A protocol's planned duration cannot replace recorded timing. Exclude windows crossing a cue or EDF boundary.

| Class | Sub-08 recorded cues | Duration ≥2.5 s (before signal rejection) |
|---|---:|---:|
| Rest | 132 | 129 |
| Volar Grasp | 130 | 125 |
| Finger Extension | 121 | 115 |
| Opponens Grasp | 128 | 125 |
| Wrist Flexion | 125 | 123 |
| Wrist Extension | 147 | 139 |
| Wrist Pronation | 141 | 134 |
| Pinch Grasp | 129 | 124 |
| Wrist Supination | 140 | 136 |

[Aggregate inventory](audit/multiclass_summary.json). These counts establish existing labels and potential windows, not usable EEG or multiclass accuracy. Recheck all intended EDFs, annotations, acquisition settings and recording eligibility before fitting. Do not relabel Peace into a listed gesture. Empty events require evidence-based recovery or exclusion.

## Stage 1 — three classes on one recorded person

Start with **Rest / Volar Grasp / Finger Extension**, a protocol-consistent neutral/close/open subset. Fix this choice before evaluating the test set. Other gestures remain available for later expansion; do not cherry-pick a subset after inspecting test accuracy.

1. Build an explicit label dictionary/aliases for spaces versus underscores. Check config imports (`PosixPath` in the current config), actual task name `olrandmcmitrain` versus notebook `motorimagery`, relative paths and subject/session selectors. Some configs list nine events but only one binary contrast; explicitly select multiclass output.
2. Validate every selected EDF header, reference, event-onset alignment and annotations. Generate counts by class/run/session after nonfinite/flat/artifact exclusions. Use continuous causal filtering in volts, explicit physical channel reorder, .5 s settling and 2 s windows initially; do not copy a notebook's 128 Hz/zero-phase assumptions into a 256 Hz live model.
3. Reserve whole later sessions or runs for test **before extracting overlapping windows**. Keep tuning sessions separate. Verify all three classes remain in each partition; record split manifests/hashes. Do not random-split nearby windows. Sub-08 has sessions 01–10 and 21–29: inspect acquisition changes across that gap before treating them as one condition.
4. Compare a simple bandpower/CSP baseline with EEGNet. Change final layer to three logits, use cross-entropy and train-only scaling/class weights. Tune early stopping, regularization and deployed smoothing/rejection only on tuning data. Equalize trial influence so long cues do not dominate solely through more windows.
5. Freeze weights, normalization, class order and the full decision policy. Evaluate held-out sessions once, with macro recall/balanced accuracy (chance 1/3), macro F1, per-class recall/precision, confusion matrix, rest false activation, rejection coverage and latency. Report trial/run-level uncertainty and a permutation test that respects independent runs/trials; overlapping windows are not independent samples.

## Stage 2 — prospective personal multiclass calibration

Keep this separate from binary code until Stage 1 is reproducible. Extend model contracts, protocol cues, GUI legends/results and packet class IDs together; the current binary contract deliberately rejects multiclass weights.

Collect balanced randomized imagery of the same three actions on the user. Plan several blocks with rest breaks, initially ~30 clean cue trials per class across fitting/tuning blocks, and an independent frozen validation block with ≥20 per class if time allows. Establish feasibility/quality first; these counts are planning targets rather than guaranteed adequacy. Keep overt execution, imagery and FES-assisted recordings distinct. Use new held-out cues rather than reporting calibration success.

For deployment, use softmax probabilities with a tuned confidence/margin requirement and explicit neutral/uncertain output; argmax alone always chooses something. Tune EMA/dwell through replay on tuning data using exactly the intended live cadence. UDP schema needs a version, class ID/name, confidence, quality and neutral state; the hand receiver must implement each gesture and timeout. Start with visual feedback. No FES action follows automatically from this plan.

## Stage 3 — expand toward nine

Add protocol gestures after reviewing training/tuning confusion and collecting adequate per-person trials. Preserve a new held-out test when changing class set. Fine-grained grasps/wrist imagery may remain confusable even when rest/action works; binary success does not guarantee nine-way success. Report full matrices and failure modes. If three classes fail prospectively, investigate signal/labels and improve within-person data before adding more.

## Schedule before Thanksgiving, November 26, 2026

| Dates | Deliverable / decision |
|---|---|
| Nov 3–6 | Finish binary release; lab SDK/LSL/montage preflight and first personal calibration |
| Nov 7–10 | Frozen binary self-test in repeated blocks; preserve evidence, decide whether signal supports proceeding |
| Nov 11–15 | Eligible multiclass EDF inventory/QA, fixed three-class split, baseline and EEGNet comparison |
| Nov 16–20 | Three-class replay and GUI/protocol/receiver changes if offline evidence supports them |
| Nov 21–24 | Prospective personal three-class fit and independent frozen tests; collect more if necessary |
| Nov 25 | Freeze results, limitations and reproducible report; reserve nine-class expansion if not yet supported |

The deadline is an evidence/reporting milestone, not a reason to reuse test data or claim unsupported accuracy. Hardware access and sufficient clean personal data determine whether the prospective stages fit this schedule.
