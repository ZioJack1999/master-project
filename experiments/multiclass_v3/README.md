# Multiclass development v3 — separate offline research

**Do not use this folder for Thursday's binary lab validation.** Use [the binary checklist](../../THURSDAY_BINARY_VALIDATION.md) and `Launch_BINARY_Lab.bat` from the repository root. The binary code, configs and model hashes are independently checked by `verify_binary_release.py`.

This package investigates 3/5/9-class recorded BCIFEST data. It imports no GUI, LSL acquisition or UDP transport. Models and individual recording-derived artifacts are saved privately under `session_reports/multiclass/`. No outputs replace binary checkpoints or the earlier three-class v2 reference.

## Changes tested

- Independent-cue training loss: average probabilities across up to four sampled windows from each cue, then compute one loss per cue. Overlap is never counted as extra independent trials.
- Eight declared candidates: motor-site log power, relative spatial log power, motor-site tangent covariance and all-site tangent covariance; each with a linear neural head or a 16-unit network. Linear heads are a lower-capacity comparison, not a deep NN.
- Stronger regularization: dropout .5 for hidden networks, weight decay .05, label smoothing .05 and covariance shrinkage .3. Training/validation curves and best-epoch restoration remain explicit.
- Validation selection averages participant-level cue balanced accuracy rather than letting the largest source participant dominate. Test people/sessions remain excluded from fitting and selection.
- A separate transfer arm fits Euclidean alignment from source training only. Target alignment uses all accepted gesture windows from the earlier ID07 sessions1–2, ignores gesture labels, and evaluates later sessions3–10. Cue boundaries and signal-quality exclusions are still available. The future evaluated recording never fits alignment. This is **prefix adaptation**, not zero-shot transfer. The nonadapted model is additionally scored on exactly the same later-session subset.
- Additional personal-only cases use ID04 session2/train,3/tune,4–5/test and ID07 sessions1–6/train,7–8/tune,9–10/test. They examine known-person calibration feasibility; different participants/splits are not interchangeable before/after comparisons.

All historical participants/session cohorts were already exposed in earlier work. These are exploratory comparisons. A new score cannot be presented as untouched confirmation or prospective wearer accuracy. No historical test result selects another candidate in this search. Further changes require a separately recorded development decision and ultimately new validation recordings.

## Run from the repository root

Install the existing repository requirements; no new dependency is required. First prepare the common private cache with `study_multiclass_transfer.py prepare` as documented in the prior transfer report. Then:

```powershell
python -m experiments.multiclass_v3.study search
python -m experiments.multiclass_v3.study score
python -m experiments.multiclass_v3.study search --config experiments/multiclass_v3/personal_experiment.json
python -m experiments.multiclass_v3.study score --config experiments/multiclass_v3/personal_experiment.json
python -m experiments.multiclass_v3.report
```

Each search freezes all selected model/development/config/data hashes before its score stage. Searches refuse to reopen a finalized selection set; score reads saved results on repeat. Do not delete frozen artifacts to chase a better score. `Runner` loads these private models for covariance-input research; they have no prospective GUI policy or actuator integration.

The alignment idea follows [He and Wu](https://arxiv.org/abs/1808.05464); its success on published datasets does not establish an improvement here. Our stricter past-prefix evaluation uses no future target statistics.

See [v3 results](../../MULTICLASS_V3_RESULTS.md) and the earlier [transfer study](../../MULTICLASS_TRANSFER_RESULTS.md). Raw EEG, recovered annotation tables, per-recording predictions and personal calibration weights remain excluded from Git.
