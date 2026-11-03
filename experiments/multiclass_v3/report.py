"""Summarize every frozen v3 case; no model selection or training."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiclass_cohort_data import load_cohort
from multiclass_features import FeatureEnsembleRunner
from multiclass_metrics import cue_probabilities, balanced_recall
from study_multiclass_transfer import case_indices, rows_for
from train_multiclass import dump
from .models import Runner

ROOT = Path(__file__).resolve().parents[2]


def paired_interval(old_p, new_p, rows, k, repeats=3000):
    y, old, runs = cue_probabilities(old_p, rows)
    ny, new, nruns = cue_probabilities(new_p, rows)
    if not np.array_equal(y, ny) or not np.array_equal(runs, nruns):
        raise ValueError('Paired comparison requires identical cue trials.')
    old, new = old.argmax(1), new.argmax(1)
    rng = np.random.default_rng(42)
    groups = [np.flatnonzero(runs == run) for run in np.unique(runs)]
    values = []
    for _ in range(repeats):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[idx])) != k:
            continue
        values.append(balanced_recall(y[idx], new[idx], k)-balanced_recall(y[idx], old[idx], k))
    return dict(delta=balanced_recall(y, new, k)-balanced_recall(y, old, k),
                paired_run_bootstrap_delta_95ci=np.percentile(values, [2.5, 97.5]).tolist(),
                n_cues=len(y), exploratory=True)


def main():
    torch.set_num_threads(2)
    parent = ROOT/'session_reports/multiclass'
    reports = [json.loads((parent/f/'study_results.json').read_text())
               for f in ['development_v3', 'personal_v3']]
    rows = reports[0]['results']+reports[1]['results']
    results = {r['case']:r for r in rows}
    previous = json.loads((ROOT/'audit/multiclass_cohort_results.json').read_text())
    old = {r['case']:r for r in previous['results']}
    cov, records, manifest = load_cohort(parent/'cohort_study_v1')
    paired = {}
    for name, r in results.items():
        if name not in old:
            continue
        classes = r['classes']
        if name.startswith('within'):
            cohort = name.split('_')[2]
            idx, y = case_indices(records, classes, manifest['config'],
                                  within=manifest['config']['within_person_splits'][cohort])
        else:
            idx, y = case_indices(records, classes, manifest['config'], held_subject='07')
        test = idx['test']
        tr = rows_for(records, test, y)
        prior = torch.load(parent/'cohort_study_v1/cases'/name/'model.pt', weights_only=False)
        current = torch.load(parent/'development_v3'/name/'model.pt', weights_only=False)
        op = FeatureEnsembleRunner(prior).from_covariances(cov[test])
        npred = Runner(current).predict(cov[test])
        paired[name] = paired_interval(op, npred, tr, len(classes))
    public_rows = copy.deepcopy(rows)
    for row in public_rows:
        dev = row['development']
        candidates = []
        for candidate in dev['candidates']:
            history = candidate['diagnostics']
            candidates.append(dict(feature_set=candidate['feature_set'], head=candidate['head'],
                inputs=candidate['inputs'], seed=history['seed'], selected_epoch=history['selected_epoch'],
                epochs_run=len(history['history']), train_ba=history['train']['mean_person_cue_ba'],
                validation_ba=history['validation']['mean_person_cue_ba'],
                train_nll=history['train']['cue_nll'], validation_nll=history['validation']['cue_nll']))
        dev['candidates'] = candidates
        # Publish the plotted selected-seed curve; full member/candidate histories
        # remain private and reproducible, without an oversized JSON review diff.
        dev['ensemble_histories'] = [h for h in dev['ensemble_histories'] if h['seed'] == 42]
    aggregate = dict(results=public_rows, paired_comparisons=paired,
                     configs=[r['config'] for r in reports],
                     binary_changed=False, binary_reference_commit='23baacbaaf7fc60627c7de62a571a3009218f6e7',
                     interpretation='All v3 historical tests exploratory; no new prospective or confirmatory accuracy.')
    dump(ROOT/'audit/multiclass_v3_results.json', aggregate)

    # Paired historical comparisons; include negative results.
    names = ['within_08_early_five', 'within_08_early_nine',
             'within_08_late_five', 'within_08_late_nine']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for ax, ns, title in [(axes[0], names, 'Within-person ID08: same historical cues'),
                           (axes[1], [f'transfer_hold_07_{s}' for s in ['three', 'five', 'nine']],
                            'Person07 held out: no target adaptation')]:
        x = np.arange(len(ns))
        before = [old[n]['neural']['cue_trials']['balanced_accuracy']*100 for n in ns]
        after = [results[n]['historical_test']['cue_trials']['balanced_accuracy']*100 for n in ns]
        ax.bar(x-.18, before, .36, color='#8498aa', label='Previous fixed study')
        ax.bar(x+.18, after, .36, color='#24799e', label='v3 validation-selected')
        for i, (a, b) in enumerate(zip(before, after)):
            ax.text(i-.18, a+1, f'{a:.1f}', ha='center', fontsize=9)
            ax.text(i+.18, b+1, f'{b:.1f}', ha='center', fontsize=9)
        ax.axhline(70, color='#a12c3b', ls='--', label='Project target 70%')
        labels = [n.removeprefix('within_08_').replace('_', '\n').removeprefix('transfer\nhold\n07\n') for n in ns]
        ax.set_xticks(x, labels)
        ax.set_ylim(0, 79)
        ax.set_ylabel('Cue balanced accuracy (%)')
        ax.set_title(title, fontsize=12)
        ax.grid(axis='y', alpha=.15)
    axes[0].legend(loc='upper right', fontsize=8)
    fig.suptitle('Multiclass v3 — exploratory comparisons, not fresh validation', fontsize=14)
    fig.savefig(ROOT/'audit/multiclass_v3_comparison.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), constrained_layout=True)
    for ax, name in zip(axes, ['within_08_late_nine', 'transfer_hold_07_nine']):
        d = results[name]['development']
        member = next(x for x in d['ensemble_histories'] if x['seed'] == 42)
        h = member['history']
        th = [r for r in h if 'train_ba' in r]
        ax.plot([r['epoch'] for r in th], [100*r['train_ba'] for r in th], label='Train (inference mode)')
        ax.plot([r['epoch'] for r in h], [100*r['validation_ba'] for r in h], label='Validation')
        ax.axvline(member['selected_epoch'], color='gray', ls=':', label='Restored checkpoint')
        ax.set(title=name.replace('_', ' '), xlabel='Epoch', ylabel='Mean-person cue BA (%)', ylim=(0, 105))
        ax.grid(alpha=.15)
        ax.legend(fontsize=8)
    fig.suptitle('Fitting remains a concern — selected-seed development curves', fontsize=14)
    fig.savefig(ROOT/'audit/multiclass_v3_fitting.png', dpi=160)
    plt.close(fig)

    def score(r): return r['historical_test']['cue_trials']['balanced_accuracy']*100
    lines = [
        '# Multiclass v3 development and separate binary lab release — November 3, 2026', '',
        '**Binary lab software is unchanged.** All 36 protected files match commit `23baacb`. '
        'Use `Launch_BINARY_Lab.bat`, `verify_binary_release.py` and '
        '[Thursday November 5 checklist](THURSDAY_BINARY_VALIDATION.md). '
        'Its physical X24 connection and frozen new-cue self-test remain to be validated in the lab.', '',
        'Multiclass work is isolated under `experiments/multiclass_v3/`; private weights and histories '
        'are in `session_reports/multiclass/development_v3` and `personal_v3`. '
        'There is no GUI/LSL/UDP entry point in this package. Binary templates, checkpoints, configs '
        'and the earlier three-class v2 model were not overwritten.', '',
        '## What changed', '',
        'Four feature sets × two neural heads were declared for each of 14 cases: 112 candidate fits '
        'plus two additional seeds for each selected ensemble. Candidates use motor-site power, '
        'relative spatial power, motor-site tangent covariance or all-site tangent covariance, '
        'with a linear neural head or a 16-unit network. Linear heads are lower-capacity controls. '
        'Independent-cue bag loss averages sampled-window probabilities before applying one loss per cue. '
        'Each epoch uses up to four windows per cue; whole people/sessions/cues remain separated. '
        'Dropout .5, weight decay .05, label smoothing .05 and covariance shrinkage .3 were tested. '
        'Selection and early stopping use source validation only; validation selection averages '
        'participant-level balanced accuracies. No new package dependency was added.', '',
        '**Every historical test here was previously exposed.** All selected model/config/data hashes '
        'were frozen before each experiment set was scored, and no test outcome selected another '
        'candidate. These remain exploratory development comparisons, not fresh confirmation. '
        'Seed42 selects each feature/head; the fixed three-seed ensemble can have a different validation '
        'score. Re-running score reads the saved results; search refuses a finalized selection set.', '',
        '## Same-person, same-session historical comparison', '',
        '| ID08 cohort / classes | Previous cue BA | v3 cue BA | Difference, percentage points | Paired run-bootstrap 95% interval for difference |',
        '|---|---:|---:|---:|---:|']
    for n in names:
        p = paired[n]
        lo, hi = np.array(p['paired_run_bootstrap_delta_95ci'])*100
        lines.append(f"| {n.removeprefix('within_08_').replace('_', ' / ')} | {old[n]['neural']['cue_trials']['balanced_accuracy']*100:.1f}% | {score(results[n]):.1f}% | {p['delta']*100:+.1f} | [{lo:+.1f}, {hi:+.1f}] |")
    lines += ['', 'Intervals resample the same recording runs for both frozen models and preserve identical cue trials. '
              'They do not describe uncertainty over future people or cap placements. A small point-score '
              'increase is not automatically a reliable improvement. Multiple methods changed together, '
              'so these comparisons cannot attribute a difference solely to cue-bag loss or regularization.', '',
              '![Historical multiclass comparison](audit/multiclass_v3_comparison.png)', '',
              '## Additional personal calibration feasibility', '',
              '| Participant / classes | Train cue BA | Tuning cue BA | Later-session cue BA | Test cues | Selected feature / head |',
              '|---|---:|---:|---:|---:|---|']
    for n in ['personal_04_five', 'personal_04_nine', 'personal_07_five', 'personal_07_nine']:
        r = results[n]; d = r['development']
        lines.append(f"| {n.removeprefix('personal_').replace('_', ' / ')} | {d['train']['mean_person_cue_ba']*100:.1f}% | {d['validation']['mean_person_cue_ba']*100:.1f}% | {score(r):.1f}% | {r['historical_test']['cue_trials']['n']} | {d['selected_feature']} / {d['head']} |")
    lines += ['', 'ID04: session2 train,3 tune,4–5 test. ID07: sessions1–6 train,7–8 tune,9–10 test. '
              'These are different people/splits from ID08, so they are not paired improvements over it. '
              'All task classes must be present in each partition; other gestures are excluded from '
              'closed-set scoring rather than renamed Rest.', '',
              '## Cross-person transfer and prefix adaptation', '',
              '| Classes | Previous zero-shot, all ID07 sessions | v3 zero-shot, all ID07 sessions | v3 zero-shot, sessions3–10 only | Prefix-EA adaptation, sessions3–10 only |',
              '|---|---:|---:|---:|---:|']
    for s in ['three', 'five', 'nine']:
        n = f'transfer_hold_07_{s}'
        r, adapted = results[n], results[f'prefix_ea_hold_07_{s}']
        later = r['same_later_sessions_without_adaptation']['cue_trials']['balanced_accuracy']*100
        lines.append(f"| {s} | {old[n]['neural']['cue_trials']['balanced_accuracy']*100:.1f}% | {score(r):.1f}% | {later:.1f}% | {score(adapted):.1f}% |")
    lines += ['', 'The two right columns use the same later-session cues. EA uses source-person training '
              'references and an unlabeled target prefix from sessions1–2. ALL accepted prefix gesture '
              'windows are included irrespective of the scored class subset; no target gesture labels '
              'fit the reference. Cue boundaries and signal-quality exclusions are available. '
              'Reference hashes are saved before predicting any later-session sample. '
              'The EA and non-EA arms select their own source-validation candidate, so this compares '
              'complete pipelines rather than isolating alignment alone. These results do not establish '
              'successful zero-shot or adapted control. Chance BA is 33.3%,20%,11.1% for 3/5/9 classes.', '',
              'The method is motivated by [He and Wu’s Euclidean alignment paper](https://arxiv.org/abs/1808.05464). '
              'Published improvements are not assumed to transfer to these recordings; future target '
              'samples never fit our prefix reference.', '',
              '## Overfitting, underfitting and limitations', '',
              '| Case | Train cue BA | Source tuning cue BA | Historical target cue BA | Selected feature / head |',
              '|---|---:|---:|---:|---|']
    for n in ['within_08_late_nine', 'transfer_hold_07_three', 'transfer_hold_07_nine']:
        r = results[n]; d = r['development']
        lines.append(f"| {n} | {d['train']['mean_person_cue_ba']*100:.1f}% | {d['validation']['mean_person_cue_ba']*100:.1f}% | {score(r):.1f}% | {d['selected_feature']} / {d['head']} |")
    lines += ['', 'Large train/tuning gaps remain for several models. Reducing a gap by lowering training '
              'accuracy does not by itself improve generalization. Some power-only candidates fit both '
              'training and tuning poorly, suggesting insufficient features/capacity, optimization limits '
              'or weak/variable task signal; this is not proof of a single cause. Low target accuracy '
              'alone does not diagnose underfitting. Inspect selected-epoch histories as well as both '
              'matched inference-mode losses; dropout training loss is a different quantity.', '',
              '![Selected-seed fitting curves](audit/multiclass_v3_fitting.png)', '',
              'No five/nine-class case reaches 70%. The preserved three-class v2 result remains '
              '81.6% cue BA on its specific historical within-person protocol, with 53.8% Volar Grasp '
              'recall and 63.9% individual-window BA; it does not certify this expanded task. '
              'Four-second windows, .5-second extraction stride and cue averaging here do not establish '
              'live per-update control. Physical imagery compliance, eye/muscle artifacts, reference '
              'and display/acquisition timing still need measurement; classifier accuracy alone does '
              'not prove a neural motor-imagery origin.', '',
              '## Next steps', '',
              '1. Complete Thursday’s unchanged binary calibration and two new frozen validation blocks. '
              'Save raw EEG, events, predictions, model/threshold and quality/exclusion counts.',
              '2. For multiclass development, inspect source-training/tuning class confusions and '
              'recording quality. Keep all historical tests as exposed evidence; do not chase their '
              'scores by repeatedly selecting variants.',
              '3. Plan a separate prospective 3→5-class session before attempting all nine. '
              'Use randomized independent cue blocks, measured event timing and a separate later '
              'validation block. More independent trials/sessions are required; no fixed trial count '
              'guarantees 70%. The personal multiclass GUI/receiver still requires implementation '
              'and prospective testing before that session.', '',
              '## Reproduce and inspect', '',
              'See [experimental package instructions](experiments/multiclass_v3/README.md). '
              '`audit/multiclass_v3_results.json` preserves all candidates/selected-seed histories and '
              'aggregate scores. Model bytes, source/prefix references and frozen artifacts remain '
              'private under `session_reports/multiclass/`. Original recordings/annotation files are '
              'unchanged. The binary integrity manifest checks files, not the installed environment '
              'or live hardware. Software tests are recorded separately in the release notes.', '']
    (ROOT/'MULTICLASS_V3_RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    print('Generated v3 aggregate, comparison/fitting figures and report.', flush=True)


if __name__ == '__main__':
    main()
