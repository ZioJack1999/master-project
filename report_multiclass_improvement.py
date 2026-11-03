"""Publish aggregate v2 evidence without raw EEG or private cue predictions."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def report(work,output):
    work=Path(work);selected=work/'selected';output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result=json.loads((selected/'test_results.json').read_text());selection=json.loads((work/'selection.json').read_text())
    manifest=json.loads((selected/'manifest.json').read_text());calibration=json.loads((selected/'class_bias_calibration.json').read_text())
    tuning=json.loads((selected/'policy_tuning.json').read_text());contract=json.loads((selected/'eegnet_multiclass.pt.json').read_text())
    runtime_path=output/'multiclass_v2_runtime.json'
    runtime=json.loads(runtime_path.read_text()) if runtime_path.exists() else None
    candidates=[{k:v for k,v in row.items() if k!='seed_histories'} for row in selection['candidates']]
    quality={}
    for part in ['train','validation','test']:
        counts=Counter()
        for run in manifest['runs']:
            if run['partition']==part:counts.update(run['counts'])
        quality[part]=dict(counts)
    summary=dict(result,selection={k:v for k,v in selection.items() if k!='candidates'},candidates=candidates,
        class_bias_calibration=dict(before=calibration['before'],selected=calibration['selected'],test_used=False),
        policy_tuning=tuning,quality=quality,declared_exclusions=manifest.get('declared_exclusions',[]),
        window_s=contract['n_times']/contract['sfreq'],
        runtime=runtime,target_scope='70% cue-level balanced accuracy on eligible trials. Per-update and every-class targets are separate.')
    (output/'multiclass_v2_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    nn=result['eegnet']['cue_trials'];bp=result['bandpower_baseline']['cue_trials'];live=result['continuous_replay']['cue_trials']
    panels=[('Neural covariance ensemble',nn['confusion_matrix'],nn['balanced_accuracy']),
            ('Bandpower baseline',bp['confusion_matrix'],bp['balanced_accuracy']),
            ('Continuous neural replay',live['confusion_matrix'],live['balanced_accuracy'])]
    labels=['Rest','Volar grasp','Finger extension'];fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    for ax,(title,cm,score) in zip(axes,panels):
        cm=np.asarray(cm);norm=cm/cm.sum(axis=1,keepdims=True);cols=labels+(['Uncertain'] if cm.shape[1]==4 else [])
        ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(cm.shape[1]):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if norm[i,j]>.5 else 'black')
        ax.set_xticks(range(len(cols)),cols,rotation=30,ha='right');ax.set_yticks(range(3),labels)
        ax.set_xlabel('Predicted');ax.set_ylabel('Recorded cue');ax.set_title(f'{title}\nCue balanced accuracy {score:.1%}',fontsize=12)
    fig.suptitle('Within-person recorded test · BCIFEST participant 08 · sessions 9–10\n4-second windows; model, offsets and policy frozen before test',fontsize=13)
    fig.savefig(output/'multiclass_v2_confusion.png',dpi=170);plt.close(fig)
    interval=nn['run_cluster_bootstrap_95ci'];q=quality['test'];classes=result['classes']
    method_table='\n'.join(f"| {row['name']} | {row['validation']['cue_trials']['balanced_accuracy']:.2%} | {min(row['validation']['cue_trials']['recall'].values()):.2%} |" for row in candidates)
    retained=sum(manifest['partitions']['test']['clean_cue_trials'].values())
    runtime_text=f"Audit-PC decoder computation: median {runtime['chunk_ms_median']:.2f} ms, 95th percentile {runtime['chunk_ms_p95']:.2f} ms across {runtime['n_outputs']} outputs. This excludes acquisition/display latency and the four-second window accumulation." if runtime else ''
    replay_cm=np.asarray(live['confusion_matrix']);skipped=live['skipped_cue_trials']
    sensitivity=[]
    for label in range(3):
        counts=replay_cm.sum(axis=1).copy();counts[label]+=skipped
        sensitivity.append(float(np.mean(np.diag(replay_cm[:,:3])/counts)))
    text=f'''# Multiclass v2: the 70% cue-level target

The selected neural model achieved **{nn['balanced_accuracy']:.2%} balanced accuracy on {nn['n']} eligible held-out cue trials**. Continuous replay achieved **{live['balanced_accuracy']:.2%} on {live['n_cue_trials']} scored cues**, with {live['skipped_cue_trials']} additional cue(s) unscored because too few clean sample-clocked windows were available. These point estimates exceed the requested 70% cue-level target. This is recorded within-person evidence; prospective personal validation is still needed.

| Held-out measure | Result |
|---|---:|
| Calibrated neural probabilities averaged per cue | {nn['balanced_accuracy']:.2%} balanced accuracy |
| Continuous neural replay, confidence/margin gating | {live['balanced_accuracy']:.2%} cue balanced accuracy |
| Fixed bandpower baseline on the same eligible cues | {bp['balanced_accuracy']:.2%} cue balanced accuracy |
| Individual neural windows | {result['eegnet']['windows']['balanced_accuracy']:.2%} balanced accuracy |
| Rest recall | {nn['recall']['rest']:.2%} |
| Volar Grasp recall | {nn['recall']['volar_grasp']:.2%} |
| Finger Extension recall | {nn['recall']['finger_extension']:.2%} |

The overall cue target is met; **70% on every class or every individual update is not met**. Volar Grasp remains the weak class. The neural cue score's exploratory run-cluster bootstrap 95% interval is {interval[0]:.1%}–{interval[1]:.1%}; within-run permutation p={nn['within_run_permutation_p']:.5f}. Only two test sessions are available, so this interval is not a guarantee for another recording or wearer.

As a post-test reporting sensitivity check, assigning the {skipped} unscored replay cue(s) as incorrect to the most adverse class lowers the replay score to {min(sensitivity):.2%}. This check changes no model, offset or policy parameters. It still excludes the incomplete cues too short to contain the required window.

## What changed

The first raw-waveform EEGNet was replaced, in this separate multiclass engine, by a **train-referenced log-covariance feature model with three small neural networks averaged across fixed seeds**. Four frequency bands (8–12, 12–16, 16–22 and 22–30 Hz) preserve both power and inter-electrode relationships. Covariance shrinkage, channel scales, reference matrices and feature normalization use training data only. The 32-unit neural heads use dropout and weight decay; complete cue trials receive equal total sampling weight within each class.

This structure is motivated by spectral/spatial EEG decoding in [Schirrmeister et al.](https://pubmed.ncbi.nlm.nih.gov/28782865/) and covariance-based multiclass BCI in [Barachant et al.](https://hal.science/hal-00681328v1/document). This implementation uses a train-only arithmetic covariance reference followed by a matrix logarithm, with a neural head.

Eight declared candidates covered two neural feature families, all20 versus motor10 channels, and 2 versus 4-second windows. All three fixed seeds (13,42,73) were averaged; no best-seed selection occurred. Window candidates were compared on **the same {selection['selection_common_validation_cues']} eligible tuning cues**, preventing a longer window from winning just by discarding different difficult cues.

Selected: **{selection['selected_candidate']}**. Its uncalibrated tuning score was {calibration['before']['cue_trials']['balanced_accuracy']:.2%}. After observing development class imbalance, bounded class log-offsets were fitted only on tuning data, before the test: {contract['class_bias_calibration']['bias']}. This increased the tuning score to {calibration['selected']['metrics']['cue_trials']['balanced_accuracy']:.2%}. Offsets and the replay policy were frozen before scoring the test; the calibration step was an adaptive development decision, not a pre-test performance result.

Replay selected EMA alpha {contract['policy']['ema_alpha']}, dwell {contract['policy']['dwell_updates']} update(s), confidence {contract['policy']['min_confidence']} and margin {contract['policy']['min_margin']} from tuning recordings. Alpha 1 means probabilities are unsmoothed. Four-second windows update every .25 seconds (93.75% overlap) and need four seconds before the first output.

{runtime_text}

## Data separation and coverage

Participant 08, BCIFEST task olrandmcmitrain: sessions **1–6 train**, **7–8 tune**, **9–10 test**. The earlier test sessions28–29 stayed exposed historical/development evidence. The selected v2 test signals were checked against the v1 signal hashes with no overlap. The two experiments use different cohorts and test sets; their percentages are not a paired estimate of the effect of changing the model.

The selected model retained {sum(manifest['partitions']['train']['clean_cue_trials'].values())} training cues, {sum(manifest['partitions']['validation']['clean_cue_trials'].values())} tuning cues and {retained} test cues. Test quality accounting: {q.get('selected_cues',0)} selected-class cues before eligibility; {q.get('short_cues',0)} too short for the window/settling; {q.get('artifact_windows',0)} artifact windows and {q.get('flat_windows',0)} flat windows rejected. The manifest keeps exact per-run exclusions. One empty event file (session10/run12), already identified in the inventory, was explicitly excluded before fitting; no labels were invented.

Replay accepted {live['known_class_update_coverage']:.1%} of clean known-class updates and {live['other_gesture_acceptance_rate']:.1%} of clean updates from the six untrained gestures. **This three-class model does not establish unknown-gesture rejection or nine-class control.** Reported accuracy excludes other gestures rather than calling them rest.

![Held-out multiclass v2 results](audit/multiclass_v2_confusion.png)

## Development comparison, before class-offset calibration

| Candidate | Common tuning cue balanced accuracy | Lowest class recall |
|---|---:|---:|
{method_table}

## Reproduce

```powershell
python improve_multiclass.py prepare --root "C:\\path\\to\\BCIFEST\\bids"
python improve_multiclass.py search
python calibrate_multiclass.py
python replay_multiclass.py --work session_reports/multiclass/sub08_three_class_v2/selected --root "C:\\path\\to\\BCIFEST\\bids"
python train_multiclass.py test --work session_reports/multiclass/sub08_three_class_v2/selected --root "C:\\path\\to\\BCIFEST\\bids"
python report_multiclass_improvement.py
```

The scored v2 experiment is now closed to tuning. Preserve its saved results; changing parameters requires unused future test data. Local raw windows, candidate weights and full manifests remain under ignored session_reports. Released reference weights are `decoder_and_models/multiclass_covariance_sub08_v2.pt` with an adjacent contract. The binary GUI's personal calibration flow is separate; prospective multiclass GUI calibration/receiver integration remains future work. Multiclass replay sends no actuator commands.
'''
    Path('MULTICLASS_IMPROVEMENT_RESULTS.md').write_text(text,encoding='utf-8')
    print('Saved multiclass v2 aggregate results, figure and report.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,default=Path('session_reports/multiclass/sub08_three_class_v2'))
    p.add_argument('--output',type=Path,default=Path('audit'));a=p.parse_args();report(a.work,a.output)
