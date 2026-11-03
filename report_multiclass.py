"""Generate aggregate multiclass evidence and scientific confusion figures."""
import argparse,json
from pathlib import Path
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def report(work,output):
    work=Path(work);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result=json.loads((work/'test_results.json').read_text());training=json.loads((work/'training.json').read_text())
    manifest=json.loads((work/'manifest.json').read_text());policy=json.loads((work/'policy_tuning.json').read_text())
    qa={}
    for name in ['train','validation','test']:
        counts=Counter()
        for run in manifest['runs']:
            if run['partition']==name:counts.update(run['counts'])
        qa[name]=dict(counts)
    summary=dict(result,training=dict(best_epoch=training['best_epoch'],history=training['history'],
        tuning=training['tuning'],elapsed_s=training['elapsed_s']),
        policy_tuning=policy,quality=dict(runs_checked=len(manifest['runs']),failures=manifest['failures'],counts=qa),
        experiment=manifest['config'])
    (output/'multiclass_three_class_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    classes=['Rest','Volar grasp','Finger extension']
    panels=[('EEGNet: mean probability per cue',result['eegnet']['cue_trials']['confusion_matrix'],result['eegnet']['cue_trials']['balanced_accuracy'],classes),
            ('Bandpower baseline: per cue',result['bandpower_baseline']['cue_trials']['confusion_matrix'],result['bandpower_baseline']['cue_trials']['balanced_accuracy'],classes),
            ('EEGNet: continuous replay + rejection',result['continuous_replay']['cue_trials']['confusion_matrix'],result['continuous_replay']['cue_trials']['balanced_accuracy'],classes+['Uncertain'])]
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    for ax,(title,cm,score,columns) in zip(axes,panels):
        cm=np.array(cm);norm=cm/cm.sum(axis=1,keepdims=True)
        ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
        for i in range(3):
            for j in range(len(columns)):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if norm[i,j]>.5 else 'black')
        ax.set_xticks(range(len(columns)),columns,rotation=30,ha='right');ax.set_yticks(range(3),classes)
        ax.set_xlabel('Predicted');ax.set_ylabel('Recorded cue');ax.set_title(f'{title}\nBalanced accuracy {score:.1%}',fontsize=11)
    fig.suptitle('BCIFEST participant 08 · sessions 28–29 held out · 3-class subset\nRecorded-data evidence; prospective personal validation pending',fontsize=13)
    fig.savefig(output/'multiclass_three_class_confusion.png',dpi=170);plt.close(fig)
    nn=result['eegnet']['cue_trials'];bp=result['bandpower_baseline']['cue_trials'];live=result['continuous_replay']['cue_trials']
    text=f'''# First three-class experiment — November 3, 2026

Classes: **Rest / Volar Grasp / Finger Extension** from BCIFEST, participant 08.
Train sessions **21–25**, tune **26–27**, held-out test **28–29**. Split and class subset were fixed before fitting. This is within-person recorded-data evaluation; physical movement/EMG and the acquisition reference have not been independently verified.

| Method | Held-out cue balanced accuracy | Cue count |
|---|---:|---:|
| EEGNet, mean raw probability per cue | {nn['balanced_accuracy']:.2%} | {nn['n']} |
| Fixed bandpower logistic baseline | {bp['balanced_accuracy']:.2%} | {bp['n']} |
| EEGNet continuous replay, EMA/dwell/rejection | {live['balanced_accuracy']:.2%} | {live['n_cue_trials']} |

Chance is 33.33%. Raw EEGNet window balanced accuracy is {result['eegnet']['windows']['balanced_accuracy']:.2%}; overlapping windows are not independent trials. Run-cluster bootstrap 95% interval for EEGNet cue balanced accuracy: {nn['run_cluster_bootstrap_95ci'][0]:.1%}–{nn['run_cluster_bootstrap_95ci'][1]:.1%}. Within-run permutation p={nn['within_run_permutation_p']:.5f}. These quantify this small recorded test; only two independent test sessions are available.

## Quality and fitting

All {len(manifest['runs'])} selected EDF/event pairs passed timing/annotation, rate, montage and duplicate-signal checks. Windows use volts, continuous causal 8–30 Hz filtering, 2-second duration, .25-second stride, .5-second settling, and at least four clean predictions per retained cue. Artifacts above 100 µV in the filtered signal and near-flat channels are rejected.

Training has {sum(manifest['partitions']['train']['clean_cue_trials'].values())} clean cues; tuning has {sum(manifest['partitions']['validation']['clean_cue_trials'].values())}; the raw-window test has {sum(manifest['partitions']['test']['clean_cue_trials'].values())}. Each class/cue receives equal total sampling weight during NN fitting. Normalization uses training data only. Best epoch {training['best_epoch']} was selected using tuning cue accuracy, with tuning loss breaking ties. No test-based model selection occurred.

Replay keeps filter/EMA history through every recorded cue, including the six untrained gesture types. Its inference windows align to the continuous sample clock, so eligible-window/cue counts can differ from onset-aligned training windows. Rejected outputs count as incorrect in balanced recall and have a separate confusion column. Policy parameters were selected on tuning sessions only and frozen before the test.

Replay accepted {live['known_class_update_coverage']:.1%} of clean known-class updates. It also accepted {live['other_gesture_acceptance_rate']:.1%} of clean updates from other protocol gestures. **A three-class softmax model is not a validated detector for unknown gestures.** The six other gestures were excluded from the three-class accuracy calculation, and their acceptance is reported separately rather than relabeled as rest.

![Held-out confusion matrices](audit/multiclass_three_class_confusion.png)

## Reproduce

Released reference weights and baseline parameters: `decoder_and_models/eegnet_bcifest_sub08_three_class.pt` (with adjacent contract) and `decoder_and_models/bandpower_bcifest_sub08_three_class.json`.

Run each stage in order, with the studio Python environment:

```powershell
python train_multiclass.py prepare --root "C:\\path\\to\\BCIFEST\\bids"
python train_multiclass.py fit
python replay_multiclass.py --root "C:\\path\\to\\BCIFEST\\bids"
python train_multiclass.py test --root "C:\\path\\to\\BCIFEST\\bids"
python report_multiclass.py
```

Raw windows, per-run manifests, baseline parameters and trained artifacts initially remain in ignored `session_reports/multiclass/sub08_three_class_v1`. Hashes freeze model, contract, prepared arrays and provenance before testing. Test scoring is one-time per work folder; subsequent invocations read the saved report or fail rather than silently rerun training. A changed model requires a newly declared experiment and unused test data.

## What this enables next

The first NN shows above-chance recorded discrimination, but Finger Extension recall is only {nn['recall']['finger_extension']:.1%}, and the baseline has higher cue balanced accuracy on this test. This test is now exposed; do not use it to choose another NN architecture, seed, channel subset or threshold. Improve the approach using training/tuning data, then declare unused test data or new prospective recordings before claiming an improvement. Reliable three-class personal control and nine-way expansion are not established.

The multiclass model/trainer and replay engine work independently of the binary studio. The binary GUI still uses its own personal rest/action flow and rejects multiclass contracts. New-wearer multiclass acquisition, personal fitting, prospective GUI cues and gesture-aware receiver behavior require the next implementation/validation stage. Do not load these recorded-reference weights as a personalized model for a different wearer. No actuator output is enabled in multiclass replay.

Before expanding to nine classes, inspect this confusion matrix and establish fresh personal three-class performance. Binary rest/action accuracy cannot predict discrimination among fine-grained imagined gestures.
'''
    Path('MULTICLASS_RESULTS.md').write_text(text,encoding='utf-8')
    print('Wrote MULTICLASS_RESULTS.md and aggregate results/figure to',output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path('session_reports/multiclass/sub08_three_class_v1'))
    p.add_argument('--output',type=Path,default=Path('audit'));a=p.parse_args();report(a.work,a.output)
