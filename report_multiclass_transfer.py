"""Report more-class feasibility, held-person transfer and fitting diagnostics."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def report(work,output):
    work=Path(work);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    study=json.loads((work/'study_results.json').read_text());manifest=json.loads((work/'cohort_manifest.json').read_text())
    results=study['results'];config=study['config'];cross=[r for r in results if r['held_subject']];within=[r for r in results if not r['held_subject']]
    primary=next(r for r in cross if r['held_subject']==config['primary_held_subject'] and r['n_classes']==9)
    recovered=sum(s['recovered_event_files'] for s in manifest['subjects'].values())
    summary=dict(study,quality=dict(edf_runs=len(manifest['runs']),failures=manifest['failures'],
        recovered_event_tables=recovered,subjects=manifest['subjects'],declared_exclusions=manifest['declared_exclusions'],
        excluded_sessions=config.get('excluded_session_reasons',{})),
        interpretation='Above-chance feasibility does not establish reliable control; large training/validation gaps remain. Three IDs only.')
    (output/'multiclass_cohort_results.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    x=np.arange(3);counts=[3,5,9];width=.24
    for j,held in enumerate(config['held_subject_order']):
        scores=[next(r for r in cross if r['held_subject']==held and r['n_classes']==k)['neural']['cue_trials']['balanced_accuracy'] for k in counts]
        bars=axes[0].bar(x+(j-1)*width,np.array(scores)*100,width,label=f'Held ID {held}')
        axes[0].bar_label(bars,fmt='%.1f',fontsize=8)
    axes[0].plot(x,[100/k for k in counts],'k--o',label='Chance')
    axes[0].set_xticks(x,[f'{k} classes' for k in counts]);axes[0].set_ylabel('Cue balanced accuracy (%)');axes[0].set_ylim(0,105)
    axes[0].set_title('Train on two people; test the third\nNo target-person fitting or normalization',fontsize=11);axes[0].legend(fontsize=8)
    x=np.arange(2)
    for j,cohort in enumerate(config['within_person_splits']):
        scores=[next(r for r in within if r['case']==f'within_08_{cohort}_{name}')['neural']['cue_trials']['balanced_accuracy'] for name in ['five','nine']]
        bars=axes[1].bar(x+(j-.5)*.32,np.array(scores)*100,.32,label=f'ID08 {cohort} cohort');axes[1].bar_label(bars,fmt='%.1f',fontsize=9)
    axes[1].plot(x,[20,100/9],'k--o',label='Chance');axes[1].set_xticks(x,['5 classes','9 classes']);axes[1].set_ylim(0,105)
    axes[1].set_title('Within-person held sessions\nRetrospective exploratory results',fontsize=11);axes[1].set_ylabel('Cue balanced accuracy (%)');axes[1].legend(fontsize=8)
    fig.suptitle('More classes and cross-person feasibility · fixed neural covariance model',fontsize=13)
    fig.savefig(output/'multiclass_transfer_overview.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for row,name in enumerate(['transfer_hold_07_nine','within_08_early_nine']):
        item=next(r for r in results if r['case']==name);hist=next(s['epochs'] for s in item['training_diagnostics']['seed_histories'] if s['seed']==42)
        ax=axes[row,0];diag=[h for h in hist if 'train_cue_balanced_accuracy' in h]
        ax.plot([h['epoch'] for h in diag],[100*h['train_cue_balanced_accuracy'] for h in diag],'-o',label='Training cues',markersize=3)
        ax.plot([h['epoch'] for h in hist],[100*h['validation_cue_balanced_accuracy'] for h in hist],label='Source tuning cues')
        best=max(hist,key=lambda h:(h['validation_cue_balanced_accuracy'],h['validation_min_class_recall'],-h['validation_cue_loss']))
        ax.axvline(best['epoch'],color='black',linestyle='--',alpha=.6,label=f"Selected epoch {best['epoch']}")
        ax.axhline(100/9,color='gray',linestyle=':',label='Chance');ax.set_ylim(0,105);ax.set_xlabel('Epoch');ax.set_ylabel('Cue balanced accuracy (%)')
        ax.set_title(name.replace('_',' ')+' · seed42',fontsize=10);ax.legend(fontsize=8)
        curve=next(c for c in study['learning_curves'] if c['case']==name);points=curve['points'];ax=axes[row,1]
        ax.plot([p['train_cues'] for p in points],[100*p['train_balanced_accuracy'] for p in points],'-o',label='Training')
        ax.plot([p['train_cues'] for p in points],[100*p['validation_balanced_accuracy'] for p in points],'-o',label='Source tuning')
        ax.axhline(100/9,color='gray',linestyle=':');ax.set_ylim(0,105);ax.set_xlabel('Independent training cues');ax.set_ylabel('Cue balanced accuracy (%)')
        ax.set_title('Whole-cue learning curve · no target scoring',fontsize=10);ax.legend(fontsize=8)
    fig.suptitle('Fitting diagnostics: training success does not imply session/person generalization',fontsize=13)
    fig.savefig(output/'multiclass_fitting_diagnostics.png',dpi=180);plt.close(fig)
    labels=['Rest','Volar','Extend','Opponens','W.Flex','W.Ext','Pron.','Supin.','Pinch'];cm=np.asarray(primary['neural']['cue_trials']['confusion_matrix'])
    norm=cm/cm.sum(axis=1,keepdims=True);fig,ax=plt.subplots(figsize=(7.5,7.2),layout='constrained');im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues')
    for i in range(9):
        for j in range(9):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=9,color='white' if norm[i,j]>.5 else 'black')
    ax.set_xticks(range(9),labels,rotation=45,ha='right');ax.set_yticks(range(9),labels);ax.set_xlabel('Predicted gesture');ax.set_ylabel('Recorded gesture')
    ax.set_title(f"Primary nine-class transfer: 04+08 → 07\nCue balanced accuracy {primary['neural']['cue_trials']['balanced_accuracy']:.1%}; chance 11.1%")
    fig.colorbar(im,ax=ax,label='Recall fraction in each true class');fig.savefig(output/'multiclass_transfer_nine_confusion.png',dpi=180);plt.close(fig)
    def percentage(value):return f'{100*value:.1f}%'
    cross_rows='\n'.join(f"| {r['n_classes']} | {r['held_subject']} | {percentage(r['neural']['cue_trials']['balanced_accuracy'])} | {percentage(r['linear_covariance_baseline']['cue_trials']['balanced_accuracy'])} | {percentage(r['chance_balanced_accuracy'])} | {r['test_cue_count']} |" for r in cross)
    within_rows='\n'.join(f"| {r['case']} | {percentage(r['training_diagnostics']['train']['cue_trials']['balanced_accuracy'])} | {percentage(r['training_diagnostics']['validation']['cue_trials']['balanced_accuracy'])} | {percentage(r['neural']['cue_trials']['balanced_accuracy'])} | {percentage(r['linear_covariance_baseline']['cue_trials']['balanced_accuracy'])} |" for r in within)
    gap_rows='\n'.join(f"| {r['case']} | {percentage(r['training_diagnostics']['train']['cue_trials']['balanced_accuracy'])} | {percentage(r['training_diagnostics']['validation']['cue_trials']['balanced_accuracy'])} | {percentage(r['neural']['cue_trials']['balanced_accuracy'])} |" for r in cross)
    subject_rows='\n'.join(f"| {sub} | {s['runs']} | {s['recovered_event_files']} | {sum(s['clean_cues'].values())} |" for sub,s in manifest['subjects'].items())
    pg=primary['training_diagnostics'];primary_score=primary['neural']['cue_trials']['balanced_accuracy']
    class_lines='\n'.join(f"- **{name}** ({len(classes)}): "+', '.join(x.replace('_',' ').title() for x in classes) for name,classes in config['class_sets'].items())
    text=f'''# More classes, cross-person training and fitting checks — November 3, 2026

The pipeline now trains **3, 5 and 9 output classes** and supports pooled-source training with a completely held-out dataset participant. Current results show limited transfer and large generalization gaps. They do **not** establish reliable five/nine-class or person-independent control. The separate three-class v2 recorded result (81.6% cue balanced accuracy) retains its original scope.

## Classes and the meaning of transfer

{class_lines}

The model recognizes frequency-power and inter-electrode covariance patterns. It does not require identical raw waveform shapes across people. Here, cross-subject evaluation asks whether those gesture-related features transfer without using the target person's labels, normalization statistics, covariance reference, thresholds or early-stopping results. No target alignment or supervised target calibration is included in the zero-shot scores.

## Data repair and provenance

{len(manifest['runs'])} EDF recordings passed channel/rate, event and unique-signal checks across IDs04,07,08. **{recovered} empty event tables were recoverable from native EDF annotations**. The recovered tables are separate private derivatives; source EDFs and TSVs remain unchanged. Recovery verifies known labels, positive durations and non-overlapping recorded intervals. Since native annotations are the recovered labels' source, they cannot independently corroborate themselves; physical display/acquisition timing is not externally measured here.

| Dataset ID | Included runs | Recovered event tables | Clean independent cues, all nine labels |
|---|---:|---:|---:|
{subject_rows}

Fifteen ID04/session01 recordings used the EEGMI-style custom electrode layout rather than the common study montage. They were excluded before fitting. ID08/session10/run12 was excluded as a previously known empty event file. No electrode sites or gesture labels were invented to increase the dataset.

## Cross-person results

For each outer fold, two people supply training and source-only tuning sessions; the third person is absent from all fitting. The architecture and class sets are fixed. Primary new held ID07 was scored first; ID08 had informed earlier model design, so the remaining folds and the aggregate are exploratory, not a pristine population benchmark. Only three dataset participant IDs are available.

| Classes | Held ID | Neural cue balanced accuracy | Linear covariance baseline | Chance | Eligible cue trials |
|---|---|---:|---:|---:|---:|
{cross_rows}

The primary nine-class result is **{primary_score:.2%}**, close to 11.11% chance. Small above-chance point estimates should not be treated as reliable gesture decoding. Per-run bootstrap intervals and within-run permutation results are in the aggregate JSON. These describe recordings within a held person, not uncertainty over the population of future wearers. No best subject or favorable class subset was selected after target scoring.

![Transfer and within-person overview](audit/multiclass_transfer_overview.png)

## Within-person more-class results

These use ID08's early or late session cohorts with whole-session partitions. Those recordings were already exposed in earlier development, so these are retrospective feasibility results. The fixed study uses 4-second windows, .5-second stride and a capped number of training draws per independent cue; it is not a paired replication of the tuned three-class v2 protocol.

| Case | Train cue BA | Tuning cue BA | Held-session cue BA | Linear baseline test BA |
|---|---:|---:|---:|---:|
{within_rows}

Each task is scored only on its declared labels; untrained gestures are excluded rather than relabeled as Rest. These offline cue scores do not establish per-update accuracy, prospective calibration, artifact-free neural origin or live actuator behavior.

## Overfitting and underfitting

| Transfer case | Training cue BA | Source tuning cue BA | New-person cue BA |
|---|---:|---:|---:|
{gap_rows}

For the primary nine-class model, training reached **{pg['train']['cue_trials']['balanced_accuracy']:.1%}**, source tuning **{pg['validation']['cue_trials']['balanced_accuracy']:.1%}**, and the held person **{primary_score:.1%}**. The large training/tuning gap is a strong warning about memorization/overfitting and session differences. The further held-person drop also reflects a person-domain transfer problem. The gap alone cannot identify how much each cause contributes.

Underfitting would be suggested by persistently poor training and tuning performance together, after checking optimization and data/labels. High training performance here shows that simply adding capacity is not an established solution. A poor held-person score by itself is not proof of underfitting.

Controls implemented: whole-person/session/cue separation; train-only feature statistics; equal weighting per person, class and cue; fixed three-seed averaging; early stopping using source tuning only; dropout, weight decay and covariance shrinkage; a lower-capacity linear baseline; matched inference-mode train/tuning losses; learning curves that add **independent cue trials**, not more overlapping copies. Large gaps remain despite these controls, so overfitting is **not claimed to be resolved**.

The displayed epoch/learning curves use seed42. Vertical markers show the selected checkpoints, which are restored after stopping; plotted later epochs do not replace those checkpoints. Ensemble metrics average all three selected members. Learning-curve fractions are 25%,50%,100% of whole source-training cues, with source tuning kept fixed and no held-person scoring. They are diagnostics and did not select a new target-tested algorithm. Optimization loss uses sampled windows and dropout; the separately recorded cue NLL/accuracy gives the matched inference-mode comparison.

![Training and independent-data learning curves](audit/multiclass_fitting_diagnostics.png)

![Primary nine-class transfer confusion](audit/multiclass_transfer_nine_confusion.png)

## Next development decisions

Keep per-person calibration as the deployment goal. Obtain new independent recordings before making confirmatory improvement claims; every target in this study is now exposed. More independent people/sessions, reliable imagery compliance/timing/reference checks and class-specific confusion review are needed before expecting high accuracy on fine hand/wrist distinctions. Review source tuning data to reduce model complexity or improve regularization/features; reserve new target data for a final evaluation.

An additional transfer experiment could evaluate **label-free alignment using only an initial calibration prefix**, followed by later unseen recordings. It must be reported as adaptation separately from zero-shot transfer. [He and Wu's alignment paper](https://arxiv.org/abs/1808.05464) provides a relevant method; using an entire future test recording to fit alignment would change the evaluation assumptions. No alignment performance claim is made here.

The existing three-class personal goal remains separate from a universal model. Five/nine-class live GUI cues, per-person multiclass fitting and receiver mappings still require their own integration and prospective validation. The reference decoders support the class lists; no actuator transport is enabled in this study.

## Reproduce

```powershell
python study_multiclass_transfer.py prepare --root "C:\\path\\to\\BCIFEST\\bids"
python study_multiclass_transfer.py run
python report_multiclass_transfer.py
python -m pytest -q
```

Private recovered annotations, covariance caches, full manifests, fold weights and scored/frozen artifacts are under `session_reports/multiclass/cohort_study_v1`. The aggregate report and reference models can be versioned without publishing EEG or participant records. Existing scored cases are reused as saved evidence rather than silently retrained against their test results.
'''
    Path('MULTICLASS_TRANSFER_RESULTS.md').write_text(text,encoding='utf-8')
    print('Saved transfer/more-class aggregate report and scientific figures.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--work',type=Path,default=Path('session_reports/multiclass/cohort_study_v1'))
    p.add_argument('--output',type=Path,default=Path('audit'));a=p.parse_args();report(a.work,a.output)
