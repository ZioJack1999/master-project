"""Tune multiclass smoothing/rejection on validation recordings only; no UDP."""
import argparse,json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import mne,torch
from itertools import product
from multiclass_data import cue_spans,digest
from multiclass_decoder import MulticlassDecoder,MulticlassPolicy
from multiclass_metrics import balanced_recall

def continuous_predictions(root,work,partition):
    work=Path(work);manifest=json.loads((work/'manifest.json').read_text());c=manifest['config']
    model=work/'eegnet_multiclass.pt';decoder=MulticlassDecoder(model);outputs=[];classes=c['classes']
    torch.set_num_threads(c['threads'])
    for item in manifest['runs']:
        if item['partition']!=partition:continue
        folder=Path(root)/f"sub-{int(c['subject']):02d}"/f"ses-{item['session']:02d}"/'eeg'
        edf=folder/(item['run']+'_eeg.edf');events=folder/(item['run']+'_events.tsv')
        if digest(edf)!=item['edf_sha256'] or digest(events)!=item['events_sha256']:raise ValueError('Recording changed after preparation.')
        raw=mne.io.read_raw_edf(edf,preload=False,verbose=False).get_data(picks=decoder.channels).astype(np.float32)
        spans=cue_spans(pd.read_csv(events,sep='\t'),raw.shape[1],c);decoder.reset()
        for a in range(0,raw.shape[1],decoder.step_samples):
            for p in decoder.process_chunk_all(raw[:,a:a+decoder.step_samples]):
                end=p['sample_end'];start=end-decoder.window_samples
                cue=next((r for r in spans if start>=r['start']+round(c['settle_s']*c['sfreq']) and end<=r['end']),None)
                outputs.append(dict(p,run=item['run'],label=cue['label'] if cue else None,
                    cue_name=cue['name'] if cue else None,trial=f"{item['run']}:cue-{cue['cue']}" if cue else None))
        print('replayed',partition,item['run'],flush=True)
    return outputs,classes,c

def apply_policy(rows,classes,policy):
    engine=MulticlassPolicy(len(classes),policy);result=[];run=None
    for row in rows:
        if run!=row['run']:engine.reset();run=row['run']
        result.append(dict(row,**engine.update(row['probabilities_raw'],row['quality'])))
    return result

def policy_metrics(rows,classes):
    k=len(classes);eligible=[r for r in rows if r['label'] is not None and r['label']>=0]
    cm=np.zeros((k,k+1),int);skipped=0
    for trial in sorted(set(r['trial'] for r in eligible)):
        all_rows=[r for r in eligible if r['trial']==trial];clean=[r for r in all_rows if r['quality']=='ok']
        if len(clean)<4 or len(clean)<.5*len(all_rows):skipped+=1;continue
        # Majority over all eligible updates; reject/artifact updates stay uncertain.
        votes=Counter(r['class_id'] for r in all_rows);maximum=max(votes.values())
        winners=[label for label,n in votes.items() if n==maximum];pred=winners[0] if len(winners)==1 else -1
        cm[all_rows[0]['label'],pred if pred>=0 else k]+=1
    if np.any(cm.sum(axis=1)==0):raise ValueError('Replay lacks clean trials for at least one class.')
    known=[r for r in eligible if r['quality']=='ok'];unknown=[r for r in rows if r['label']==-1 and r['quality']=='ok']
    rest=classes.index('rest') if 'rest' in classes else None
    return dict(n_cue_trials=int(cm.sum()),skipped_cue_trials=skipped,
        balanced_accuracy=float(np.mean(np.diag(cm[:,:k])/cm.sum(axis=1))),
        confusion_matrix=cm.tolist(),columns=classes+['uncertain'],
        class_recall={name:float(cm[i,i]/cm[i].sum()) for i,name in enumerate(classes)},
        rest_false_activation_rate=float((cm[rest,:k].sum()-cm[rest,rest])/cm[rest].sum()) if rest is not None else None,
        known_class_update_coverage=float(np.mean([r['class_id']>=0 for r in known])),
        other_gesture_updates=len(unknown),other_gesture_acceptance_rate=float(np.mean([r['class_id']>=0 for r in unknown])) if unknown else None,
        warning='Three-class subset only. Other protocol gestures are out-of-set; softmax rejection is not a validated unknown-gesture detector.')

def tune(root,work):
    work=Path(work)
    if (work/'test_results.json').exists():raise ValueError('Cannot tune after exposing test results.')
    from train_multiclass import verify_frozen
    frozen=verify_frozen(work)
    rows,classes,c=continuous_predictions(root,work,'validation');candidates=[]
    grid=c.get('policy_search',dict(ema_alpha=[.3],dwell_updates=[2],min_confidence=[.4,.5,.6],min_margin=[0.,.05,.1]))
    for alpha,dwell,confidence,margin in product(grid['ema_alpha'],grid['dwell_updates'],grid['min_confidence'],grid['min_margin']):
        policy=dict(ema_alpha=alpha,dwell_updates=dwell,min_confidence=confidence,min_margin=margin)
        metrics=policy_metrics(apply_policy(rows,classes,policy),classes)
        candidates.append(dict(policy=policy,metrics=metrics))
    def objective(r):
        m=r['metrics'];p=r['policy']
        if c.get('policy_search'):
            return (m['balanced_accuracy'],min(m['class_recall'].values()),-(m['rest_false_activation_rate'] or 0),p['min_confidence'],p['min_margin'],-p['dwell_updates'],p['ema_alpha'])
        return (m['balanced_accuracy'],-(m['rest_false_activation_rate'] or 0),p['min_confidence'],p['min_margin'])
    best=max(candidates,key=objective)
    best['policy']['validation_tuned']=True
    contract_path=work/'eegnet_multiclass.pt.json';contract=json.loads(contract_path.read_text());contract['policy']=best['policy']
    contract_path.write_text(json.dumps(contract,indent=2),encoding='utf-8')
    frozen['contract_sha256']=digest(contract_path);(work/'frozen.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    report=dict(selection_scope='Validation sessions only; test not scored.',selected=best,candidates=candidates)
    (work/'policy_tuning.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(best,indent=2),flush=True)
    return report

def evaluate_frozen_replay(root,work):
    work=Path(work);c=json.loads((work/'eegnet_multiclass.pt.json').read_text())
    if not c.get('policy',{}).get('validation_tuned'):raise ValueError('Tune/freeze replay policy on validation first.')
    rows,classes,config=continuous_predictions(root,work,'test')
    metrics=policy_metrics(apply_policy(rows,classes,c['policy']),classes)
    return dict(policy=c['policy'],cue_trials=metrics,scope='Continuous causal recorded replay; model and policy frozen before test.')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--work',type=Path,default=Path('session_reports/multiclass/sub08_three_class_v1'))
    a=p.parse_args();tune(a.root,a.work)
