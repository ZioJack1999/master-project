"""Fixed 3/5/9-class feasibility study with person-disjoint transfer folds."""
import argparse,copy,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from scipy.special import softmax
from multiclass_data import digest
from multiclass_cohort_data import load_cohort,prepare_cohort
from multiclass_features import FeaturePreprocessor,FeatureEnsembleRunner
from multiclass_metrics import evaluate,cue_probabilities
from improve_multiclass import fit_member
from train_multiclass import dump

def person_class_cue_weights(y,records):
    groups=defaultdict(lambda:defaultdict(dict));labels_by_trial={}
    for i,r in enumerate(records):
        key=(r['subject'],r['trial']);label=int(y[i])
        if key in labels_by_trial and labels_by_trial[key]!=label:raise ValueError('One cue mixes class labels.')
        labels_by_trial[key]=label;groups[r['subject']][label].setdefault(r['trial'],[]).append(i)
    result=np.zeros(len(y))
    for subject,labels in groups.items():
        for label,trials in labels.items():
            for trial,indices in trials.items():result[indices]=1./(len(groups)*len(labels)*len(trials)*len(indices))
    return result

def case_indices(records,classes,config,held_subject=None,within=None):
    result={p:[] for p in ['train','validation','test']};mapped=[]
    for i,r in enumerate(records):
        mapped.append(classes.index(r['label_name']) if r['label_name'] in classes else -1)
        if mapped[-1]<0:continue
        if held_subject is not None:
            part='test' if r['subject']==held_subject else 'validation' if r['session'] in config['source_validation_sessions'][r['subject']] else 'train'
        else:
            if r['subject']!='08':continue
            part=next((p for p,sessions in within.items() if r['session'] in sessions),None)
            if part is None:continue
        result[part].append(i)
    result={p:np.array(a,dtype=int) for p,a in result.items()};mapped=np.array(mapped,dtype=np.int64)
    for part,indices in result.items():
        if set(mapped[indices])!=set(range(len(classes))):raise ValueError(f'{part} lacks evaluation classes.')
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        if {records[i]['trial'] for i in result[a]}&{records[i]['trial'] for i in result[b]}:raise ValueError('Cue leakage between partitions.')
        if {(records[i]['subject'],records[i]['session']) for i in result[a]}&{(records[i]['subject'],records[i]['session']) for i in result[b]}:raise ValueError('Recording-session leakage.')
    if held_subject is not None:
        if any(records[i]['subject']==held_subject for p in ['train','validation'] for i in result[p]):raise ValueError('Held participant leaked into development.')
    return result,mapped

def rows_for(records,indices,labels):return [dict(records[i],label=int(labels[i])) for i in indices]

def diagnostics(probabilities,truth,records,classes):
    result=evaluate(probabilities,truth,records,classes)
    y,p,_=cue_probabilities(probabilities,records)
    result['cue_nll']=float(-np.log(np.maximum(p[np.arange(len(y)),y],1e-12)).mean())
    return result

def fit_payload(cov,y,train_rows,val_cov,val_y,val_rows,classes,config,seeds=None):
    c=dict(config,classes=classes);weights=person_class_cue_weights(y,train_rows)
    pre=FeaturePreprocessor.fit(cov,weights,c['family'],c['covariance_shrinkage'])
    train_x=pre.transform(cov);val_x=pre.transform(val_cov);states=[];histories=[]
    for seed in seeds or c['ensemble_seeds']:
        state,p,history=fit_member(train_x,y,weights,val_x,val_y,val_rows,c,pre.parameters,c['family'],cov.shape[-1],seed,train_rows=train_rows)
        states.append(state);histories.append(dict(seed=seed,epochs=history))
    payload=dict(arch='feature_ensemble',family=c['family'],n_classes=len(classes),n_channels=cov.shape[-1],
        n_times=round(c['window_s']*c['sfreq']),hidden_units=c['hidden_units'],dropout=c['dropout'],preprocessing=pre.parameters,
        member_states=states,train_mean=np.zeros(cov.shape[-1],np.float32),train_std=np.ones(cov.shape[-1],np.float32))
    runner=FeatureEnsembleRunner(payload)
    train=diagnostics(runner.from_covariances(cov),y,train_rows,classes)
    val=diagnostics(runner.from_covariances(val_cov),val_y,val_rows,classes)
    baseline=LogisticRegression(C=1.,max_iter=2000).fit(train_x,y,sample_weight=weights*len(y)/weights.sum())
    baseline_parameters=dict(coef=baseline.coef_.tolist(),intercept=baseline.intercept_.tolist())
    return payload,dict(seed_histories=histories,train=train,validation=val,
        train_validation_gap=train['cue_trials']['balanced_accuracy']-val['cue_trials']['balanced_accuracy'],
        train_independent_cues=len({r['trial'] for r in train_rows}),validation_independent_cues=len({r['trial'] for r in val_rows}),
        diagnostic_scope='A gap may reflect overfitting and/or recording shift; low training and validation can also reflect insufficient signal or labels.'),baseline_parameters

def run_case(work,name,classes,indices,labels,cov,records,manifest,config,data_hashes,held_subject=None):
    case=Path(work)/'cases'/name;case.mkdir(parents=True,exist_ok=True)
    result_path=case/'results.json';freeze_path=case/'frozen.json'
    if result_path.exists():return json.loads(result_path.read_text())
    tr,va,te=(indices[p] for p in ['train','validation','test'])
    tr_rows=rows_for(records,tr,labels);va_rows=rows_for(records,va,labels)
    if not freeze_path.exists():
        print('FIT',name,'train cues',len({r['trial'] for r in tr_rows}),'val cues',len({r['trial'] for r in va_rows}),flush=True)
        # Held-person covariances are not indexed until after the artifact freeze.
        payload,training,baseline=fit_payload(np.asarray(cov[tr]),labels[tr],tr_rows,np.asarray(cov[va]),labels[va],va_rows,classes,config)
        model=case/'model.pt';torch.save(payload,model);dump(case/'training.json',training);dump(case/'baseline.json',baseline)
        contract=dict(schema_version=2,task_mode='multiclass',arch='feature_ensemble',classes=classes,channels=manifest['channels'],
            role='source_pooled_transfer_reference' if held_subject else 'exploratory_within_person_reference',
            sfreq=config['sfreq'],n_times=payload['n_times'],normalization='feature_train_only',
            filter=dict(mode='causal',fmin=8.,fmax=30.,order=4),artifact_limit_v=config['artifact_limit_v'],
            min_signal_std_v=config['min_signal_std_v'],sha256=digest(model),verified=True,
            verified_meaning='Documented preprocessing/split; transfer study only, no prospective certification.',
            source_subjects=sorted({r['subject'] for r in tr_rows}),held_subject=held_subject,
            study_id=config['experiment_id'],study_case=name,policy_status='No live policy/actuator validation.')
        dump(str(model)+'.json',contract)
        dump(freeze_path,dict(model_sha256=digest(model),contract_sha256=digest(str(model)+'.json'),baseline_sha256=digest(case/'baseline.json'),
            data_hashes=data_hashes,classes=classes,source_subjects=contract['source_subjects'],held_subject=held_subject))
    frozen=json.loads(freeze_path.read_text());model=case/'model.pt'
    for key,path in [('model_sha256',model),('contract_sha256',Path(str(model)+'.json')),('baseline_sha256',case/'baseline.json')]:
        if frozen[key]!=digest(path):raise ValueError('Frozen transfer artifact changed.')
    if frozen['data_hashes']!=data_hashes:raise ValueError('Frozen study input changed.')
    payload=torch.load(model,map_location='cpu',weights_only=False);runner=FeatureEnsembleRunner(payload)
    target_cov=np.asarray(cov[te]);test_rows=rows_for(records,te,labels)
    p=runner.from_covariances(target_cov);neural=evaluate(p,labels[te],test_rows,classes,True,config['seed'])
    baseline=json.loads((case/'baseline.json').read_text());features=runner.preprocessor.transform(target_cov)
    bp=softmax(features@np.array(baseline['coef']).T+np.array(baseline['intercept']),axis=1)
    training=json.loads((case/'training.json').read_text())
    result=dict(case=name,classes=classes,n_classes=len(classes),chance_balanced_accuracy=1/len(classes),held_subject=held_subject,
        train_subjects=sorted({r['subject'] for r in tr_rows}),validation_subjects=sorted({r['subject'] for r in va_rows}),
        test_subjects=sorted({r['subject'] for r in test_rows}),neural=neural,
        linear_covariance_baseline=evaluate(bp,labels[te],test_rows,classes,True,config['seed']),
        training_diagnostics=training,test_cue_count=len({r['trial'] for r in test_rows}),
        scope='Zero-shot held-ID transfer; no target labels or statistics fitted.' if held_subject else 'Retrospective session-held-out feasibility; ID08 sessions already informed earlier development.')
    dump(result_path,result)
    print('SCORED',name,'NN cue BA',neural['cue_trials']['balanced_accuracy'],'linear',result['linear_covariance_baseline']['cue_trials']['balanced_accuracy'],flush=True)
    return result

def trial_subsample(indices,labels,records,fraction,seed):
    rng=np.random.default_rng(seed);groups=defaultdict(set)
    for i in indices:groups[(records[i]['subject'],int(labels[i]))].add(records[i]['trial'])
    keep=set()
    for names in groups.values():
        names=sorted(names);n=max(1,int(np.ceil(fraction*len(names))))
        keep.update(rng.choice(names,n,replace=False).tolist())
    return np.array([i for i in indices if records[i]['trial'] in keep],dtype=int)

def learning_curve(work,case_name,classes,indices,labels,cov,records,config):
    path=Path(work)/'cases'/case_name/'learning_curve.json'
    if path.exists():return json.loads(path.read_text())
    points=[];tr=indices['train'];va=indices['validation'];va_rows=rows_for(records,va,labels)
    for fraction in config['learning_curve_fractions']:
        chosen=trial_subsample(tr,labels,records,fraction,config['seed']);train_rows=rows_for(records,chosen,labels)
        print('LEARNING CURVE',case_name,fraction,flush=True)
        payload,diagnostic,baseline=fit_payload(np.asarray(cov[chosen]),labels[chosen],train_rows,np.asarray(cov[va]),labels[va],va_rows,classes,config,seeds=[config['seed']])
        points.append(dict(fraction=fraction,train_cues=len({r['trial'] for r in train_rows}),
            train_balanced_accuracy=diagnostic['train']['cue_trials']['balanced_accuracy'],
            validation_balanced_accuracy=diagnostic['validation']['cue_trials']['balanced_accuracy'],
            train_nll=diagnostic['train']['cue_nll'],validation_nll=diagnostic['validation']['cue_nll']))
    report=dict(case=case_name,seed=config['seed'],points=points,test_evaluated=False,
        scope='Whole-cue training fractions, source validation only; not used to select the final algorithm.')
    dump(path,report);return report

def run_study(work):
    work=Path(work);cov,records,manifest=load_cohort(work);config=manifest['config'];torch.set_num_threads(config['threads'])
    if (work/'study_results.json').exists():return json.loads((work/'study_results.json').read_text())
    hashes=dict(covariances_sha256=digest(work/'covariances.npy'),records_sha256=digest(work/'records.json'),manifest_sha256=digest(work/'cohort_manifest.json'))
    results=[];learning=[]
    for held in config['held_subject_order']:
        for class_set in config['cross_class_order']:
            classes=config['class_sets'][class_set];indices,labels=case_indices(records,classes,config,held_subject=held)
            name=f'transfer_hold_{held}_{class_set}'
            results.append(run_case(work,name,classes,indices,labels,cov,records,manifest,config,hashes,held))
    for cohort,split in config['within_person_splits'].items():
        for class_set in config['within_person_class_sets']:
            classes=config['class_sets'][class_set];indices,labels=case_indices(records,classes,config,within=split)
            name=f'within_08_{cohort}_{class_set}'
            results.append(run_case(work,name,classes,indices,labels,cov,records,manifest,config,hashes))
    for name,held,within in [('transfer_hold_07_nine',config['primary_held_subject'],None),('within_08_early_nine',None,config['within_person_splits']['early'])]:
        classes=config['class_sets']['nine'];indices,labels=case_indices(records,classes,config,held_subject=held,within=within)
        learning.append(learning_curve(work,name,classes,indices,labels,cov,records,config))
    report=dict(experiment_id=config['experiment_id'],config=config,subjects=manifest['subjects'],results=results,
        learning_curves=learning,data_hashes=hashes,
        warning='Three dataset participant IDs only. Outer folds are exploratory; ID08 informed earlier method design. Primary new held ID07 was not used to fit scales, weights or stopping.')
    dump(work/'study_results.json',report);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','run'])
    p.add_argument('--root',type=Path);p.add_argument('--config',type=Path,default=Path('multiclass_cohort_experiment.json'))
    p.add_argument('--work',type=Path,default=Path('session_reports/multiclass/cohort_study_v1'));a=p.parse_args()
    if a.stage=='prepare':
        if a.root is None:p.error('--root is required')
        prepare_cohort(a.root,json.loads(a.config.read_text()),a.work)
    else:run_study(a.work)
