"""Multiclass metrics with cue votes and run-cluster uncertainty."""
import numpy as np
from sklearn.metrics import confusion_matrix,f1_score

def balanced_recall(y,p,n_classes):
    y=np.asarray(y);p=np.asarray(p)
    return float(np.mean([np.mean(p[y==i]==i) for i in range(n_classes)]))

def cue_probabilities(probabilities,records):
    names=sorted(set(r['trial'] for r in records));truth=[];values=[];runs=[]
    for name in names:
        indices=[i for i,r in enumerate(records) if r['trial']==name]
        labels=set(records[i]['label'] for i in indices)
        if len(labels)!=1: raise ValueError('One cue has conflicting labels.')
        truth.append(labels.pop());values.append(np.mean(probabilities[indices],axis=0));runs.append(records[indices[0]]['run'])
    return np.asarray(truth),np.stack(values),np.asarray(runs)

def classification(y,p,classes):
    y=np.asarray(y);p=np.asarray(p);k=len(classes)
    if set(y)!=set(range(k)): raise ValueError('Every evaluation class must have true examples.')
    cm=confusion_matrix(y,p,labels=np.arange(k))
    return dict(n=len(y),balanced_accuracy=balanced_recall(y,p,k),
        accuracy=float(np.mean(y==p)),macro_f1=float(f1_score(y,p,labels=np.arange(k),average='macro',zero_division=0)),
        confusion_matrix=cm.tolist(),recall={c:float(cm[i,i]/cm[i].sum()) for i,c in enumerate(classes)},
        rest_false_activation_rate=float(np.mean(p[y==classes.index('rest')]!=classes.index('rest'))) if 'rest' in classes else None)

def evaluate(probabilities,y,records,classes,uncertainty=False,seed=42):
    result=dict(windows=classification(y,np.argmax(probabilities,axis=1),classes))
    truth,probs,runs=cue_probabilities(probabilities,records);pred=probs.argmax(axis=1)
    result['cue_trials']=classification(truth,pred,classes)
    result['cue_trials']['rule']='Mean raw softmax probability per eligible cue, then argmax; overlapping windows are not independent.'
    if uncertainty:
        rng=np.random.default_rng(seed);unique=np.unique(runs);scores=[];null=[]
        observed=balanced_recall(truth,pred,len(classes))
        for _ in range(3000):
            idx=np.concatenate([np.flatnonzero(runs==run) for run in rng.choice(unique,len(unique),replace=True)])
            if len(np.unique(truth[idx]))==len(classes): scores.append(balanced_recall(truth[idx],pred[idx],len(classes)))
            perm=truth.copy()
            for run in unique:
                j=np.flatnonzero(runs==run);perm[j]=rng.permutation(perm[j])
            null.append(balanced_recall(perm,pred,len(classes)))
        result['cue_trials'].update(run_cluster_bootstrap_95ci=np.percentile(scores,[2.5,97.5]).tolist(),
            within_run_permutation_p=float((1+np.sum(np.asarray(null)>=observed-1e-12))/(len(null)+1)),
            uncertainty_warning='Run-cluster bootstrap and within-run label permutations; exploratory with few independent sessions.')
    return result

def trial_class_weights(y,records):
    """Equal total sampler weight per class, then per cue within class."""
    y=np.asarray(y);trial=np.array([r['trial'] for r in records]);weights=np.empty(len(y))
    for label in np.unique(y):
        groups=np.unique(trial[y==label])
        for group in groups:
            idx=np.flatnonzero(trial==group)
            if len(set(y[idx]))!=1: raise ValueError('Cue mixes labels.')
            weights[idx]=1./(len(groups)*len(idx))
    return weights
