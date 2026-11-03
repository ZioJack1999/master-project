"""Cue protocols and descriptive, trial-based held-out evaluation."""
import numpy as np
import uuid
from sklearn.metrics import balanced_accuracy_score,confusion_matrix

def score_trials(truth,pred,permutations=5000,seed=42):
    y=np.asarray(truth,dtype=int); p=np.asarray(pred,dtype=int)
    if len(y)!=len(p) or len(y)<4 or set(y)!={0,1}: raise ValueError('Need complete predictions from both trial classes.')
    score=float(balanced_accuracy_score(y,p))
    rng=np.random.default_rng(seed)
    def balanced(labels):
        return .5*(np.mean(p[labels==0]==0)+np.mean(p[labels==1]==1))
    null=np.array([balanced(rng.permutation(y)) for _ in range(permutations)])
    return dict(n_trials=len(y),balanced_accuracy=score,confusion_matrix=confusion_matrix(y,p,labels=[0,1]).tolist(),
                permutation_p=float((1+(null>=score-1e-12).sum())/(permutations+1)),
                rest_false_positive_rate=float(p[y==0].mean()),action_sensitivity=float(p[y==1].mean()),
                unit_of_analysis='one majority decision per complete cue trial; not overlapping windows')

class CueProtocol:
    def __init__(self,kind,now,window_s=4.,trials_per_class=3,duration=8.,seed=42):
        if kind not in ('calibration','validation'): raise ValueError('Unknown protocol.')
        if duration<=window_s+.5: raise ValueError('Cue duration must exceed window plus .5 second settling.')
        self.kind,self.duration,self.window_s=kind,float(duration),window_s
        self.run_id=uuid.uuid4().hex
        self.labels=np.tile([0,1],trials_per_class)
        np.random.default_rng(seed).shuffle(self.labels)
        self.index=0
        self.start=float(now)
        self.events=[dict(run_id=self.run_id,kind=kind,trial=1,label=int(self.labels[0]),start=self.start,end=None)]
        self.rows=[]
        self.done=False
    def advance(self,now):
        if self.done or now-self.start<self.duration: return
        # Actual cue transitions follow the display/worker clock; never label a
        # backlog as if the participant had already seen the next cue.
        self.events[-1]['end']=float(now)
        self.index+=1
        if self.index==len(self.labels):
            self.done=True
            return
        self.start=float(now)
        self.events.append(dict(run_id=self.run_id,kind=self.kind,trial=self.index+1,label=int(self.labels[self.index]),start=self.start,end=None))
    def add(self,packet,sample_time):
        if self.done: return
        if sample_time-self.window_s < self.start+.5 or sample_time>self.start+self.duration: return
        self.rows.append(dict(trial=self.index+1,label=int(self.labels[self.index]),p_smooth=packet['p_smooth'],
                              decision=packet['grasp_command'],quality=packet['quality']))
    def finish(self,threshold):
        if not self.done: raise ValueError('Protocol has not finished.')
        grouped=[]
        for i,label in enumerate(self.labels,1):
            all_rows=[r for r in self.rows if r['trial']==i]
            rows=[r for r in all_rows if r['quality']=='ok']
            if len(rows)<4 or len(rows)<.5*len(all_rows):
                raise ValueError(f'Trial {i} has insufficient clean full-window predictions; repeat the run.')
            grouped.append((int(label),rows))
        if self.kind=='calibration':
            # Equal weight to each trial, sweep the same EMA probabilities used live.
            candidates=np.linspace(.1,.9,81)
            def objective(th):
                return np.mean([np.mean([(r['p_smooth']>=th)==bool(label) for r in rows]) for label,rows in grouped])
            best=max(candidates,key=lambda th:(objective(th),-abs(th-.5)))
            return dict(kind='calibration_fit_only',threshold=float(best),fit_balanced_accuracy=float(objective(best)),
                        n_trials=len(grouped),warning='Threshold fitted on these cues; this score is not validation.')
        y=[label for label,rows in grouped]
        pred=[int(np.mean([r['decision'] for r in rows])>.5) for label,rows in grouped]
        return dict(kind='held_out_cue_trials',threshold=float(threshold),**score_trials(y,pred))
