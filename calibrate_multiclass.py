"""Fit bounded multiclass decision offsets on validation only, before testing."""
import argparse,json
from pathlib import Path
from itertools import product
import numpy as np
import torch
from scipy.special import softmax
from train_multiclass import verify_frozen,load_prepared,dump
from multiclass_data import digest
from multiclass_features import FeatureEnsembleRunner
from multiclass_metrics import evaluate

def select_bias(probabilities,y,records,classes,grid):
    if len(classes)!=3 or classes[0]!='rest':raise ValueError('Three-class rest-first calibration required.')
    candidates=[]
    for a,b in product(grid,grid):
        bias=np.array([0.,a,b]);p=softmax(np.log(np.maximum(probabilities,1e-12))+bias,axis=1)
        metrics=evaluate(p,y,records,classes)
        score=(metrics['cue_trials']['balanced_accuracy'],min(metrics['cue_trials']['recall'].values()),-float(np.linalg.norm(bias)))
        candidates.append(dict(bias=bias.tolist(),metrics=metrics,score=score))
    best=max(candidates,key=lambda r:tuple(r['score']))
    return best,candidates

def calibrate(work,config_path):
    work=Path(work)
    if (work/'test_results.json').exists():raise ValueError('Cannot fit class offsets after test exposure.')
    if (work/'class_bias_calibration.json').exists():raise ValueError('Class-offset calibration is already frozen.')
    frozen=verify_frozen(work);X,y,records,manifest,c=load_prepared(work)
    torch.set_num_threads(c['threads']);declaration=json.loads(Path(config_path).read_text())['class_bias_calibration']
    checkpoint=work/'eegnet_multiclass.pt';payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if payload['arch']!='feature_ensemble':raise ValueError('Feature neural ensemble required.')
    va=np.array([i for i,r in enumerate(records) if r['partition']=='validation']);rows=[records[i] for i in va]
    probabilities=FeatureEnsembleRunner(payload).predict(X[va])
    before=evaluate(probabilities,y[va],rows,c['classes'])
    best,candidates=select_bias(probabilities,y[va],rows,c['classes'],declaration['other_class_grid'])
    payload['post_logit_bias']=np.array(best['bias'],np.float32);torch.save(payload,checkpoint)
    contract_path=Path(str(checkpoint)+'.json');contract=json.loads(contract_path.read_text());contract['sha256']=digest(checkpoint)
    contract['class_bias_calibration']=dict(bias=best['bias'],scope='Validation only; no test use.',declaration=declaration,
        probability_definition='Mean neural softmax, validation-fitted class log offsets, then EMA.')
    dump(contract_path,contract);frozen.update(model_sha256=digest(checkpoint),contract_sha256=digest(contract_path));dump(work/'frozen.json',frozen)
    report=dict(before=before,selected=best,candidates=candidates,test_scored=False,declaration_sha256=digest(config_path))
    dump(work/'class_bias_calibration.json',report)
    print(json.dumps(dict(bias=best['bias'],before=before['cue_trials'],after=best['metrics']['cue_trials']),indent=2),flush=True)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',type=Path,default=Path('session_reports/multiclass/sub08_three_class_v2/selected'))
    p.add_argument('--config',type=Path,default=Path('multiclass_improvement_experiment.json'))
    a=p.parse_args();calibrate(a.work,a.config)
