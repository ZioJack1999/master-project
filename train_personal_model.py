"""Train a person's rest/imagery EEGNet on LIVE calibration cues only.

Collect at least 18 clean calibration trials (one default 144s run) first.
Separate entire cue trials for tuning/test. Run NEW prospective validation after
training; the small internal test set cannot establish reliable control.
"""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.signal import sosfilt
from realtime_bci_decoder import RealtimeBCIDecoder
from train_lab_model import train_export

def load_personal(base,checkpoint,run_ids=None):
    base=Path(base)
    meta=json.loads(Path(str(base)+'_metadata.json').read_text(encoding='utf-8'))
    if meta.get('source')!='live': raise ValueError('Personal training requires LIVE data; synthetic demo is excluded.')
    data=np.load(str(base)+'_raw.npz',allow_pickle=False)
    d=RealtimeBCIDecoder(checkpoint,send_udp=False)
    channels=data['channels'].tolist()
    if channels!=d.channels or not np.isclose(float(data['sfreq']),d.sfreq): raise ValueError('Session/base model montage or sample rate mismatch.')
    raw=data['eeg_volts'];ts=data['lsl_timestamps'];events=pd.read_csv(str(base)+'_events.csv')
    if len(ts)<d.window_samples or not np.isfinite(raw).all() or np.any(np.diff(ts)<=0): raise ValueError('Invalid or insufficient session samples.')
    if 'kind' not in events or 'run_id' not in events: raise ValueError('Cue provenance missing; record using the corrected studio.')
    events=events[events['kind']=='calibration']
    if run_ids is not None: events=events[events['run_id'].isin(run_ids)]
    if 'aborted' in events: events=events[events['aborted'].isna()]
    filtered=np.empty_like(raw)
    breaks=np.unique(np.r_[0,np.flatnonzero(np.diff(ts)>.1)+1,
                          [r['raw_start'] for r in meta.get('raw_segments',[])],len(ts)]).astype(int)
    for a,b in zip(breaks[:-1],breaks[1:]): filtered[:,a:b]=sosfilt(d.sos,raw[:,a:b],axis=1)
    bad=(abs(filtered)>d.artifact_limit).any(axis=0)
    xs,ys,groups,records=[],[],[],[]
    for r in events.to_dict('records'):
        if not np.isfinite(r['end']) or r['label'] not in (0,1): continue
        start=np.searchsorted(ts,r['start']+.5);end=np.searchsorted(ts,r['end'])
        group=str(r['run_id'])+'_'+str(int(r['trial']))
        for a in range(start,end-d.window_samples+1,round(.5*d.sfreq)):
            b=a+d.window_samples
            if bad[a:b].any() or np.any(np.diff(ts[a:b])>.1) or np.any(filtered[:,a:b].std(axis=1)<d.min_signal_std): continue
            xs.append(filtered[:,a:b].copy());ys.append(int(r['label']));groups.append(group)
            records.append(dict(subject='personal',session=base.name,run=group,start=int(a),end=int(b),label=int(r['label'])))
    if len(set(groups))<18: raise ValueError('Need at least 18 clean calibration trials. Improve signal quality and repeat the complete 18-cue calibration.')
    for label in (0,1):
        if len(set(g for g,y in zip(groups,ys) if y==label))<6: raise ValueError('Need at least six clean trials of each class.')
    return np.stack(xs),np.array(ys),np.array(groups),records,channels

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session-base',required=True,type=Path,help='Saved session path without _raw.npz or other suffix.')
    p.add_argument('--base-model',type=Path,default=Path(__file__).with_name('decoder_and_models')/'personal_template_x24_stock20.pt')
    p.add_argument('--output',type=Path,default=Path('decoder_and_models/eegnet_personal.pt'))
    p.add_argument('--epochs',type=int,default=60);p.add_argument('--unfreeze',action='store_true')
    p.add_argument('--pretrained',action='store_true',help='Optional initialization from base weights; default trains on this person from scratch.')
    args=p.parse_args();torch.set_num_threads(1)
    if args.pretrained:
        from realtime_bci_decoder import load_contract
        if load_contract(args.base_model).get('role')=='personal_training_template':
            raise ValueError('An untrained template cannot initialize pretrained fitting. Omit --pretrained.')
    X,y,groups,records,channels=load_personal(args.session_base,args.base_model)
    train_export(X,y,groups,records,args.output,channels,epochs=args.epochs,patience=12,batch_size=16,
                 initial_checkpoint=args.base_model if args.pretrained else None,freeze_backbone=args.pretrained and not args.unfreeze)
    contract_path=Path(str(args.output)+'.json');contract=json.loads(contract_path.read_text(encoding='utf-8'))
    contract.update(personal_model=True,initialization='pretrained' if args.pretrained else 'scratch')
    contract_path.write_text(json.dumps(contract,indent=2),encoding='utf-8')
    print('Set checkpoint and its saved threshold in lab_config.json, restart the studio, and record NEW validation cues.')
if __name__=='__main__': main()
