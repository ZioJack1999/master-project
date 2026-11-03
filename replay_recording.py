"""Replay EDF EEG through the exact causal live engine; score cue-contained windows."""
import argparse,json,time
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import torch
from realtime_bci_decoder import RealtimeBCIDecoder
from protocols import score_trials
from sklearn.metrics import balanced_accuracy_score,confusion_matrix

def replay(edf,events,checkpoint,output,allow_unverified=False):
    d=RealtimeBCIDecoder(checkpoint,allow_unverified=allow_unverified,send_udp=False)
    raw=mne.io.read_raw_edf(edf,preload=False,verbose=False)
    if not np.isclose(raw.info['sfreq'],d.sfreq): raise ValueError('EDF sample rate does not match checkpoint.')
    missing=set(d.channels)-set(raw.ch_names)
    if missing: raise ValueError(f'Missing channels: {sorted(missing)}')
    data=raw.get_data(picks=d.channels).astype(np.float32);events=pd.read_csv(events,sep='\t')
    cue_spans=[]
    for trial,r in events.iterrows():
        if r['trial_type'] not in ['rest','action']: continue
        a=int(r['sample']);b=a+round(float(r['duration'])*d.sfreq)
        cue_spans.append((int(trial),a,b,int(r['trial_type']=='action')))
    rows=[];latencies=[]
    for start in range(0,data.shape[1],d.step_samples):
        t0=time.perf_counter();outs=d.process_chunk_all(data[:,start:start+d.step_samples],emit=False)
        elapsed=(time.perf_counter()-t0)*1000
        for packet in outs:
            latencies.append(elapsed)
            a=packet['sample_end']-d.window_samples;b=packet['sample_end']
            cue=next(((trial,label) for trial,s,e,label in cue_spans if a>=s+round(.5*d.sfreq) and b<=e),None)
            rows.append(dict(packet,trial=cue[0] if cue else None,label=cue[1] if cue else None))
    eligible=[r for r in rows if r['trial'] is not None and r['quality']=='ok']
    if not eligible or set(r['label'] for r in eligible)!={0,1}:
        raise ValueError('Cannot score both classes using clean cue-contained windows. A four-second model cannot fit a three-second rest cue.')
    y=[r['label'] for r in eligible];pred=[r['grasp_command'] for r in eligible]
    gt=[];pr=[]
    for trial in sorted(set(r['trial'] for r in eligible)):
        subset=[r for r in eligible if r['trial']==trial]
        if len(subset)<2: continue
        gt.append(subset[0]['label']);pr.append(int(np.mean([r['grasp_command'] for r in subset])>.5))
    run=Path(edf).stem.replace('_eeg','')
    used=d.contract.get('training_runs',[])+d.contract.get('validation_runs',[])
    result=dict(recording=Path(edf).name,held_out_from_training_and_tuning=run not in used and bool(d.contract.get('training_runs')),
                model_sha256=d.contract['sha256'],threshold=d.threshold,
                n_clean_cue_windows=len(eligible),n_predictions=len(rows),
                window_balanced_accuracy=float(balanced_accuracy_score(y,pred)),
                window_confusion_matrix=confusion_matrix(y,pred,labels=[0,1]).tolist(),
                cue_trials=score_trials(gt,pr) if len(gt)>=4 and set(gt)=={0,1} else None,
                inference_chunk_ms_median=float(np.median(latencies)),inference_chunk_ms_p95=float(np.percentile(latencies,95)),
                warning='Offline recorded-data replay; prospective imagery control and end-to-end hardware latency are untested.')
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    pd.DataFrame(rows).to_csv(str(output)+'.predictions.csv',index=False)
    d.close();return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edf',required=True,type=Path);p.add_argument('--events',required=True,type=Path)
    p.add_argument('--ckpt',type=Path,default=Path(__file__).with_name('decoder_and_models')/'eegnet_sub08_causal20_fixed.pt')
    p.add_argument('--output',type=Path,default=Path('session_reports/replay.json'));p.add_argument('--allow-unverified-model',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    print(json.dumps(replay(args.edf,args.events,args.ckpt,args.output,args.allow_unverified_model),indent=2))
if __name__=='__main__': main()
