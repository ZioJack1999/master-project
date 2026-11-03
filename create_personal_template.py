"""Create an explicitly untrained EEGNet template for a verified electrode montage."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from networks import EEGNet

MONTAGES = {
    'x24_stock20': ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4','T5','P3','Pz','P4','T6','O1','POz','O2'],
    'eegmi_custom20': ['F3','F1','Fz','F2','F4','C3','C1','Cz','C2','C4','CPz','P3','P1','Pz','P2','P4','POz','O1','Oz','O2'],
}

def create_template(output, montage='x24_stock20'):
    channels = MONTAGES[montage]
    torch.manual_seed(42)
    model = EEGNet(len(channels),512)
    output = Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(model_state=model.state_dict(),arch='eegnet',n_channels=len(channels),n_times=512,
                    train_mean=np.zeros(len(channels),np.float32),train_std=np.full(len(channels),10e-6,np.float32)),output)
    contract = dict(channels=channels,sfreq=256,n_times=512,normalization='fixed',
        filter=dict(mode='causal',fmin=8.,fmax=30.,order=4),classes=['rest','action'],verified=True,
        verified_meaning='Architecture and preprocessing only. Random weights: no fitted classifier or accuracy claim.',
        role='personal_training_template',montage=montage,threshold=.5,artifact_limit_v=100e-6,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    Path(str(output)+'.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    return contract

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--montage',choices=MONTAGES,default='x24_stock20')
    p.add_argument('--output',type=Path,default=Path('decoder_and_models/personal_template_x24_stock20.pt'))
    args=p.parse_args()
    create_template(args.output,args.montage)
    print(f'Created untrained {args.montage} template: {args.output}. Collect personal calibration before use.')
