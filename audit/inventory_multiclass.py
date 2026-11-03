"""Inventory task-specific BIDS metadata; never traverse derivatives or participant forms."""
import argparse
import json
from pathlib import Path
import pandas as pd
CLASSES=['Opponens Grasp','Finger Extension','Volar Grasp','Wrist Flexion','Wrist Extension','Wrist Pronation','Rest','Pinch Grasp','Wrist Supination']

def inventory(root):
    paths=sorted(Path(root).glob('sub-*/ses-*/eeg/*task-olrandmcmitrain*_events.tsv'))
    rows=[];errors=[]
    for p in paths:
        try:
            d=pd.read_csv(p,sep='\t')
            labels=d['trial_type'].astype(str).str.replace('_',' ',regex=False).str.strip()
            edf=p.with_name(p.name.replace('_events.tsv','_eeg.edf'))
            eligible=(d.duration>=2.5)&labels.isin(CLASSES)
            rows.append(dict(subject=p.parents[2].name,session=p.parents[1].name,
                run=p.name.split('_run-')[1].split('_')[0],edf_exists=edf.exists(),n_events=len(d),
                classes=labels.value_counts().to_dict(),duration_min=float(d.duration.min()),
                duration_max=float(d.duration.max()),columns=list(d.columns),
                eligible_2s_cues=labels[eligible].value_counts().to_dict(),
                short_cues=int((d.duration<2.5).sum()),unknown_labels=sorted(set(labels)-set(CLASSES))))
        except Exception as exc: errors.append(dict(file=p.name,error=f'{type(exc).__name__}: {exc}'))
    return dict(event_files=len(paths),readable_files=len(rows),errors=errors,runs=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True,type=Path)
    p.add_argument('--output',type=Path,default=Path('audit/multiclass_inventory.local.json'))
    a=p.parse_args();r=inventory(a.root)
    a.output.write_text(json.dumps(r,indent=2),encoding='utf-8')
    print('Found',r['event_files'],'readable',r['readable_files'],'errors',len(r['errors']))
    for s in sorted(set(x['subject'] for x in r['runs'])):
        rows=[x for x in r['runs'] if x['subject']==s]
        counts={}
        for row in rows:
            for k,v in row['classes'].items(): counts[k]=counts.get(k,0)+v
        print(s,'runs',len(rows),'sessions',len(set(x['session'] for x in rows)),
              'paired_edfs',sum(x['edf_exists'] for x in rows),'counts',counts)
