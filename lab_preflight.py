"""Print lab readiness without issuing any hand commands."""
import argparse,json,importlib.metadata
from pathlib import Path
from realtime_bci_decoder import load_contract,RealtimeBCIDecoder
from lsl_acquisition import stream_metadata,StreamAdapter

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,default=Path(__file__).with_name('lab_config.json'))
    p.add_argument('--inspect-lsl',action='store_true');args=p.parse_args()
    config=json.loads(args.config.read_text(encoding='utf-8-sig'));root=Path(__file__).resolve().parent
    result=dict(packages={},model={},streams=[],ready=False)
    for name in ['numpy','scipy','torch','pandas','matplotlib','mne','scikit-learn','pylsl']:
        try: result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['packages'][name]='MISSING'
    try:
        path=(root/config['checkpoint']).resolve();contract=load_contract(path)
        decoder=RealtimeBCIDecoder(path,allow_unverified=config.get('allow_unverified_model',False),send_udp=False)
        result['model']=dict(path=str(path),channels=decoder.channels,window_s=decoder.window_samples/decoder.sfreq,contract_recorded=contract['verified'])
    except Exception as exc:
        result['model']['error']=str(exc)
        try: contract=load_contract(root/config['checkpoint'])
        except Exception: contract=None
    try:
        from pylsl import resolve_byprop,StreamInlet
        for stream in resolve_byprop('type','EEG',timeout=3.):
            inlet=StreamInlet(stream,max_buflen=1)
            try:
                meta=stream_metadata(inlet.info(timeout=2.));item={k:v for k,v in meta.items() if k!='xml'}
                if contract:
                    try: StreamAdapter(meta,contract,config);item['compatible_metadata']=True
                    except Exception as exc: item['compatible_metadata']=False;item['error']=str(exc)
                result['streams'].append(item)
            finally: inlet.close_stream()
    except Exception as exc: result['lsl_error']=str(exc)
    compatible=[s for s in result['streams'] if s.get('compatible_metadata')
                and (not config.get('stream_name') or s['name']==config['stream_name'])
                and (not config.get('source_id') or s['source_id']==config['source_id'])]
    result['ready']=not result['model'].get('error') and len(compatible)==1
    result['readiness_scope']='Metadata and checkpoint check only. Fresh data and prospective accuracy still require a live run.'
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result['ready'] or args.inspect_lsl else 1)
if __name__=='__main__': main()
