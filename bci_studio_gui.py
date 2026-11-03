"""Local BCI studio: validated LSL acquisition, real calibration, and held-out cues."""
import argparse
from collections import deque
from datetime import datetime,timezone
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlparse,parse_qs
import webbrowser
import numpy as np
import torch
from realtime_bci_decoder import RealtimeBCIDecoder,load_contract
from lsl_acquisition import connect
from protocols import CueProtocol

ROOT=Path(__file__).resolve().parent

def clock_now():
    from pylsl import local_clock
    return local_clock()

class Studio:
    def __init__(self,config,root=ROOT,clock=clock_now):
        self.root=Path(root);self.config=dict(config);self.clock=clock
        self.lock=threading.RLock();self.stop=threading.Event()
        self.inlet=self.adapter=self.decoder=self.protocol=None
        self.checkpoint=(self.root/config['checkpoint']).resolve()
        self.base_checkpoint=self.checkpoint
        self.contract=load_contract(self.checkpoint,config.get('contract'))
        self.reports=(self.root/config.get('reports_dir','session_reports')).resolve()
        self.reports.mkdir(parents=True,exist_ok=True)
        self.state=dict(is_connected=False,is_streaming=False,demo_mode=False,is_calibrating=False,
            calib_phase='idle',calib_timer=0.,calib_progress=0.,calib_trial=0,total_trials=0,
            calib_acc=None,calib_th=.5,threshold=float(config.get('threshold',self.contract.get('threshold',.5))),p_raw=0.,p_smooth=0.,
            grasp_command=0,continuous_flex=0.,latency_ms=0.,step_count=0,stream_name='No EEG stream',
            udp_target=f"{config.get('udp_host','127.0.0.1')}:{config.get('udp_port',5005)}",packets_sent=0,
            error='',quality='warming_up',model_verified=bool(self.contract['verified']),validation_result=None,
            calibration_result=None,protocol_kind='',source='live',sample_age_s=None)
        self.state['window_s']=self.contract['n_times']/self.contract['sfreq']
        self.state.update(is_training=False,training_status='',personalized=False,
                          participant_id=config.get('participant_id','self'),model_path=str(self.checkpoint),
                          personalization_enabled=config.get('personalize_on_calibration',False),
                          calibration_cues=2*int(config.get('personal_calibration_pairs',9)))
        self.state.update(model_role=self.contract.get('role','trained_classifier'),
                          trial_duration_s=float(config.get('trial_duration_s',8.)))
        self.hist={key:deque([0.]*60,maxlen=60) for key in ['eeg_c3','eeg_cz','eeg_c4','prob_history','ema_history']}
        self.scope={key:deque([0.]*512,maxlen=512) for key in ['wave_c3','wave_cz','wave_c4']}
        self.rows=[];self.raw=[];self.timestamps=[];self.events=[];self.results=[]
        self.raw_segments=[];self.raw_sample_count=0
        self.training_cancel=threading.Event();self.training_thread=None
        self.started_at=None;self.last_data=None;self.segment=0;self.rng=np.random.default_rng(42)
        self.replay_data=None;self.replay_events=[];self.replay_position=0;self.replay_next=0.;self.replay_anchor=None
        if config.get('replay_edf'):
            import mne
            recording=Path(config['replay_edf']).resolve()
            raw=mne.io.read_raw_edf(recording,preload=False,verbose=False)
            if not np.isclose(raw.info['sfreq'],self.contract['sfreq']): raise ValueError('Replay sample rate differs from model.')
            self.replay_data=raw.get_data(picks=self.contract['channels']).astype(np.float32)
            events=recording.with_name(recording.name.replace('_eeg.edf','_events.tsv'))
            if events.exists():
                import pandas as pd
                self.replay_events=pd.read_csv(events,sep='\t').to_dict('records')
            self.state.update(source='recorded_replay',stream_name=recording.name,is_connected=True,recorded_cue='unlabeled')
        self.load_decoder(False)
    def load_decoder(self,demo):
        if self.decoder: self.decoder.close()
        self.decoder=None
        try:
            self.decoder=RealtimeBCIDecoder(self.checkpoint,threshold=self.state['threshold'],
                udp_host=self.config.get('udp_host','127.0.0.1'),udp_port=self.config.get('udp_port',5005),
                contract_path=self.config.get('contract'),
                allow_unverified=demo or self.config.get('allow_unverified_model',False),send_udp=not demo and self.replay_data is None)
            self.state['error']=''
        except Exception as exc: self.state['error']=str(exc)
    def snapshot(self):
        with self.lock: return dict(self.state,**{k:list(v) for k,v in self.hist.items()},**{k:list(v) for k,v in self.scope.items()})
    def neutral(self,reason):
        self.state.update(grasp_command=0,continuous_flex=0.,p_raw=0.,p_smooth=0.,quality=reason)
        if self.decoder:
            self.decoder.send_rest(reason);self.state['packets_sent']=self.decoder.packets_sent
    def cancel_protocol(self,reason):
        if self.protocol:
            event=dict(self.protocol.events[-1]);event['end']=self.clock();event['aborted']=reason
            self.events.extend(self.protocol.events[:-1]+[event])
            self.protocol=None
        self.state.update(is_calibrating=False,calib_phase='idle',protocol_kind='')
    def pause(self,reason='paused'):
        self.state['is_streaming']=False
        self.cancel_protocol(reason)
        self.neutral(reason)
        if self.decoder: self.decoder.reset_state()
    def reset(self):
        self.pause('reset')
        self.rows.clear();self.raw.clear();self.timestamps.clear();self.events.clear();self.results.clear()
        self.raw_segments.clear();self.raw_sample_count=0
        self.started_at=None;self.segment=0
        for history in self.hist.values(): history.clear();history.extend([0.]*60)
        for history in self.scope.values(): history.clear();history.extend([0.]*512)
        self.replay_position=0;self.replay_anchor=None;self.replay_next=0.
        self.state.update(step_count=0,validation_result=None)
        if not self.state['personalized']: self.state.update(calib_acc=None,calibration_result=None)
    def action(self,path,query):
        with self.lock:
            if path=='/api/toggle_stream':
                if self.state['is_training']: raise ValueError('Personal model fitting is in progress. Wait or cancel first.')
                if self.protocol: raise ValueError('Stop the cue run using Stop / Cancel first.')
                if self.state['is_streaming']: self.pause()
                else:
                    if self.decoder is None: raise ValueError(self.state['error'] or 'No model loaded.')
                    if not self.state['demo_mode'] and not self.state['is_connected']: raise ValueError('No fresh, compatible EEG stream.')
                    self.decoder.reset_state();self.segment+=1;self.state['is_streaming']=True
                    self.raw_segments.append(dict(segment=self.segment,raw_start=self.raw_sample_count))
                    self.replay_anchor=None;self.replay_next=0.
            elif path=='/api/stop':
                self.training_cancel.set();self.pause('cancelled')
            elif path=='/api/reset':
                if self.state['is_training']: raise ValueError('Cancel personal model fitting before resetting.')
                self.reset()
            elif path=='/api/toggle_demo':
                if self.state['is_training']: raise ValueError('Cancel personal fitting before switching source.')
                if self.replay_data is not None: raise ValueError('This studio is replaying a recording. Restart without --replay-edf for live/demo input.')
                if self.rows or self.state['is_streaming']: raise ValueError('Save or reset the current session before switching data source.')
                self.state['demo_mode']=not self.state['demo_mode']
                self.state['source']='synthetic_demo' if self.state['demo_mode'] else 'live'
                if self.inlet: self.inlet.close_stream()
                self.inlet=self.adapter=None
                self.state['is_connected']=False
                self.load_decoder(self.state['demo_mode'])
            elif path=='/api/set_threshold':
                if self.protocol or self.state['is_training']: raise ValueError('Threshold is frozen during cue runs and model fitting.')
                value=float(query.get('val',[''])[0])
                if not np.isfinite(value) or not .01<=value<=.99: raise ValueError('Threshold must be between .01 and .99.')
                self.state['threshold']=value
                if self.decoder:
                    self.decoder.threshold=value;self.decoder.dwell_counter=0
            elif path in ('/api/calibrate','/api/validate'):
                if self.state['is_training']: raise ValueError('A personal model is already being fitted.')
                if self.protocol: raise ValueError('A cue run is already in progress.')
                if not self.state['is_streaming'] or self.decoder is None: raise ValueError('Start a compatible stream first.')
                if path=='/api/validate' and self.state['demo_mode']: raise ValueError('Synthetic demo cannot validate the NN.')
                if self.replay_data is not None: raise ValueError('Recorded replay cannot run prospective cues. Restart without --replay-edf for your live self-test.')
                kind='calibration' if path=='/api/calibrate' else 'validation'
                if kind=='validation' and self.contract.get('role')=='personal_training_template':
                    raise ValueError('Run personal calibration first. The architecture template has no fitted classifier.')
                if kind=='validation' and self.config.get('require_personal_calibration',False) and not self.state['personalized']:
                    raise ValueError('Run personal calibration first. Validation must use a frozen personal model.')
                personal=kind=='calibration' and self.config.get('personalize_on_calibration',False) and not self.state['demo_mode']
                if personal:
                    import re
                    participant=query.get('participant',[self.state['participant_id']])[0]
                    if re.fullmatch(r'[A-Za-z0-9_-]{1,40}',participant) is None: raise ValueError('Use a short pseudonymous participant ID: letters, digits, underscore or hyphen.')
                    if self.rows: self.pause('new_personal_calibration');self.save();self.reset()
                    self.state.update(participant_id=participant,personalized=False,threshold=float(self.config.get('threshold',.5)),error='')
                    self.state.update(calib_acc=None,calibration_result=None,validation_result=None)
                    self.checkpoint=self.base_checkpoint;self.contract=load_contract(self.checkpoint)
                    self.state['model_path']=str(self.checkpoint)
                    self.state['model_role']=self.contract.get('role','trained_classifier')
                    self.load_decoder(False)
                    if self.decoder is None: raise ValueError(self.state['error'])
                    self.state['is_streaming']=True;self.segment+=1
                    self.raw_segments.append(dict(segment=self.segment,raw_start=self.raw_sample_count))
                pairs=int(self.config.get('personal_calibration_pairs',9) if personal else self.config.get('calibration_pairs',3) if kind=='calibration' else self.config.get('validation_pairs',10))
                self.protocol=CueProtocol(kind,self.clock(),self.decoder.window_samples/self.decoder.sfreq,
                    trials_per_class=pairs,duration=float(self.config.get('trial_duration_s',8.)),
                    seed=int(self.config.get('seed',42))+len(self.results))
                self.state.update(is_calibrating=True,protocol_kind=kind,total_trials=2*pairs)
                self.neutral('cue_run_started')
                self.refresh_protocol(self.clock())
            elif path=='/api/export_figure':
                if self.state['is_training']: raise ValueError('Calibration data are already saved. Wait for fitting or cancel before exporting.')
                self.pause('saved')
                return self.save()
            else: raise ValueError('Unknown operation.')
            return self.snapshot()
    def refresh_protocol(self,now):
        if not self.protocol: return
        p=self.protocol;p.advance(now)
        if p.done:
            self.events.extend(p.events)
            protocol_success=False
            personal=p.kind=='calibration' and self.config.get('personalize_on_calibration',False) and self.state['source']=='live'
            try:
                if personal:
                    # Fit the NN from raw calibration EEG below. Random template
                    # probabilities are not a person's calibration score.
                    result=dict(kind='personal_eeg_collection',n_trials=len(p.events),
                                warning='EEG collected. Personal NN fitting and tuning are pending.')
                else: result=p.finish(self.state['threshold'])
                protocol_success=True
                result['run_id']=p.run_id
                result['source']=self.state['source'];result['model_verified']=self.contract['verified']
                result['scope']='exploratory_self_test' if not self.contract['verified'] else 'prospective_cue_test'
                self.results.append(result)
                if p.kind=='calibration' and not personal:
                    self.state.update(threshold=result['threshold'],calib_th=result['threshold'],
                        calib_acc=result['fit_balanced_accuracy'],calibration_result=result)
                    self.decoder.threshold=result['threshold'];self.decoder.dwell_counter=0
                elif p.kind=='validation': self.state['validation_result']=result
            except ValueError as exc: self.state['error']=str(exc)
            self.protocol=None;self.state.update(is_calibrating=False,calib_phase='idle',protocol_kind='')
            self.pause('cue_run_complete')
            saved=self.save()
            if protocol_success and personal:
                self.start_personal_training(saved['base'],p.run_id)
            return
        remaining=max(0.,p.duration-(now-p.start))
        self.state.update(calib_trial=p.index+1,calib_phase='action' if p.labels[p.index] else 'rest',
                          calib_timer=remaining,calib_progress=remaining/p.duration)
    def ingest(self,raw,ts,now):
        # Caller holds the lock; reports and control changes cannot split a row.
        if not self.state['is_streaming']: return
        if self.started_at is None: self.started_at=float(ts[0])
        self.raw.append(raw.copy());self.timestamps.append(np.asarray(ts).copy())
        self.raw_sample_count+=raw.shape[1]
        before=self.decoder.samples_seen;t0=time.perf_counter()
        output_ready=self.state['personalized'] or not self.config.get('require_personal_calibration',False)
        protocol_output=self.protocol is None or (self.protocol.kind=='validation' and self.config.get('validation_udp_output',True))
        outputs=self.decoder.process_chunk_all(raw,emit=self.state['source']=='live' and protocol_output and output_ready)
        for ch,key in [('C3','wave_c3'),('Cz','wave_cz'),('C4','wave_c4')]:
            idx=next((i for i,n in enumerate(self.decoder.channels) if n.casefold()==ch.casefold()),None)
            if idx is not None: self.scope[key].extend((self.decoder.last_chunk_filtered[idx]*1e6).tolist())
        latency=(time.perf_counter()-t0)*1000
        self.state['latency_ms']=latency
        for packet in outputs:
            offset=packet['sample_end']-before-1
            sample_time=float(ts[offset])
            if self.protocol: self.protocol.add(packet,sample_time)
            row=dict(packet,lsl_time=sample_time,time_s=sample_time-self.started_at,segment=self.segment,
                     participant_id=self.state['participant_id'],model_sha256=self.contract['sha256'],
                     source=self.state['source'],latency_ms=latency,protocol=self.protocol.kind if self.protocol else '',
                     protocol_run_id=self.protocol.run_id if self.protocol else None,
                     trial=self.protocol.index+1 if self.protocol else None,
                     cue=int(self.protocol.labels[self.protocol.index]) if self.protocol else None)
            self.rows.append(row)
            for key in ('p_raw','p_smooth','grasp_command','continuous_flex','quality'): self.state[key]=packet[key]
            self.state['step_count']+=1
            self.hist['prob_history'].append(packet['p_raw']);self.hist['ema_history'].append(packet['p_smooth'])
            for ch,key in [('C3','eeg_c3'),('Cz','eeg_cz'),('C4','eeg_c4')]:
                idx=next((i for i,n in enumerate(self.decoder.channels) if n.casefold()==ch.casefold()),None)
                value=float(np.sqrt(np.mean(self.decoder.last_filtered[idx]**2))*1e6) if idx is not None else 0.
                self.hist[key].append(value);row[ch+'_rms_uv']=value if idx is not None else None
        self.state['packets_sent']=self.decoder.packets_sent
    def save(self):
        import pandas as pd
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        name='session_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        base=self.reports/name
        pd.DataFrame(self.rows).to_csv(str(base)+'_predictions.csv',index=False)
        pd.DataFrame(self.events).to_csv(str(base)+'_events.csv',index=False)
        data=np.concatenate(self.raw,axis=1) if self.raw else np.empty((len(self.contract['channels']),0),np.float32)
        times=np.concatenate(self.timestamps) if self.timestamps else np.empty(0)
        np.savez_compressed(str(base)+'_raw.npz',eeg_volts=data,lsl_timestamps=times,channels=np.array(self.contract['channels']),sfreq=self.contract['sfreq'])
        meta=dict(created_utc=datetime.now(timezone.utc).isoformat(),source=self.state['source'],config=self.config,
                  participant_id=self.state['participant_id'],raw_segments=self.raw_segments,
                  personal_calibration=self.state['calibration_result'],
                  model_contract=self.contract,stream=self.adapter.metadata if self.adapter and not self.state['demo_mode'] else None,
                  results=self.results,n_predictions=len(self.rows),n_raw_samples=data.shape[1],
                  calibration_is_validation=False,error=self.state['error'])
        Path(str(base)+'_metadata.json').write_text(json.dumps(meta,indent=2,allow_nan=False),encoding='utf-8')
        fig,axes=plt.subplots(2,1,figsize=(11,6),sharex=True)
        if self.rows:
            t=[r['time_s'] for r in self.rows]
            axes[0].plot(t,[r['p_raw'] for r in self.rows],label='p(action)')
            axes[0].plot(t,[r['p_smooth'] for r in self.rows],label='EMA')
            axes[0].plot(t,[r['threshold'] for r in self.rows],'--',label='Threshold at prediction')
            axes[0].legend()
            axes[1].step(t,[r['grasp_command'] for r in self.rows],where='post')
        axes[0].set_ylim(0,1);axes[0].set_ylabel('Probability')
        axes[1].set_ylabel('Decision');axes[1].set_xlabel('Seconds from first recorded sample')
        fig.suptitle('SYNTHETIC DEMO - NOT VALIDATION' if self.state['demo_mode'] else 'EEG cue session - see metadata for measured results')
        fig.tight_layout();fig.savefig(str(base)+'_plot.png',dpi=130);plt.close(fig)
        self.state['last_saved']=str(base)
        return dict(status='saved',base=str(base),filename_png=name+'_plot.png',filename_csv=name+'_predictions.csv',folder=str(self.reports))

    def start_personal_training(self,snapshot_base,run_id):
        self.training_cancel=threading.Event()
        cancel=self.training_cancel
        participant=self.state['participant_id']
        self.state.update(is_training=True,training_status='Preparing personal calibration EEG',calib_acc=None,error='')
        output=self.reports/'personal_models'/f'{participant}_{run_id}.pt'
        def progress(row):
            with self.lock:
                self.state['training_status']=f"Fitting personal NN, epoch {row['epoch']}, tuning score {row['validation_balanced_accuracy']:.1%} (not final validation)"
        def fit_personal():
            try:
                from train_personal_model import load_personal
                from train_lab_model import train_export
                if self.config.get('personal_initialization','scratch')=='pretrained' and load_contract(self.base_checkpoint).get('role')=='personal_training_template':
                    raise ValueError('Choose scratch initialization for an untrained architecture template.')
                X,y,groups,records,channels=load_personal(snapshot_base,self.base_checkpoint,run_ids=[run_id])
                result=train_export(X,y,groups,records,output,channels,
                    epochs=int(self.config.get('personal_training_epochs',60)),patience=12,batch_size=16,
                    initial_checkpoint=self.base_checkpoint if self.config.get('personal_initialization','scratch')=='pretrained' else None,
                    freeze_backbone=self.config.get('personal_initialization','scratch')=='pretrained' and not self.config.get('personal_unfreeze',False),
                    progress_callback=progress,stop_event=cancel)
                with self.lock:
                    if cancel.is_set() or self.stop.is_set(): raise InterruptedError('Personal calibration cancelled.')
                    contract_path=Path(str(output)+'.json')
                    personal_contract=json.loads(contract_path.read_text(encoding='utf-8'))
                    personal_contract.update(personal_model=True,participant_id=participant,calibration_run_id=run_id,
                        initialization=self.config.get('personal_initialization','scratch'))
                    contract_path.write_text(json.dumps(personal_contract,indent=2),encoding='utf-8')
                    self.checkpoint=output;self.contract=load_contract(output)
                    self.state['threshold']=self.contract['threshold']
                    self.load_decoder(False)
                    if self.decoder is None: raise ValueError(self.state['error'])
                    calibration=dict(kind='personal_nn_calibration',run_id=run_id,participant_id=participant,
                        model_path=str(output),model_sha256=self.contract['sha256'],threshold=self.contract['threshold'],
                        tuning_balanced_accuracy=result['validation_balanced_accuracy_at_threshold'],
                        internal_test_balanced_accuracy=result['test_balanced_accuracy'],
                        warning='Personal fitting and tuning completed. Record new validation cues to evaluate performance.')
                    self.results.append(calibration)
                    self.state.update(personalized=True,model_path=str(output),calibration_result=calibration,
                        model_role='trained_classifier',
                        calib_acc=calibration['tuning_balanced_accuracy'],training_status='Personal model ready. Start Stream, then Validate on new cues.',error='')
                    self.save()
            except Exception as exc:
                with self.lock:
                    self.state.update(error=str(exc),training_status='Personal model fitting failed or was cancelled.',personalized=False)
            finally:
                with self.lock: self.state['is_training']=False
        self.training_thread=threading.Thread(target=fit_personal,daemon=True)
        self.training_thread.start()
    def worker(self):
        retry_at=0.;demo_end=None;demo_next=0.
        while not self.stop.is_set():
            now=self.clock()
            with self.lock:
                self.refresh_protocol(now)
                demo=self.state['demo_mode']
            if demo:
                with self.lock:
                    if self.state['is_streaming'] and self.decoder and now>=demo_next:
                        n=self.decoder.step_samples
                        end=now;ts=end-np.arange(n-1,-1,-1)/self.decoder.sfreq
                        raw=self.rng.normal(0,5e-6,(self.decoder.n_ch,n)).astype(np.float32)
                        self.ingest(raw,ts,now);demo_next=now+n/self.decoder.sfreq
                self.stop.wait(.01);continue
            if self.replay_data is not None:
                with self.lock:
                    if self.state['is_streaming'] and now>=self.replay_next:
                        start=self.replay_position;end=min(start+self.decoder.step_samples,self.replay_data.shape[1])
                        if start==end: self.pause('recording_complete')
                        else:
                            if self.replay_anchor is None: self.replay_anchor=now-(end-1)/self.decoder.sfreq
                            ts=self.replay_anchor+np.arange(start,end)/self.decoder.sfreq
                            self.ingest(self.replay_data[:,start:end],ts,now)
                            self.replay_position=end;self.replay_next=max(self.replay_next,now)+(end-start)/self.decoder.sfreq
                            cue=next((r['trial_type'] for r in self.replay_events if r['sample']<=end-1<r['sample']+r['duration']*self.decoder.sfreq),'unlabeled')
                            self.state.update(recorded_cue=cue,replay_time_s=end/self.decoder.sfreq)
                self.stop.wait(.01);continue
            if self.inlet is None:
                if now<retry_at: self.stop.wait(.05);continue
                try:
                    inlet,adapter=connect(self.config,self.contract)
                    with self.lock:
                        self.inlet,self.adapter=inlet,adapter
                        self.state['stream_name']=adapter.metadata['name'];self.last_data=self.clock()
                        if self.decoder: self.state['error']=''
                except Exception as exc:
                    with self.lock:
                        self.state['is_connected']=False;self.state['error']=str(exc)
                    retry_at=self.clock()+2.
                    continue
            try:
                chunk,ts=self.inlet.pull_chunk(timeout=.05,max_samples=64)
                now=self.clock()
                with self.lock:
                    now=self.clock()
                    if self.state['demo_mode']: continue
                    if chunk:
                        if not self.state['is_streaming'] and now-ts[-1]>self.adapter.max_age:
                            self.inlet.flush();self.adapter.reset_timestamps()
                            self.state['is_connected']=False;self.last_data=now
                            continue
                        raw,ts=self.adapter.transform(chunk,ts,now)
                        self.last_data=now;self.state['is_connected']=True;self.state['sample_age_s']=max(0.,now-ts[-1])
                        self.ingest(raw,ts,now)
                    elif self.last_data is not None and now-self.last_data>1.:
                        raise ValueError('EEG stream timed out; commands stopped. Reconnect and start again.')
            except Exception as exc:
                with self.lock:
                    self.state['error']=str(exc);self.state['is_connected']=False;self.pause('stream_error')
                    if self.inlet: self.inlet.close_stream()
                    self.inlet=None;self.adapter=None;retry_at=self.clock()+1.
        with self.lock:
            self.pause('shutdown')
            if self.decoder: self.decoder.close()
            if self.inlet: self.inlet.close_stream()

class Handler(BaseHTTPRequestHandler):
    studio=None
    def log_message(self,*args): pass
    def respond(self,status,value,content_type='application/json'):
        body=json.dumps(value,allow_nan=False).encode() if content_type=='application/json' else value.encode()
        self.send_response(status);self.send_header('Content-Type',content_type+'; charset=utf-8')
        self.send_header('Content-Length',str(len(body)));self.end_headers();self.wfile.write(body)
    def do_GET(self):
        if self.path in ('/','/index.html'): self.respond(200,(ROOT/'studio.html').read_text(encoding='utf-8'),'text/html')
        elif self.path=='/api/state': self.respond(200,self.studio.snapshot())
        else: self.respond(404,dict(error='Not found'))
    def do_POST(self):
        origin=self.headers.get('Origin')
        if origin and origin not in (f'http://127.0.0.1:{self.server.server_port}',f'http://localhost:{self.server.server_port}'):
            self.respond(403,dict(error='Local console origin required.'));return
        try:
            url=urlparse(self.path)
            result=self.studio.action(url.path,parse_qs(url.query))
            self.respond(200,result)
        except (ValueError,RuntimeError) as exc:
            with self.studio.lock: self.studio.state['error']=str(exc)
            self.respond(409,dict(error=str(exc)))
        except Exception as exc: self.respond(500,dict(error=str(exc)))

def main(default_protocol=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,default=ROOT/'lab_config.json');p.add_argument('--port',type=int,default=8050)
    p.add_argument('--demo',action='store_true');p.add_argument('--no-browser',action='store_true')
    p.add_argument('--mode',choices=['lsl','calibrate','simulate'],default='lsl')
    p.add_argument('--ckpt');p.add_argument('--trials',type=int);p.add_argument('--trial-dur',type=float)
    p.add_argument('--udp-host');p.add_argument('--udp-port',type=int);p.add_argument('--threshold',type=float)
    p.add_argument('--lsl-stream');p.add_argument('--allow-unverified-model',action='store_true')
    p.add_argument('--replay-edf',type=Path,help='Paced recorded-data visualization; sends no hand commands and cannot run prospective validation.')
    args=p.parse_args();config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    for key,value in [('checkpoint',args.ckpt),('calibration_pairs',args.trials),('trial_duration_s',args.trial_dur),('udp_host',args.udp_host),('udp_port',args.udp_port),('threshold',args.threshold),('stream_name',args.lsl_stream)]:
        if value is not None: config[key]=value
    if args.allow_unverified_model: config['allow_unverified_model']=True
    if args.replay_edf: config['replay_edf']=str(args.replay_edf.resolve())
    torch.set_num_threads(int(config.get('torch_threads',1)))
    studio=Studio(config)
    if args.demo or args.mode=='simulate': studio.action('/api/toggle_demo',{})
    if default_protocol or args.mode=='calibrate': print('Open the studio, Start Stream, then select Calibrate. Calibration requires visible cues.')
    handler=type('StudioHandler',(Handler,),{'studio':studio})
    server=ThreadingHTTPServer(('127.0.0.1',args.port),handler)
    worker=threading.Thread(target=studio.worker,daemon=True);worker.start()
    url=f'http://127.0.0.1:{args.port}'
    print(f'BCI Studio: {url}\nReports: {studio.reports}\nModel: {studio.checkpoint}\n{studio.state["error"]}',flush=True)
    if not args.no_browser: webbrowser.open(url)
    try: server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt: pass
    finally:
        studio.training_cancel.set();studio.stop.set();worker.join(timeout=5.);server.server_close()
        if studio.training_thread: studio.training_thread.join(timeout=10.)
        if studio.rows:
            with studio.lock: studio.save()
if __name__=='__main__': main()
