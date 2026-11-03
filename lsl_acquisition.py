"""LSL stream selection, metadata mapping, units, and timestamp checks."""
import time
import xml.etree.ElementTree as ET
import numpy as np

UNITS={'v':1.,'volt':1.,'volts':1.,'uv':1e-6,'µv':1e-6,'μv':1e-6,'microvolt':1e-6,'microvolts':1e-6,'mv':1e-3}

def stream_metadata(info):
    root=ET.fromstring(info.as_xml())
    channels=root.findall('./desc/channels/channel')
    return dict(name=info.name(),source_id=info.source_id(),type=info.type(),sfreq=info.nominal_srate(),
                channel_count=info.channel_count(),labels=[c.findtext('label','') for c in channels],
                units=[c.findtext('unit','') for c in channels],xml=info.as_xml())

def select_stream(streams,config):
    chosen=[s for s in streams if (not config.get('stream_name') or s.name()==config['stream_name'])
            and (not config.get('source_id') or s.source_id()==config['source_id'])]
    if len(chosen)!=1:
        raise ValueError(f'Expected one EEG stream, found {len(chosen)}. Set stream_name/source_id in lab_config.json.')
    return chosen[0]

class StreamAdapter:
    def __init__(self,metadata,contract,config=None):
        config=config or {}
        self.metadata=metadata
        self.sfreq=float(contract['sfreq'])
        if not np.isclose(metadata['sfreq'],self.sfreq):
            raise ValueError(f"Stream is {metadata['sfreq']} Hz; model requires {self.sfreq} Hz. Resampling is not implemented.")
        self.count=int(metadata['channel_count'])
        labels=config.get('channel_labels') or metadata['labels']
        if len(labels)!=self.count: raise ValueError('Missing channel labels: supply the verified physical channel_labels in config.')
        normalized=[x.strip().casefold() for x in labels]
        self.indices=[]
        for expected in contract['channels']:
            matches=[i for i,label in enumerate(normalized) if label==expected.casefold()]
            if len(matches)!=1:
                raise ValueError(f'Missing or duplicate electrode {expected}. Match the actual montage or retrain; do not relabel other sites.')
            self.indices.append(matches[0])
        units=metadata.get('units',[])
        self.scales=[]
        for i in self.indices:
            unit=config.get('input_unit') or (units[i] if i<len(units) else '')
            key=unit.strip().casefold()
            if key not in UNITS: raise ValueError(f'Unknown EEG unit {unit!r}; set input_unit only after checking the SDK/recorded signal.')
            self.scales.append(UNITS[key])
        self.max_age=float(config.get('max_sample_age_s',1.))
        self.max_gap=float(config.get('max_gap_s',.1))
        self.last_timestamp=None
    def reset_timestamps(self): self.last_timestamp=None
    def transform(self,chunk,timestamps,now):
        x=np.asarray(chunk,dtype=np.float64)
        ts=np.asarray(timestamps,dtype=float)
        if x.ndim!=2 or x.shape!=(len(ts),self.count) or not len(ts): raise ValueError('LSL data/timestamp shape mismatch.')
        if not np.isfinite(ts).all() or np.any(np.diff(ts)<=0): raise ValueError('Invalid or nonmonotonic timestamps.')
        if self.last_timestamp is not None and ts[0]<=self.last_timestamp: raise ValueError('Timestamp clock moved backwards.')
        diffs=np.diff(np.r_[self.last_timestamp,ts]) if self.last_timestamp is not None else np.diff(ts)
        if np.any(diffs>self.max_gap): raise ValueError('EEG samples were dropped or the stream stalled; decoder reset.')
        if now-ts[-1]>self.max_age or ts[-1]-now>.1: raise ValueError('Stale or unsynchronized EEG timestamps; check LSL clocks/network.')
        out=(x[:,self.indices]*np.asarray(self.scales)).T.astype(np.float32)
        if not np.isfinite(out).all(): raise ValueError('Nonfinite EEG input.')
        self.last_timestamp=float(ts[-1])
        return out,ts

def connect(config,contract):
    from pylsl import StreamInlet,resolve_byprop,proc_clocksync
    streams=resolve_byprop('type','EEG',timeout=1.)
    chosen=select_stream(streams,config)
    inlet=StreamInlet(chosen,max_buflen=2,processing_flags=proc_clocksync,recover=False)
    try:
        info=inlet.info(timeout=3.)
        adapter=StreamAdapter(stream_metadata(info),contract,config)
        inlet.open_stream(timeout=3.)
        return inlet,adapter
    except Exception:
        inlet.close_stream()
        raise
