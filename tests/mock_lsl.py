"""Synthetic 30-channel stream for a real LSL transport test. No EEG recording."""
import sys,time
import numpy as np
from pylsl import StreamInfo,StreamOutlet,local_clock
labels=['Epoch','Offset','Hour','Min','Sec','MilliSec']+['O2','O1','POz','P4','Pz','P3','C4','Cz','C3','F4','Fz','F3']+['EKG','AUX1','AUX2','AUX3','Extra1','Extra2','Extra3','Extra4','Extra5','Extra6','Extra7','Extra8']
info=StreamInfo(sys.argv[1],'EEG',30,256,'float32',sys.argv[1])
channels=info.desc().append_child('channels')
for label in labels:
    ch=channels.append_child('channel');ch.append_child_value('label',label);ch.append_child_value('unit','uV')
out=StreamOutlet(info,chunk_size=8)
print('READY',flush=True)
rng=np.random.default_rng(42);target=local_clock()
while True:
    target+=8/256
    time.sleep(max(0,target-local_clock()))
    data=rng.normal(0,5,(8,30)).astype('float32');data[:,:6]=10000
    out.push_chunk(data,timestamp=target)
