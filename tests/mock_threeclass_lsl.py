"""Synthetic ABM-shaped outlet for transport checks, never accuracy evidence."""
import sys
import time
import numpy as np
from pylsl import StreamInfo, StreamOutlet, local_clock

stock = ['Fp1','Fp2','F7','F3','Fz','F4','F8','T3','C3','Cz','C4','T4',
         'T5','P3','Pz','P4','T6','O1','POz','O2']
labels = ['Epoch','Offset','Hour','Min','Sec','MilliSec']+stock[::-1]+['ECG','AUX1','AUX2','AUX3']
name = sys.argv[1]
info = StreamInfo(name, 'EEG', len(labels), 256, 'float32', name)
channels = info.desc().append_child('channels')
for i, label in enumerate(labels):
    node = channels.append_child('channel')
    node.append_child_value('label', label)
    node.append_child_value('unit', 'uV' if 6 <= i < 26 else 'index')
outlet = StreamOutlet(info, chunk_size=8)
print('READY', flush=True)
rng = np.random.default_rng(814)
target = local_clock()
sample = 0
while True:
    target += 8/256
    time.sleep(max(0., target-local_clock()))
    t = (sample+np.arange(8))/256
    data = np.full((8, 30), 10000., dtype=np.float32)
    for ch in range(20):
        # Microvolt numeric scale; distinct but stationary signals. No task labels
        # or model-class synthetic signal are supplied to the classifier.
        data[:,6+ch] = (3*np.sin(2*np.pi*(10+ch%4*4)*t+ch*.3)
                       +rng.normal(0., 2., len(t)))
    outlet.push_chunk(data, timestamp=target)
    sample += 8
