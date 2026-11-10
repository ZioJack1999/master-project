"""Direct x86 Athena acquisition, with named channels and SDK sample continuity.

The monotonic clock starts at initial packet arrival and applies bounded
corrections when the device clock runs ahead. Device transport latency is unmeasured.
"""
import ctypes as ct
from datetime import datetime
from pathlib import Path
import os
import struct
import time
import numpy as np
from lsl_acquisition import StreamAdapter


class Device(ct.Structure):
    _pack_ = 1
    _fields_ = [('name', ct.c_wchar * 256), ('comm_port', ct.c_int),
                ('ecg_position', ct.c_int), ('channels', ct.c_int),
                ('esu_type', ct.c_int), ('timestamp_type', ct.c_int),
                ('device_handle', ct.c_int), ('device_id', ct.c_wchar * 260)]


class Channels(ct.Structure):
    _pack_ = 1
    _fields_ = [('names', (ct.c_char * 20) * 24), ('used', ct.c_int * 24),
                ('quality_used', ct.c_int * 24), ('decontaminated', ct.c_int * 24),
                ('is_eeg', ct.c_int * 24), ('is_flex', ct.c_int), ('channel_map', ct.c_int)]


def sample_counters(packet):
    """_ABM_DATA_TIME epoch (seconds) and within-second sample offset."""
    values = np.asarray(packet, dtype=float)
    if values.ndim != 2 or values.shape[1] != 30 or not np.isfinite(values).all():
        raise ValueError('Invalid Athena RAW packet.')
    epoch, offset = values[:, 0], values[:, 1]
    if np.any(np.abs(epoch - np.rint(epoch)) > .01) or np.any(np.abs(offset - np.rint(offset)) > .01):
        raise ValueError('Nonintegral Athena epoch/offset.')
    if np.any(epoch < 0) or np.any(offset < 0) or np.any(offset > 256):
        raise ValueError('Invalid Athena epoch/offset range.')
    counters = np.rint(epoch).astype(np.int64) * 256 + np.rint(offset).astype(np.int64)
    if np.any(np.diff(counters) <= 0):
        raise ValueError('Athena sample counter moved backwards or repeated.')
    return counters


class AthenaInlet:
    def __init__(self, config, contract, clock=time.perf_counter):
        if struct.calcsize('P') != 4:
            raise RuntimeError('Athena must run in the existing 32-bit mastersp Python.')
        if contract['sfreq'] != 256:
            raise ValueError('The X24 RAW backend supplies 256 Hz only.')
        if config.get('input_unit') != 'uV':
            raise ValueError('Confirm the Athena RAW numeric unit and set input_unit to uV.')
        root = Path(config.get('athena_sdk', r'C:\ABM\B-Alert\SDK')).resolve()
        self.handles = [os.add_dll_directory(str(p)) for p in
                        (root / 'bin', root.parent / 'Rome') if p.is_dir()]
        self.dll = ct.WinDLL(str(root / 'bin' / 'ABM_Athena.dll'))
        self.started = False
        self.closed = False
        self.clock = clock
        self.last_counter = None
        self.anchor = None
        self.pending = np.empty((0, 30), dtype=np.float32)
        self.pending_times = np.empty(0)
        def function(name, args, result=ct.c_int):
            fn = getattr(self.dll, name)
            fn.argtypes, fn.restype = args, result
            return fn
        self.set_config = function('SetConfigPath', [ct.c_wchar_p])
        self.find = function('GetDeviceInfoKeepConnection', [ct.POINTER(Device)], ct.POINTER(Device))
        self.close = function('CloseCurrentConnection', [])
        self.init = function('InitSessionForCurrentConnection', [ct.c_int] * 4)
        self.start = function('StartAcquisitionForCurrentConnection', [])
        self.stop = function('StopAcquisition', [])
        self.destination = function('SetDestinationFile', [ct.c_wchar_p])
        self.channel_info = function('GetEEGChannelsInfo', [ct.POINTER(Channels)])
        self.packet_info = function('GetPacketChannelNmbInfo', [ct.POINTER(ct.c_int)] * 5)
        self.raw_data = function('GetRawData', [ct.POINTER(ct.c_int)], ct.POINTER(ct.c_float))
        try:
            if self.set_config(str(root.parent / 'Config') + '\\') != 1:
                raise RuntimeError('Athena rejected its configuration folder.')
            pointer = self.find(None)
            if not pointer:
                raise RuntimeError('X24 not detected. Check power, ESU pairing and exclusive device owners.')
            # Keep the connection's device-info storage owned by the SDK too.
            self.device_pointer = pointer
            device = Device.from_buffer_copy(ct.string_at(pointer, ct.sizeof(Device)))
            self.device_name = device.name
            if not device.name.startswith('X24 10-20 ') or device.channels != 24:
                raise ValueError('This direct backend supports the first-generation X24 10-20 profile only.')
            directory = Path(config.get('reports_dir', 'session_reports')).resolve() / 'athena_vendor'
            directory.mkdir(parents=True, exist_ok=True)
            self.vendor_destination = directory / (datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.edf')
            if self.destination(str(self.vendor_destination)) != 1:
                raise RuntimeError('Athena rejected its private output destination.')
            if self.init(0, 0, -1, 0) != 1:
                raise RuntimeError('Athena RAW session initialization failed.')
            widths = [ct.c_int() for _ in range(5)]
            if self.packet_info(*(ct.byref(v) for v in widths)) != 1 or widths[0].value != 24:
                raise ValueError('Unexpected Athena RAW packet channel count.')
            info = Channels()
            if self.channel_info(ct.byref(info)) != 1:
                raise RuntimeError('Athena did not report the active channel map.')
            labels = [bytes(name).split(b'\0')[0].decode('ascii').strip() for name in info.names]
            if any(not labels[i] or not info.used[i] for i in range(20)):
                raise ValueError('Athena channel map has missing/disabled EEG sites.')
            self.metadata = dict(name=device.name, source_id='athena:' + device.name,
                type='EEG', sfreq=256., channel_count=30,
                labels=['Epoch', 'Offset', 'Hour', 'Min', 'Sec', 'MilliSec'] + labels,
                units=[''] * 6 + ['uV'] * 24, transport='athena_direct',
                timestamp_method='SDK epoch/offset mapped to monotonic arrival clock with bounded lead correction',
                vendor_destination=str(self.vendor_destination))
            # Validate channels before starting acquisition.
            self.adapter = StreamAdapter(self.metadata, contract, config)
            if self.start() != 1:
                raise RuntimeError('Athena did not start RAW acquisition.')
            self.started = True
        except Exception:
            self.close_stream()
            raise

    def _read(self):
        size = ct.c_int()
        pointer = self.raw_data(ct.byref(size))
        if size.value < 0 or size.value > 256 * 60:
            raise ValueError('Invalid Athena RAW sample count.')
        if not pointer or size.value == 0:
            return
        # This installed SDK's GetRawData pointer is SDK-owned. Copy before the
        # next call. FreeBuffer is for separately allocated return buffers;
        # applying it here corrupts the vendor heap (verified on this build).
        packet = np.ctypeslib.as_array(pointer, shape=(size.value * 30,)).reshape(size.value, 30).copy()
        now = self.clock()
        counters = sample_counters(packet)
        previous = self.last_counter
        if previous is not None and counters[0] <= previous:
            raise ValueError('Athena sample counter reset; reconnect required.')
        self.last_counter = int(counters[-1])
        if self.anchor is None:
            self.anchor = now - counters[-1] / 256.
        elif previous is not None and self.last_counter-previous == len(counters):
            lead = self.anchor + counters[-1]/256. - now
            if lead > .010:
                # At most 1 ms of correction per second of contiguous samples.
                # Leave real gaps and old/backlogged packets visible to freshness checks.
                self.anchor -= min(lead-.010, .001*min(len(counters), 64)/256.)
        timestamps = self.anchor + counters / 256.
        self.pending = np.concatenate((self.pending, packet))
        self.pending_times = np.r_[self.pending_times, timestamps]

    def pull_chunk(self, timeout=.05, max_samples=64):
        end = time.perf_counter() + timeout
        while len(self.pending) == 0:
            self._read()
            if len(self.pending) or time.perf_counter() >= end:
                break
            time.sleep(.005)
        n = min(len(self.pending), max_samples)
        chunk, timestamps = self.pending[:n].tolist(), self.pending_times[:n].tolist()
        self.pending, self.pending_times = self.pending[n:], self.pending_times[n:]
        return chunk, timestamps

    def flush(self):
        self._read()
        self.pending = np.empty((0, 30), dtype=np.float32)
        self.pending_times = np.empty(0)

    def close_stream(self):
        if self.closed:
            return
        self.closed = True
        try:
            if self.started:
                if self.stop() != 1:
                    raise RuntimeError('Athena did not acknowledge stopping acquisition.')
        finally:
            self.close()
            self.started = False


def connect(config, contract):
    inlet = AthenaInlet(config, contract)
    return inlet, inlet.adapter
