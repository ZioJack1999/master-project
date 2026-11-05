"""Detect an X24; optionally check a short raw acquisition without a decoder."""
import argparse
import ctypes as ct
from datetime import datetime
import json
import math
import os
from pathlib import Path
import struct
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sdk', type=Path, default=Path(r'C:\ABM\B-Alert\SDK'))
    parser.add_argument('--sample-seconds', type=float, default=0,
                        help='Optional 1-30 second RAW transport check; SDK files stay in session_reports.')
    args = parser.parse_args()
    if args.sample_seconds != 0 and not 1 <= args.sample_seconds <= 30:
        parser.error('--sample-seconds must be zero (detection only) or 1-30')
    if struct.calcsize('P') != 4:
        raise SystemExit('Run with the 32-bit mastersp Python (Athena is x86).')
    root = args.sdk.resolve()
    # Keep DLL-directory handles alive throughout all SDK calls.
    handles = [os.add_dll_directory(str(p)) for p in
               (root / 'bin', root.parent / 'Rome') if p.is_dir()]
    dll = ct.WinDLL(str(root / 'bin' / 'ABM_Athena.dll'))

    class Device(ct.Structure):
        _pack_ = 1
        _fields_ = [('name', ct.c_wchar * 256), ('comm_port', ct.c_int),
                    ('ecg_position', ct.c_int), ('channels', ct.c_int),
                    ('esu_type', ct.c_int), ('timestamp_type', ct.c_int),
                    ('device_handle', ct.c_int), ('device_id', ct.c_wchar * 260)]

    dll.SetConfigPath.argtypes = [ct.c_wchar_p]
    dll.SetConfigPath.restype = ct.c_int
    dll.GetDeviceInfoKeepConnection.argtypes = [ct.POINTER(Device)]
    dll.GetDeviceInfoKeepConnection.restype = ct.POINTER(Device)
    dll.CloseCurrentConnection.argtypes = []
    dll.CloseCurrentConnection.restype = ct.c_int
    callback_type = ct.WINFUNCTYPE(None, ct.c_wchar_p, ct.c_int)

    def message(text, code):
        print(json.dumps({'detection_message': text, 'code': code}), flush=True)

    callback = callback_type(message)
    dll.RegisterCallbackDeviceDetectionInfo.argtypes = [callback_type]
    dll.RegisterCallbackDeviceDetectionInfo.restype = ct.c_int
    dll.RegisterCallbackDeviceDetectionInfo(callback)
    config_status = dll.SetConfigPath(str(root.parent / 'Config') + '\\')
    print(json.dumps({'config_status': config_status, 'sdk': str(root)}), flush=True)
    if config_status != 1:
        raise SystemExit('Athena rejected the installed configuration path.')
    started = False
    try:
        print('Detecting headset (no recording or impedance check)...', flush=True)
        pointer = dll.GetDeviceInfoKeepConnection(None)
        if not pointer:
            raise SystemExit('Athena did not detect a headset. Check power, pairing and SDK/license status.')
        device = pointer.contents
        print(json.dumps({'detected': True, 'name': device.name,
                          'channels': device.channels, 'esu_type': device.esu_type,
                          'comm_port': device.comm_port}, indent=2), flush=True)
        if args.sample_seconds:
            if not device.name.startswith('X24 10-20 ') or device.channels != 24:
                raise SystemExit('RAW probe supports only the detected first-generation X24 10-20 profile.')
            dll.InitSessionForCurrentConnection.argtypes = [ct.c_int] * 4
            dll.InitSessionForCurrentConnection.restype = ct.c_int
            dll.SetDestinationFile.argtypes = [ct.c_wchar_p]
            dll.SetDestinationFile.restype = ct.c_int
            dll.StartAcquisitionForCurrentConnection.argtypes = []
            dll.StartAcquisitionForCurrentConnection.restype = ct.c_int
            dll.StopAcquisitionKeepConnection.argtypes = []
            dll.StopAcquisitionKeepConnection.restype = ct.c_int
            dll.GetPacketChannelNmbInfo.argtypes = [ct.POINTER(ct.c_int)] * 5
            dll.GetPacketChannelNmbInfo.restype = ct.c_int
            dll.GetRawData.argtypes = [ct.POINTER(ct.c_int)]
            dll.GetRawData.restype = ct.POINTER(ct.c_float)
            destination = Path(__file__).resolve().parents[1] / 'session_reports' / 'connection_checks'
            destination.mkdir(parents=True, exist_ok=True)
            destination /= datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.edf'
            if dll.SetDestinationFile(str(destination)) != 1:
                raise SystemExit('SDK rejected the private recording destination.')
            # EegAcqDefine.h: X24QEEG/10-20=0, RAW session=0; live hardware, no playback.
            status = dll.InitSessionForCurrentConnection(0, 0, -1, 0)
            print(json.dumps({'init_session_status': status}), flush=True)
            if status != 1:
                raise SystemExit('SDK did not initialize RAW acquisition.')
            widths = [ct.c_int() for _ in range(5)]
            status = dll.GetPacketChannelNmbInfo(*(ct.byref(v) for v in widths))
            channels = widths[0].value
            # TypeDefDataTypes.h _ABM_DATA_SAMPLES: six timestamp floats,
            # followed by nRawPacketChannelsNmb signal floats.
            width = channels + 6
            print(json.dumps({'packet_info_status': status, 'raw_signal_channels': channels,
                              'raw_packet_fields': width}), flush=True)
            if status != 1 or channels != device.channels:
                raise SystemExit('SDK returned an invalid RAW packet geometry.')
            status = dll.StartAcquisitionForCurrentConnection()
            print(json.dumps({'start_status': status}), flush=True)
            if status != 1:
                raise SystemExit('SDK did not start RAW acquisition.')
            started = True
            count = 0
            nonfinite = 0
            begin = time.monotonic()
            while time.monotonic() - begin < args.sample_seconds:
                size = ct.c_int()
                data = dll.GetRawData(ct.byref(size))
                if size.value < 0 or size.value > 256 * 60:
                    raise RuntimeError('SDK returned an invalid sample count.')
                if data and size.value:
                    nonfinite += sum(not math.isfinite(data[i]) for i in range(size.value * width))
                    count += size.value
                time.sleep(.02)
            print(json.dumps({'raw_samples_received': count, 'nonfinite_fields': nonfinite,
                              'observation_seconds': time.monotonic() - begin,
                              'sdk_output': str(destination),
                              'scope': 'Transport only; montage, reference, numeric units, timestamps and electrode quality still require validation.'},
                             indent=2), flush=True)
            if count == 0 or nonfinite:
                raise SystemExit('RAW transport check failed.')
    finally:
        try:
            if started:
                print(json.dumps({'stop_status': dll.StopAcquisitionKeepConnection()}), flush=True)
        finally:
            print(json.dumps({'connection_closed': dll.CloseCurrentConnection()}), flush=True)


if __name__ == '__main__':
    main()
