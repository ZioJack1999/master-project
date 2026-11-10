"""Start x86 mastersp acquisition and an x64 motor-imagery GUI."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parent
SOURCE_ID = 'abm_x24_mastersp'


def acquisition_environment():
    prefix = Path.home() / 'miniconda3/envs/mastersp'
    runtime = ROOT / '.runtime/x24-lsl32'
    if not (prefix / 'python.exe').exists() or not (runtime / 'pylsl/lib/lsl.dll').exists():
        raise RuntimeError('mastersp or the local 32-bit pylsl runtime is missing. See X24_CONNECTION_STATUS.md.')
    env = dict(os.environ)
    env['PATH'] = os.pathsep.join(str(p) for p in (prefix, prefix / 'Library/bin', prefix / 'Scripts')) + os.pathsep + env.get('PATH', '')
    env['PYTHONPATH'] = os.pathsep.join((str(runtime), str(ROOT)))
    env['PYTHONNOUSERSITE'] = '1'
    env['ABMSDK'] = r'C:\ABM\B-Alert\SDK'
    return prefix / 'python.exe', env


def probe_stream(config, contract, process, timeout=30.):
    from pylsl import resolve_byprop, StreamInlet, proc_clocksync, local_clock
    from lsl_acquisition import stream_metadata, StreamAdapter
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'X24 bridge exited ({process.returncode}); inspect its log in session_reports/x24_bridge.')
        # Wait for a second matching outlet instead of returning immediately
        # after the first; otherwise the duplicate guard below is ineffective.
        streams = resolve_byprop('source_id', SOURCE_ID, minimum=2, timeout=1.)
        if len(streams) > 1:
            raise RuntimeError('Duplicate X24 outlets found. Close the other bridge before retrying.')
        if not streams:
            continue
        inlet = StreamInlet(streams[0], max_buflen=1, processing_flags=proc_clocksync, recover=False)
        try:
            metadata = stream_metadata(inlet.info(timeout=3.))
            adapter = StreamAdapter(metadata, contract, config)
            inlet.open_stream(timeout=3.)
            inlet.flush()
            samples = 0
            first = last = None
            until = time.monotonic() + 3.
            while time.monotonic() < until:
                chunk, timestamps = inlet.pull_chunk(timeout=.1, max_samples=64)
                if chunk:
                    _, ts = adapter.transform(chunk, timestamps, local_clock())
                    samples += len(ts)
                    first = float(ts[0]) if first is None else first
                    last = float(ts[-1])
                    if samples >= 256:
                        return dict(name=metadata['name'], source_id=metadata['source_id'],
                                    channels=metadata['labels'], unit=metadata['units'][0], samples=samples,
                                    nominal_srate=metadata['sfreq'], measured_srate=(samples - 1)/(last - first))
            raise RuntimeError('X24 LSL outlet did not deliver fresh samples.')
        finally:
            inlet.close_stream()
    raise RuntimeError('Timed out waiting for the X24 bridge. Check headset power and pairing.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['binary', 'threeclass', 'multiclass'], default='binary')
    parser.add_argument('--port', type=int)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], help='Three-class model device (default: auto).')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--check-only', action='store_true', help='Probe both processes and exit without starting the GUI.')
    parser.add_argument('--verify-only', action='store_true', help='Verify the distribution without starting acquisition.')
    args = parser.parse_args()
    if struct.calcsize('P') != 8:
        parser.error('The supervisor/decoder must use 64-bit Python; only acquisition uses mastersp.')
    if args.task == 'binary' and args.device is not None:
        parser.error('--device is supported by the three-class GUI.')
    if args.task == 'multiclass':
        from multiclass_lab_bundle import verify_bundle
        config = verify_bundle(ROOT)
        count = len(config['classes'])
        task_name, default_port = f'{count}class', 8050+count
    else:
        from verify_binary_release import verify
        if not verify()['binary_release_unchanged']:
            raise RuntimeError('Original binary release integrity check failed.')
        config_name = 'lab_config.json' if args.task == 'binary' else 'threeclass_config.json'
        config = json.loads((ROOT / config_name).read_text(encoding='utf-8-sig'))
        task_name, default_port = args.task, 8052 if args.task == 'binary' else 8051
    if args.port is None:
        args.port = default_port
    config.update(stream_name='BAlert_X24', source_id=SOURCE_ID, input_unit='', channel_labels=None)
    if args.task == 'binary':
        from realtime_bci_decoder import load_contract, RealtimeBCIDecoder
        checkpoint = ROOT / config['checkpoint']
        contract = load_contract(checkpoint)
        decoder = RealtimeBCIDecoder(checkpoint, send_udp=False)
        decoder.close()
    elif args.task == 'threeclass':
        from verify_threeclass_bundle import verify as verify_threeclass
        from threeclass_preflight import inspect
        if not verify_threeclass()['threeclass_package_valid']:
            raise RuntimeError('Three-class runtime integrity check failed.')
        result = inspect(config, ROOT)
        if not result['configuration_valid'] or not result['reference_compatible']:
            raise RuntimeError('Invalid three-class configuration: ' + json.dumps(result))
        from multiclass_decoder import MulticlassDecoder
        import numpy as np
        import torch
        requested = args.device or 'auto'
        cuda_available = requested != 'cpu' and torch.cuda.is_available()
        if requested == 'cuda' and not cuda_available:
            raise RuntimeError('CUDA is unavailable to this PyTorch runtime. Check the CUDA wheel and NVIDIA driver.')
        config['device'] = 'cuda:0' if cuda_available else 'cpu'
        decoder = MulticlassDecoder(ROOT / config['checkpoint'], device=config['device'])
        if cuda_available:
            window = np.random.default_rng(0).normal(0, 5e-6, (decoder.n_ch, decoder.window_samples))
            result = decoder.process_chunk_all(window)
            if len(result) != 1 or result[0]['quality'] != 'ok':
                raise RuntimeError('CUDA model warm-up did not produce a prediction.')
            print(f'Three-class model on {config["device"]}: {torch.cuda.get_device_name(0)}', flush=True)
        else:
            print('Three-class model on CPU.', flush=True)
        del decoder
        contract = dict(channels=config['channels'], sfreq=config['sfreq'])
    else:
        contract = dict(channels=config['channels'], sfreq=config['sfreq'])
    if args.verify_only:
        print(json.dumps(dict(verified=True, task=task_name, classes=config.get('classes', contract.get('classes')))))
        return
    from pylsl import resolve_byprop
    if resolve_byprop('source_id', SOURCE_ID, timeout=1.):
        raise RuntimeError('An X24 bridge is already running. Use its decoder GUI or stop it before launching another.')
    with socket.socket() as test_socket:
        test_socket.bind(('127.0.0.1', args.port))
    # Preserve any user-local config: use a separate ignored run-specific config.
    directory = ROOT / 'session_reports/x24_bridge'
    directory.mkdir(parents=True, exist_ok=True)
    run_id = task_name + '_' + time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    config_path = directory / (run_id + '.config.json')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    stop_file = directory / (run_id + '.stop')
    python32, env = acquisition_environment()
    with (directory / (run_id + '.log')).open('w', encoding='utf-8') as log:
        bridge = subprocess.Popen([str(python32), '-u', str(ROOT / 'x24_lsl_bridge.py'),
                                   '--stop-file', str(stop_file)], cwd=ROOT, env=env,
                                  stdout=log, stderr=subprocess.STDOUT,
                                  creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        try:
            report = probe_stream(config, contract, bridge)
            report.update(python32=str(python32), python64=sys.executable,
                          bridge_pid=bridge.pid, task=task_name,
                          packages={name: importlib.metadata.version(name) for name in
                                    ['numpy', 'scipy', 'torch', 'pylsl', 'pandas', 'scikit-learn']})
            if args.task != 'multiclass':
                report['binary_release_unchanged'] = True
            if args.task == 'threeclass':
                report['model_device'] = config['device']
            (directory / (run_id + '.preflight.json')).write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps(report, indent=2), flush=True)
            if args.check_only:
                return
            if args.task == 'binary':
                from bci_studio_gui import main as studio_main
            else:
                from threeclass_studio import main as studio_main
            sys.argv = ['studio', '--config', str(config_path), '--port', str(args.port)]
            if args.no_browser:
                sys.argv.append('--no-browser')
            print('X24 ready. Start Stream to inspect signals, then Personal calibration. Ctrl+C stops both processes.', flush=True)
            studio_main()
        finally:
            stop_file.write_text('stop', encoding='ascii')
            try:
                bridge.wait(timeout=15.)
            except subprocess.TimeoutExpired:
                bridge.terminate()
                bridge.wait(timeout=5.)
            if bridge.returncode:
                print(f'X24 bridge exited with code {bridge.returncode}. See {log.name}', file=sys.stderr)


if __name__ == '__main__':
    main()
