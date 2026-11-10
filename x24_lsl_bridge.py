"""Publish real Athena X24 EEG to LSL from the existing x86 mastersp Python."""
import argparse
import faulthandler
import json
from pathlib import Path
import time
import numpy as np
from athena_acquisition import AthenaInlet

ROOT = Path(__file__).resolve().parent
STREAM_NAME = 'BAlert_X24'
SOURCE_ID = 'abm_x24_mastersp'


def make_info(metadata, channels):
    from pylsl import StreamInfo
    info = StreamInfo(STREAM_NAME, 'EEG', len(channels), 256., 'float32', SOURCE_ID)
    desc = info.desc()
    desc.append_child_value('manufacturer', 'Advanced Brain Monitoring')
    desc.append_child_value('device_name', metadata['name'])
    desc.append_child_value('acquisition', '32-bit Athena RAW via mastersp')
    desc.append_child_value('timestamp_method', 'SDK epoch/offset mapped to LSL arrival clock with bounded lead correction; transport latency unmeasured')
    labels = desc.append_child('channels')
    for name in channels:
        channel = labels.append_child('channel')
        channel.append_child_value('label', name)
        channel.append_child_value('unit', 'V')
        channel.append_child_value('type', 'EEG')
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', action='store_true', help='Enable native exception diagnostics (including SDK-handled first-chance exceptions).')
    parser.add_argument('--duration', type=float, default=0, help='Seconds to run; zero runs until stopped.')
    parser.add_argument('--stop-file', type=Path)
    parser.add_argument('--sdk', type=Path, default=Path(r'C:\ABM\B-Alert\SDK'))
    args = parser.parse_args()
    if args.diagnostic:
        faulthandler.enable()
    if not np.isfinite(args.duration) or args.duration < 0:
        parser.error('Duration must be finite and nonnegative.')
    from pylsl import StreamOutlet, local_clock
    contract = json.loads((ROOT / 'decoder_and_models/personal_template_x24_stock20.pt.json').read_text())
    config = dict(input_unit='uV', athena_sdk=str(args.sdk), reports_dir=str(ROOT / 'session_reports/x24_bridge'))
    inlet = AthenaInlet(config, contract, clock=local_clock)
    count, begin, last_data, last_status = 0, local_clock(), local_clock(), 0.
    try:
        outlet = StreamOutlet(make_info(inlet.metadata, contract['channels']), chunk_size=8, max_buffered=2)
        print(json.dumps(dict(status='ready', name=STREAM_NAME, source_id=SOURCE_ID,
                              channels=contract['channels'], sfreq=256, unit='V',
                              device=inlet.device_name, vendor_output=str(inlet.vendor_destination))), flush=True)
        while not args.stop_file or not args.stop_file.exists():
            now = local_clock()
            if args.duration and now - begin >= args.duration:
                break
            chunk, timestamps = inlet.pull_chunk(timeout=.04, max_samples=64)
            now = local_clock()
            if chunk:
                try:
                    raw, ts = inlet.adapter.transform(chunk, timestamps, now)
                except ValueError as exc:
                    raise RuntimeError(f'{exc} last_sample_age_s={now-timestamps[-1]:.6f}; '
                                       f'host_elapsed_s={now-begin:.6f}; '
                                       f'sdk_epoch_offset={chunk[-1][:2]}; pending={len(inlet.pending)}') from exc
                # Supply each original timestamp; never compress real gaps or restamp a backlog.
                outlet.push_chunk(raw.T.tolist(), ts.tolist())
                last_data = now
                count += len(ts)
            elif now - last_data > 1.:
                raise RuntimeError('X24 stopped sending data; LSL bridge stopped.')
            if now - last_status >= 5:
                print(json.dumps(dict(status='streaming', samples=count, seconds=round(now - begin, 3))), flush=True)
                last_status = now
    finally:
        inlet.close_stream()
        print(json.dumps(dict(status='stopped', samples=count)), flush=True)


if __name__ == '__main__':
    main()
