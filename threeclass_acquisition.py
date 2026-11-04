"""Identity-targeted discovery for the separate three-class lab application."""
from lsl_acquisition import StreamAdapter, select_stream, stream_metadata


def discover_eeg_streams(config, timeout=1.):
    """Collect matching identities before choosing an EEG source.

    A broad minimum-one EEG resolve can return an unrelated outlet before the
    configured source responds. Query its source ID/name directly instead, and
    wait for a possible duplicate rather than accepting the first response.
    """
    from pylsl import resolve_byprop, resolve_streams
    if config.get('source_id'):
        streams = resolve_byprop('source_id', config['source_id'], minimum=2, timeout=timeout)
    elif config.get('stream_name'):
        streams = resolve_byprop('name', config['stream_name'], minimum=2, timeout=timeout)
    else:
        streams = resolve_streams(wait_time=timeout)
    return [s for s in streams if s.type() == 'EEG']


def connect(config, contract):
    from pylsl import StreamInlet, proc_clocksync
    chosen = select_stream(discover_eeg_streams(config), config)
    inlet = StreamInlet(chosen, max_buflen=2, processing_flags=proc_clocksync, recover=False)
    try:
        adapter = StreamAdapter(stream_metadata(inlet.info(timeout=3.)), contract, config)
        inlet.open_stream(timeout=3.)
        return inlet, adapter
    except Exception:
        inlet.close_stream()
        raise
