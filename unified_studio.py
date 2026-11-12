"""One browser console for the independent motor-imagery profiles."""

import argparse
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import threading
import webbrowser

import torch

from bci_studio_gui import Studio as BinaryStudio
from multiclass_lab_bundle import CLASS_SETS, expanded_config
from threeclass_studio import Handler, ThreeClassStudio


ROOT = Path(__file__).resolve().parent
PROFILES = {
    'threeclass': 'Three-class',
    'binary': 'Binary',
    **{f'{count}class': f'{count}-class (experimental)' for count in CLASS_SETS},
}
STREAM_SETTINGS = ('stream_name', 'source_id', 'input_unit', 'channel_labels')


class UnifiedStudio:
    def __init__(self, threeclass_config, binary_config, root=ROOT, *, demo=False):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.threeclass_config = dict(threeclass_config)
        self.binary_config = dict(binary_config)
        self.binary_config.update({key: threeclass_config[key] for key in STREAM_SETTINGS
                                   if key in threeclass_config})
        self.profile = 'threeclass'
        self.revision = 0
        self.saved_recording = None
        self.engine = self.build_engine(self.profile, demo)
        self.worker = self.start_worker(self.engine)

    def build_engine(self, profile, demo):
        if profile == 'binary':
            engine = BinaryStudio(self.binary_config.copy(), root=self.root)
        else:
            config = (self.threeclass_config if profile == 'threeclass'
                      else expanded_config(self.threeclass_config, int(profile.removesuffix('class'))))
            engine = ThreeClassStudio(config, root=self.root)
        if demo:
            engine.action('/api/demo')
        return engine

    @staticmethod
    def start_worker(engine):
        kwargs = {'save_on_stop': False} if isinstance(engine, ThreeClassStudio) else {}
        worker = threading.Thread(target=engine.worker, kwargs=kwargs, daemon=True)
        worker.start()
        return worker

    def snapshot(self, seen=False):
        with self.lock:
            return dict(self.engine.snapshot(seen=seen), profile=self.profile,
                        profile_revision=self.revision, profiles=PROFILES)

    def recording(self):
        engine = self.engine
        return (engine.session_id, len(engine.raw), len(engine.events),
                len(engine.predictions), len(engine.results))

    def save_recording(self):
        recording = self.recording()
        if any(recording[1:]) and recording != self.saved_recording:
            self.engine.save()
            self.saved_recording = recording

    def switch(self, profile):
        if not isinstance(profile, str) or profile not in PROFILES:
            raise ValueError('Select a supported class profile.')
        if profile == self.profile:
            return self.snapshot()
        engine = self.engine
        with engine.lock:
            if (engine.state['streaming'] or engine.protocol or
                    engine.state['training'] or engine.state.get('is_training')):
                raise ValueError('Stop the stream or finish fitting before switching profiles.')
            self.save_recording()
            replacement = self.build_engine(profile, engine.state['mode'] == 'demo')
            engine.stop_event.set()
        self.worker.join()
        self.engine = replacement
        self.profile = profile
        self.revision += 1
        self.saved_recording = None
        self.worker = self.start_worker(replacement)
        return self.snapshot()

    def action(self, path, body):
        with self.lock:
            if body.get('profile_revision', self.revision) != self.revision:
                raise ValueError('The class profile changed. Retry in the current profile.')
            if path == '/api/profile':
                return self.switch(body.get('profile'))
            with self.engine.lock:
                result = self.engine.action(path, body)
                if path in ('/api/save', '/api/export_figure'):
                    self.saved_recording = self.recording()
                return result

    def close(self):
        with self.lock:
            self.engine.cancel_fit.set()
            self.engine.stop_event.set()
            self.worker.join()
            training = self.engine.training_thread
            if training:
                training.join()
            with self.engine.lock:
                self.save_recording()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'threeclass_config.json')
    parser.add_argument('--port', type=int, default=8051)
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8-sig'))
    binary = json.loads((ROOT/'lab_config.json').read_text(encoding='utf-8-sig'))
    torch.set_num_threads(int(config.get('torch_threads', 1)))
    with ThreadingHTTPServer(('127.0.0.1', args.port), Handler) as server:
        studio = UnifiedStudio(config, binary, demo=args.demo)
        handler = type('UnifiedHandler', (Handler,), {'studio': studio})
        server.RequestHandlerClass = handler
        url = f'http://127.0.0.1:{server.server_port}'
        print(f'BCI Studio: {url}', flush=True)
        try:
            if not args.no_browser:
                webbrowser.open(url)
                if args.demo:
                    from virtual_hand_viewer import launch_viewer_window
                    launch_viewer_window(f'{url}/virtual_hand')
            server.serve_forever(poll_interval=.2)
        except KeyboardInterrupt:
            pass
        finally:
            studio.close()


if __name__ == '__main__':
    main()
