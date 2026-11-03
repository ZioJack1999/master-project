"""End-to-end real liblsl -> adapter -> GUI HTTP -> model -> UDP."""
import json,socket,subprocess,sys,threading,time,urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
import numpy as np
from bci_studio_gui import Studio,Handler
from test_pipeline import checkpoint

def test_lsl_http_udp_and_disconnect(checkpoint,tmp_path):
    name='BCI_Audit_'+str(time.time_ns())
    sender=subprocess.Popen([sys.executable,str(Path(__file__).with_name('mock_lsl.py')),name],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    receiver=socket.socket(socket.AF_INET,socket.SOCK_DGRAM);receiver.bind(('127.0.0.1',0));receiver.settimeout(5.)
    s=Studio(dict(checkpoint=str(checkpoint[0]),stream_name=name,source_id=name,udp_port=receiver.getsockname()[1],reports_dir=str(tmp_path/'reports')))
    handler=type('IntegrationHandler',(Handler,),{'studio':s})
    server=ThreadingHTTPServer(('127.0.0.1',0),handler)
    worker=threading.Thread(target=s.worker,daemon=True);http=threading.Thread(target=server.serve_forever,daemon=True)
    base=f'http://127.0.0.1:{server.server_port}'
    def api(path,post=False):
        req=urllib.request.Request(base+path,method='POST' if post else 'GET')
        with urllib.request.urlopen(req,timeout=10.) as response: return json.load(response)
    try:
        assert sender.stdout.readline().strip()=='READY'
        worker.start();http.start()
        deadline=time.monotonic()+12.
        while not s.state['is_connected'] and time.monotonic()<deadline: time.sleep(.05)
        assert s.state['is_connected'],s.state['error']
        api('/api/toggle_stream',True)
        packet=json.loads(receiver.recvfrom(8192)[0])
        assert packet['sample_end']==512 and packet['quality']=='ok'
        api('/api/set_threshold?val=.7',True)
        deadline=time.monotonic()+3.
        while packet.get('threshold')!=.7 and time.monotonic()<deadline: packet=json.loads(receiver.recvfrom(8192)[0])
        assert packet['threshold']==.7
        assert api('/api/state')['step_count']>0
        api('/api/stop',True)
        while packet.get('quality')!='cancelled': packet=json.loads(receiver.recvfrom(8192)[0])
        assert packet['grasp_command']==0 and packet['continuous_flex']==0.
        steps=s.state['step_count'];api('/api/toggle_stream',True);time.sleep(.1)
        assert s.state['step_count']==steps
        sender.terminate();sender.wait(timeout=5.)
        deadline=time.monotonic()+4.
        while s.state['is_streaming'] and time.monotonic()<deadline: time.sleep(.05)
        assert not s.state['is_streaming'] and not s.state['is_connected']
        assert s.state['grasp_command']==0
        assert s.raw and np.max(np.abs(np.concatenate(s.raw,axis=1)))<100e-6
        saved=api('/api/export_figure',True)
        assert saved['status']=='saved'
    finally:
        s.stop.set()
        if worker.is_alive(): worker.join(timeout=5.)
        server.shutdown();server.server_close();receiver.close()
        if sender.poll() is None: sender.terminate();sender.wait(timeout=5.)
