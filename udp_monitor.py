"""Display virtual-hand UDP commands; returns to REST on missing packets."""
import argparse,json,socket,time

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--port',type=int,default=5005)
    args=p.parse_args();sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    sock.bind(('127.0.0.1',args.port));sock.settimeout(.75)
    print(f'Listening on UDP 127.0.0.1:{args.port}. Close before starting the Unity receiver.',flush=True)
    try:
        while True:
            try:
                packet=json.loads(sock.recvfrom(8192)[0]);print(packet,flush=True)
            except socket.timeout: print('REST (packet timeout)',flush=True)
            except (ValueError,UnicodeDecodeError): print('REST (invalid packet)',flush=True)
    except KeyboardInterrupt: pass
    finally: sock.close()
if __name__=='__main__': main()
