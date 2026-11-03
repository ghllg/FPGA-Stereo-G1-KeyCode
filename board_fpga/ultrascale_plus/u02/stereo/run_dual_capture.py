"""Functional two-camera concurrent capture on current Linux, without PL."""
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import socket
import sys
import tarfile
import time
import uuid
import argparse

U02=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(U02.parents[1]/'bringup/axu5evb_e_boot'))
sys.path.insert(0,str(U02/'inference'))
from uart_console import Console
from validate_image_pipeline_resident import connect_file_server

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
query=U02/'reports/stereo_board_camera_query_1791812950301773100/state.json'
inventory=json.loads(query.read_text())
assert inventory['complete'] and inventory['capture_devices']==['/dev/video0','/dev/video2']
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--small',action='store_true',help='One verified lower-resolution fallback after USB bandwidth rejection')
parser.add_argument('--yuyv',action='store_true',help='Bounded raw YUYV 320x240/30 diagnostic, without changing the production MJPEG entry')
parser.add_argument('--hd',action='store_true',help='Required 1280x720 MJPEG mode')
parser.add_argument('--frames',type=int,default=90,help='2..1200 real MJPEG frames per camera')
parser.add_argument('--candidate-uvc',action='store_true',help='Explicitly verify an already-bound isolated 720p UVC candidate')
parser.add_argument('--record-all',action='store_true',help='Save every real720p JPEG, total bounded to128MiB')
parser.add_argument('--armed',action='store_true',help='Wait for local start_capture after both real cameras are ready')
args=parser.parse_args()
assert sum((args.small,args.yuyv,args.hd))<=1 and 2<=args.frames<=1200
assert not args.candidate_uvc or args.hd
assert not args.record_all or args.hd and args.frames<=600
assert not args.armed or args.record_all
frames=30 if args.yuyv else args.frames
width,height=(1280,720) if args.hd else (320,240) if args.small or args.yuyv else (640,480)
binary=U02/'reports'/('test_dual_sync_record' if args.record_all else 'probe_dual_yuyv' if args.yuyv else 'test_dual_capture_720p' if args.hd else 'test_dual_capture_small' if args.small else 'test_dual_capture')
sources=[binary,Path(__file__),U02/'stereo'/('probe_dual_yuyv.cpp' if args.yuyv else 'test_dual_capture.cpp'),
    U02/'inference/host/camera_capture.cpp',U02/'inference/host/camera_capture.h',query]
frozen={str(path):sha(path) for path in sources}
raw=binary.read_bytes();assert len(raw)<4*1024*1024
remote='/tmp/u02_'+uuid.uuid4().hex
output=U02/('reports/stereo_dual_capture_'+str(time.time_ns()));output.mkdir()
state=dict(complete=False,remote=remote,input_sha256=frozen,frames_requested_per_camera=frames,
    requested_size=[width,height],
    requested_format='YUYV' if args.yuyv else 'MJPEG',
    existing_capture_driver='u02_uvcvideo' if args.candidate_uvc else 'uvcvideo',
    record_all_frames=args.record_all,armed=False,
    board_programmed=False,modules_loaded=False,reboot=False,startup_files_modified=False,
    whole_network_inference=False,exposure_sync_verified=False,performance_benchmark=False)
def save():(output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
try:
    with (output/'uart.log').open('wb') as log:
        console=Console('COM9',log)
        try:
            console.sync()
            def checked(command,timeout=20):
                code,text=console.run(command,timeout)
                assert code==0,(command,code,text[-1200:])
                return text
            identity=checked('uname -r; cat /proc/sys/kernel/random/boot_id; cat /proc/modules')
            assert inventory['boot_id'] in identity and '6.6.40' in identity
            existing=re.findall(r'(?m)^(u02_\w+)\s+\d+\s',identity)
            assert set(existing)==({'u02_uvcvideo'} if args.candidate_uvc else set())
            if args.candidate_uvc:
                for device in ('1-1.4:1.0','1-1.2:1.0'):
                    assert re.search(r'(?m)^\S*/bus/usb/drivers/u02_uvcvideo\r*$',checked('readlink /sys/bus/usb/devices/'+device+'/driver'))
                assert re.search(r'(?m)^2048\r*$',checked('cat /sys/module/u02_uvcvideo/parameters/payload_limit'))
            state['boot_id']=inventory['boot_id'];save()
            nonce=uuid.uuid4().hex;ready='__DUAL_READY_'+nonce;done='__DUAL_DONE_'+nonce
            console.send((f"mkdir {remote}; timeout 45 nc -l -p 18086 > {remote}/capture_dual < /dev/null & u02_dual_rx=$!; sleep 1; cat /proc/net/tcp; printf '\\n{ready}\\n'; wait $u02_dual_rx; printf '\\n{done}:%s\\n' \"$?\"\n").encode())
            listener,_=console.wait(rb'(?:^|\r?\n)'+ready.encode()+rb'\r?\n',10)
            assert re.search(rb'\b[0-9A-F]{8}:46A6\s+[0-9A-F]{8}:[0-9A-F]{4}\s+0A\b',listener)
            with connect_file_server(('192.168.50.2',18086)) as connection:
                connection.sendall(raw);connection.shutdown(socket.SHUT_WR)
            _,match=console.wait(rb'(?:^|\r?\n)'+done.encode()+rb':([0-9]+)\r?\n',30)
            assert int(match.group(1))==0
            text=checked('sha256sum '+remote+'/capture_dual')
            assert re.search(r'(?m)^'+sha(binary)+r'\s',text)
            checked('chmod 700 '+remote+'/capture_dual')
            arguments='' if args.yuyv else f'{frames} {width} {height}'
            if args.record_all:
                arguments+=' --record-all'
                memory=checked("awk '/^MemAvailable:/{print $2}' /proc/meminfo")
                assert any(int(value)>=768*1024 for value in re.findall(r'(?m)^(\d+)\r*$',memory))
                space=checked('df -k /tmp')
                lines=[line.split() for line in space.splitlines() if re.match(r'^(rootfs|tmpfs|/dev/)',line)]
                assert lines and int(lines[-1][3])>=272*1024,space
            deadline=15 if frames<=300 else 55
            if args.armed:
                arguments+=' --armed'
                checked(f'cd {remote}; (timeout 260 ./capture_dual {arguments} > capture.log 2>&1 < /dev/null & u02_sync_pid=$!; printf "%s\\n" "$u02_sync_pid" > capture_pid; wait "$u02_sync_pid"; printf "%s\\n" "$?" > capture_exit) < /dev/null > launcher.log 2>&1 & sleep 1')
                checked(f'cd {remote}; for i in $(seq 1 100); do test -f ready && break; test -f capture_exit && break; sleep 0.1; done; test -f ready && cat ready',15)
                state['armed']=True;save();print('SYNC_CAPTURE_ARMED',output,flush=True)
                state['armed_wall_time_ns']=time.time_ns();save()
                wait_until=time.monotonic()+180
                while not (output/'start_capture').is_file():
                    assert time.monotonic()<wait_until,'Timed out waiting for the announced sync capture start'
                    time.sleep(0.05)
                assert (output/'start_capture').read_text().strip()=='START'
                checked(f'cd {remote} && touch go')
                state['start_announced']=True;save()
                text=checked(f'cd {remote}; for i in $(seq 1 500); do test -f capture_exit && break; sleep 0.1; done; test -f capture_exit && cat capture_exit',55)
                codes=re.findall(r'(?m)^(\d+)\r*$',text);assert len(codes)==1
                code=int(codes[0])
            else:
                code,text=console.run(f'cd {remote} && timeout {deadline} ./capture_dual {arguments} > capture.log 2>&1',deadline+5)
            state['capture_exit_code']=code
            state['capture_log']=checked('cat '+remote+'/capture.log')
            state['kernel_tail']=checked('dmesg | tail -n 45')
            state['post_capture_identity']=checked('cat /proc/sys/kernel/random/boot_id; cat /proc/modules; cat /sys/class/net/eth0/carrier')
            assert inventory['boot_id'] in state['post_capture_identity']
            save()
            assert code==0,(code,state['capture_log'][-1600:])
            text=checked('cat '+remote+'/capture.log')
            marker='U02_DUAL_YUYV_CAPTURE_PASS frames_per_camera=30' if args.yuyv else f'U02_DUAL_CAPTURE_FUNCTIONAL_PASS frames_per_camera={frames}'
            assert marker in text
            state['capture_pass_marker']=text;save()
            allowed={'left_0.yuyv','right_0.yuyv','capture.log'} if args.yuyv else {'left.csv','right.csv','left_0.jpg',f'left_{frames-1}.jpg','right_0.jpg',f'right_{frames-1}.jpg','capture.log'}
            if args.record_all:
                allowed={'left.csv','right.csv','capture.log'}|{f'{side}_{i}.jpg' for side in ('left','right') for i in range(frames)}
                members='left.csv right.csv capture.log left_*.jpg right_*.jpg'
            else:members=' '.join(sorted(allowed))
            text=checked(f'cd {remote} && tar -czf capture_outputs.tar.gz {members} && sha256sum capture_outputs.tar.gz')
            match=re.search(r'(?m)^([0-9a-f]{64})\s+capture_outputs.tar.gz',text);assert match
            expected=match.group(1)
            checked(f'timeout 45 nc -l -p 18087 < {remote}/capture_outputs.tar.gz > /dev/null & sleep 1')
            received=bytearray()
            with connect_file_server(('192.168.50.2',18087)) as connection:
                while True:
                    chunk=connection.recv(1024*1024)
                    if not chunk:break
                    received.extend(chunk);assert len(received)<(280 if args.record_all else 8)*1024*1024
            assert hashlib.sha256(received).hexdigest()==expected
            (output/'capture_outputs.tar.gz').write_bytes(received)
            with tarfile.open(fileobj=io.BytesIO(received),mode='r:gz') as archive:
                assert {member.name for member in archive.getmembers()}==allowed
                assert sum(member.size for member in archive.getmembers())<(258 if args.record_all else 8)*1024*1024
                for member in archive.getmembers():
                    assert member.isfile() and member.size<4*1024*1024
                    (output/member.name).write_bytes(archive.extractfile(member).read())
            state['post_capture_identity']=checked('cat /proc/sys/kernel/random/boot_id; cat /proc/modules; cat /sys/class/net/eth0/carrier')
            assert inventory['boot_id'] in state['post_capture_identity']
        finally:console.port.close()
    for side in ('left','right'):
        if args.yuyv:
            assert (output/(side+'_0.yuyv')).stat().st_size==153600
            rows=re.findall(r'(?m)^FRAME '+side+r' index=(\d+) sequence=(\d+) timestamp_us=(\d+) flags=(\d+) receive_monotonic_ns=(\d+) bytes=(\d+)\r?$',(output/'capture.log').read_text())
            assert len(rows)==30 and all(int(row[0])==i and int(row[5])==153600 for i,row in enumerate(rows))
            assert all(int(b[1])>int(a[1]) and int(b[2])>int(a[2]) and int(b[4])>int(a[4]) for a,b in zip(rows,rows[1:]))
            continue
        with (output/(side+'.csv')).open(newline='') as file:rows=list(csv.DictReader(file))
        assert len(rows)==frames and all(int(row['capture_index'])==i for i,row in enumerate(rows))
        for first,second in zip(rows,rows[1:]):
            assert int(second['sequence'])>int(first['sequence'])
            assert int(second['source_timestamp_us'])>int(first['source_timestamp_us'])
        for index in (range(frames) if args.record_all else (0,frames-1)):
            jpeg=(output/(side+'_'+str(index)+'.jpg')).read_bytes()
            assert jpeg[:2]==b'\xff\xd8' and jpeg[-2:]==b'\xff\xd9' and len(jpeg)==int(rows[index]['jpeg_bytes'])
    assert all(sha(Path(path))==digest for path,digest in frozen.items())
    state.update(complete=True,simultaneous_streaming_verified=True,
        artifact_sha256={path.name:sha(path) for path in output.iterdir() if path.is_file() and path.name!='state.json'},
        scope=f'Concurrent {width}x{height} '+('YUYV diagnostic: 30 frames each, raw first-frame evidence.' if args.yuyv else f'MJPEG production-path capture: {frames} frames each and JPEG evidence.')+' Two real USB devices, raw source/receive times; no neural inference, PL use, calibration or exposure synchronization acceptance.')
except Exception as error:state['error']=str(error);raise
finally:save();print('STEREO_DUAL_CAPTURE_RECORD',output,flush=True)
