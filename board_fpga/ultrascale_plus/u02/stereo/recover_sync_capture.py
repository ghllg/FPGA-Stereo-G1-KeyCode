"""Recover saved real frames from a completed capture after archive timeout.

Never rerun capture or rewrite the original failure record. Restore only the
known temporary camera driver; preserve original kernel/boot/PL configuration.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import socket
import sys
import tarfile
import time
import cv2
import csv

U02=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(U02.parents[1]/'bringup/axu5evb_e_boot'))
sys.path.insert(0,str(U02/'inference'))
from uart_console import Console
from validate_image_pipeline_resident import connect_file_server

prior=U02/'reports/stereo_dual_capture_1791817491372981800'
proof=json.loads((prior/'state.json').read_text())
assert proof['capture_exit_code']==0 and proof['record_all_frames'] and not proof['complete']
remote=proof['remote'];assert re.fullmatch(r'/tmp/u02_[0-9a-f]{32}',remote)
frames=proof['frames_requested_per_camera'];assert frames==450
output=U02/('reports/stereo_sync_capture_recovered_'+str(time.time_ns()));output.mkdir()
state=dict(complete=False,prior_record=str(prior),capture_rerun=False,exposure_sync_verified=False,
    frames_per_camera=frames,devices_restored=False,module_removed=False,
    startup_files_modified=False,kernel_changed=False,reboot=False,board_programmed=False)
digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
state.update(prior_state_sha256=digest(prior/'state.json'),recovery_source_sha256=digest(Path(__file__)))
console=None
def save():(output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
with (output/'uart.log').open('wb') as log:
    try:
        console=Console('COM9',log)
        console.sync()
        def checked(command,timeout=30):
            code,text=console.run(command,timeout);assert code==0,(command,code,text[-2000:]);return text
        assert proof['boot_id'] in checked('cat /proc/sys/kernel/random/boot_id')
        text=checked('cat '+remote+'/capture_exit '+remote+'/capture.log')
        assert re.search(r'(?m)^0\r*$',text) and 'U02_DUAL_CAPTURE_FUNCTIONAL_PASS frames_per_camera=450' in text
        state['capture_completed_proof']=text
        text=checked('sha256sum '+remote+'/capture_outputs.tar.gz',50)
        match=re.search(r'(?m)^([0-9a-f]{64})\s',text);assert match
        expected=match.group(1)
        checked(f'timeout 60 nc -l -p 18089 < {remote}/capture_outputs.tar.gz > /dev/null & sleep 1')
        received=bytearray()
        with connect_file_server(('192.168.50.2',18089)) as connection:
            connection.settimeout(45)
            while True:
                block=connection.recv(1024*1024)
                if not block:break
                received.extend(block);assert len(received)<140*1024*1024
        assert hashlib.sha256(received).hexdigest()==expected
        (output/'capture_outputs.tar.gz').write_bytes(received)
        state['archive_sha256']=expected;state['archive_bytes']=len(received);save()
        allowed={'left.csv','right.csv','capture.log'}|{f'{side}_{i}.jpg' for side in ('left','right') for i in range(frames)}
        with tarfile.open(fileobj=io.BytesIO(received),mode='r:gz') as archive:
            members=archive.getmembers();assert {m.name for m in members}==allowed
            assert len(members)==len(allowed) and sum(m.size for m in members)<130*1024*1024
            for member in members:
                assert member.isfile() and member.size<4*1024*1024
                (output/member.name).write_bytes(archive.extractfile(member).read())
        # Full JPEG decode for every saved real frame, not only marker checks.
        for side in ('left','right'):
            rows=list(csv.DictReader((output/(side+'.csv')).open()));assert len(rows)==frames
            for index,row in enumerate(rows):
                image=output/(side+'_'+str(index)+'.jpg')
                assert int(row['capture_index'])==index and image.stat().st_size==int(row['jpeg_bytes'])
                decoded=cv2.imread(str(image));assert decoded is not None and decoded.shape==(720,1280,3)
        state.update(complete=True,all_900_jpegs_fully_decoded=True,
            scope='Recovery of existing saved720p real capture; no replay/rerun, no NN or exposure-sync claim.')
    except Exception as error:
        state['error']=str(error);raise
    finally:
        try:
            if console is not None:
                for device in ('1-1.4:1.0','1-1.2:1.0'):
                    code,text=console.run('readlink /sys/bus/usb/devices/'+device+'/driver',20)
                    assert code==0
                    if re.search(r'(?m)^\S*/bus/usb/drivers/u02_uvcvideo\r*$',text):
                        checked("printf '%s' "+device+' > /sys/bus/usb/drivers/u02_uvcvideo/unbind')
                        checked("printf '%s' "+device+' > /sys/bus/usb/drivers/uvcvideo/bind')
                    else:assert re.search(r'(?m)^\S*/bus/usb/drivers/uvcvideo\r*$',text)
                    assert re.search(r'(?m)^\S*/bus/usb/drivers/uvcvideo\r*$',checked('readlink /sys/bus/usb/devices/'+device+'/driver'))
                state['devices_restored']=True
                if re.search(r'(?m)^u02_uvcvideo\s',checked('cat /proc/modules')):checked('rmmod u02_uvcvideo')
                assert not re.search(r'(?m)^u02_uvcvideo\s',checked('cat /proc/modules'))
                state['module_removed']=True
                state['post_restore']=checked('cat /proc/sys/kernel/random/boot_id; cat /proc/modules; cat /sys/class/net/eth0/carrier')
        except Exception as error:
            state['restore_error']=str(error);state['complete']=False;raise
        finally:
            if console is not None:console.port.close()
            save();print('STEREO_SYNC_CAPTURE_RECOVERED',output,flush=True)
