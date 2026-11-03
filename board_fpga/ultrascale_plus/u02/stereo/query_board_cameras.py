"""Verify independent capture nodes using the existing hash-bound query tool."""
import json
from pathlib import Path
import re
import sys
import time

U02=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(U02.parents[1]/'bringup/axu5evb_e_boot'))
from uart_console import Console

tool='/tmp/u02_fb90cfde1e554cb79f56b8d478835279/camera_probe'
digest='1dc7996ac5a833a34de883c1e228ad21904bc76961a770b1067188cd6c443e04'
output=U02/('reports/stereo_board_camera_query_'+str(time.time_ns()));output.mkdir()
state=dict(complete=False,query_only=True,streaming_started=False,board_programmed=False,
    exposure_sync_verified=False,nodes=[],camera_probe_sha256=digest)
try:
    with (output/'uart.log').open('wb') as log:
        console=Console('COM9',log)
        try:
            console.sync()
            code,text=console.run('sha256sum '+tool,15)
            assert code==0 and re.search(r'(?m)^'+digest+r'\s',text)
            code,text=console.run('cat /proc/sys/kernel/random/boot_id',15)
            assert code==0
            match=re.search(r'(?m)^([0-9a-f]{8}-[0-9a-f-]{27,})\s*$',text);assert match
            state['boot_id']=match.group(1)
            for index in range(4):
                device='/dev/video'+str(index)
                code,text=console.run(tool+' '+device,15)
                assert code==0
                match=re.search(r'(?m)^DEVICE (\S+) driver=(\S+) card=(.*?) bus=(\S+) capabilities=0x([0-9a-f]+)',text)
                assert match and match.group(1)==device
                caps=int(match.group(5),16)
                code,sysfs=console.run('readlink -f /sys/class/video4linux/video'+str(index)+'/device',15)
                assert code==0
                state['nodes'].append(dict(device=device,driver=match.group(2),card=match.group(3),
                    bus=match.group(4),capabilities=caps,video_capture=bool(caps&1),
                    metadata=bool(caps&0x800000),streaming=bool(caps&0x4000000),
                    query_output=text,sysfs_identity=sysfs))
            capture=[node for node in state['nodes'] if node['video_capture'] and node['streaming'] and not node['metadata']]
            assert len({node['bus'] for node in capture})==len(capture), 'Capture nodes share a physical bus'
            state.update(complete=True,independent_capture_cameras=len(capture),
                capture_devices=[node['device'] for node in capture],
                simultaneous_streaming_verified=False,
                scope='Actual QUERYCAP and distinct USB bus capture identities; no concurrent streaming, exposure sync or calibration identity acceptance.')
        finally:console.port.close()
except Exception as error:state['error']=str(error);raise
finally:
    (output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    print('STEREO_BOARD_CAMERA_QUERY_RECORD',output,flush=True)
