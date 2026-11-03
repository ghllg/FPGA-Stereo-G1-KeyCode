"""Temporarily enable VIDEO negotiation logs and restore the existing setting."""
import json
from pathlib import Path
import re
import sys
import time

U02=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(U02.parents[1]/'bringup/axu5evb_e_boot'))
from uart_console import Console

prior=U02/'reports/stereo_dual_capture_1791813892131525000/state.json'
proof=json.loads(prior.read_text());remote=proof['remote']
assert re.fullmatch(r'/tmp/u02_[0-9a-f]{32}',remote) and not proof['complete']
binary_digest=next(value for path,value in proof['input_sha256'].items() if Path(path).name=='test_dual_capture_small')
parameter='/sys/module/uvcvideo/parameters/trace'
output=U02/('reports/stereo_uvc_trace_'+str(time.time_ns()));output.mkdir()
state=dict(complete=False,driver_replaced=False,modules_changed=False,kernel_changed=False,
    reboot=False,startup_files_modified=False,trace_restored=False,exposure_sync_verified=False)
try:
    with (output/'uart.log').open('wb') as log:
        console=Console('COM9',log);original=None
        try:
            console.sync()
            def checked(command,timeout=20):
                code,text=console.run(command,timeout)
                assert code==0,(command,code,text[-1000:])
                return text
            text=checked('cat '+parameter)
            values=re.findall(r'(?m)^(\d+)\r*$',text);assert len(values)==1
            original=int(values[0]);assert 0<=original<=0xffffffff
            state['original_trace']=original
            assert re.search(r'(?m)^'+binary_digest+r'\s',checked('sha256sum '+remote+'/capture_dual'))
            assert proof['boot_id'] in checked('cat /proc/sys/kernel/random/boot_id')
            # UVC_DBG_VIDEO=1<<10 was checked in the matching local kernel header.
            checked("printf '%s\\n' "+str(original|1024)+' > '+parameter)
            code,text=console.run(f'cd {remote} && timeout 15 ./capture_dual 90 320 240 > trace_capture.log 2>&1',20)
            state['capture_exit_code']=code;state['capture_response']=text
            state['kernel_tail']=checked('dmesg | tail -n 100',30)
            (output/'kernel_tail.txt').write_text(state['kernel_tail'],encoding='utf-8')
            state['capture_log']=checked('cat '+remote+'/trace_capture.log')
            assert code==1 and 'Camera STREAMON failed' in state['capture_log']
        finally:
            if original is not None:
                code,text=console.run("printf '%s\\n' "+str(original)+' > '+parameter+' && cat '+parameter,15)
                assert code==0 and re.search(r'(?m)^'+str(original)+r'\r*$',text)
                state['trace_restored']=True
            console.port.close()
    state.update(complete=True,scope='Matching-source VIDEO-only UVC negotiation trace for existing failed small mode, original trace restored. No driver/kernel replacement or PL access; diagnostic only.')
except Exception as error:state['error']=str(error);raise
finally:
    (output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    print('STEREO_UVC_TRACE_RECORD',output,flush=True)
