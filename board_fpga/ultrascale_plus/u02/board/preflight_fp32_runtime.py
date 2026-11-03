"""Read-only UART inventory before candidate PL deployment; no register access."""
import json
from pathlib import Path
import re
import sys
import time

U02=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(U02.parents[1]/'bringup/axu5evb_e_boot'))
from uart_console import Console

output=U02/('reports/fp32_runtime_preflight_'+str(time.time_ns()));output.mkdir()
state=dict(complete=False,board_accessed=True,registers_accessed=False,board_programmed=False,
    modules_loaded=False,reboot=False,startup_files_modified=False,commands={})
try:
    with (output/'uart.log').open('wb') as log:
        console=Console('COM9',log)
        try:
            console.sync()
            commands={'kernel':'uname -r','boot_id':'cat /proc/sys/kernel/random/boot_id',
                'modules':'cat /proc/modules','cmdline':'cat /proc/cmdline',
                'memory':'cat /proc/meminfo','video_nodes':'ls /sys/class/video4linux',
                'network':'cat /sys/class/net/eth0/carrier'}
            for name,command in commands.items():
                code,text=console.run(command,timeout=15)
                state['commands'][name]=dict(exit_code=code,text=text)
                assert code==0, 'Read-only command failed: '+name
            match=re.search(r'(?m)^([0-9a-f]{8}-[0-9a-f-]{27,})\s*$',state['commands']['boot_id']['text'])
            assert match
            state['boot_id']=match.group(1)
            state['active_u02_modules']=re.findall(r'(?m)^(u02_\w+)\s+\d+\s',state['commands']['modules']['text'])
            assert re.search(r'(?m)^6\.6\.40\s*$',state['commands']['kernel']['text'])
            state.update(complete=True,scope='Current Linux/kernel/boot/module/memory/video/network read-only inventory; excludes JTAG configuration identity, PL clocks, MMIO, DMA or camera streaming acceptance.')
        finally:console.port.close()
except Exception as error:state['error']=str(error);raise
finally:
    (output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    print('FP32_RUNTIME_PREFLIGHT_RECORD',output,flush=True)
