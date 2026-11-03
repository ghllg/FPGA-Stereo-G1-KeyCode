"""Export the already verified HLS solution; never rerun synthesis or simulation."""
import json
import os
from pathlib import Path
import subprocess
import time
from synthesize_fp32_ooc import U02,sha,check_rtl

record=U02/'reports/fp32_wide_coalesced_cosim_1791866727589024900'
proof=check_rtl(record)
project=Path(proof['project']);source=project/'zu5ev/syn/verilog'
assert project==U02/'inference/stream_hls_22_fp32_wide_coalesced'
assert not (project/'zu5ev/impl/ip').exists(), 'Preserve prior package attempt'
frozen={str(p):sha(p) for p in source.iterdir() if p.is_file()}
frozen[str(record/'state.json')]=sha(record/'state.json')
script=U02/'inference/hls/export_stream.tcl'
frozen.update({str(script):sha(script),str(Path(__file__)):sha(Path(__file__))})
assert all(sha(Path(path))==digest for path,digest in frozen.items())
output=U02/('reports/fp32_wide_coalesced_ip_export_'+str(time.time_ns()));output.mkdir()
state=dict(complete=False,input_sha256=frozen,project=str(project),
    hls_synthesis_rerun=False,rtl_simulation_rerun=False,board_accessed=False,
    hardware_verified=False,scope='Package preparation only; OOC/DRC/top-route and board acceptance remain independent gates.')
def save():(output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
try:
    variables=os.environ.copy();variables.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),
        U02_STREAM_HLS_PROJECT=str(project))
    with (output/'export.log').open('wb') as log:
        process=subprocess.Popen(['powershell.exe','-NoProfile','-Command',
            "& 'D:/Develop/Xilink/Vivado/2018.3/bin/vivado_hls.bat' -f inference/hls/export_stream.tcl; exit $LASTEXITCODE"],
            cwd=U02,env=variables,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
        state['pid']=process.pid;save();code=process.wait()
    state.update(terminal=True,exit_code=code,log_sha256=sha(output/'export.log'));save()
    assert all(sha(Path(path))==digest for path,digest in frozen.items())
    assert code==0, 'Export failed; preserve original package and diagnose before retry'
    assert (project/'zu5ev/impl/ip/component.xml').is_file()
    state.update(complete=True,component_sha256=sha(project/'zu5ev/impl/ip/component.xml'),
                 package_prepared=True,package_rtl_and_float_dependencies_audited=False)
except Exception as error:state['error']=str(error);raise
finally:save();print('FP32_WIDE_COALESCED_IP_EXPORT_RECORD',output,flush=True)
