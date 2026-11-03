"""Synthesize the already verified complete-model C candidate, without board access."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

U02=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    c=U02/'reports/fp32_output_csim_1791798185723520500';proof=json.loads((c/'state.json').read_text())
    assert proof['complete'] and proof['terminal'] and proof['exit_code']==0 and sha(c/'csim.log')==proof['log_sha256']
    assert all(sha(Path(p))==value for p,value in proof['input_sha256'].items())
    assert shutil.disk_usage(U02).free>10*1024**3
    project=Path(proof['project']);assert not (project/'zu5ev/syn/report/u02_stream_csynth.xml').exists()
    frozen=dict(proof['input_sha256']);frozen[str(Path(__file__))]=sha(Path(__file__))
    script=U02/'inference/hls/synth_fp32_output.tcl';frozen[str(script)]=sha(script)
    out=U02/('reports/fp32_output_synthesis_'+str(time.time_ns()));out.mkdir()
    state=dict(complete=False,input_sha256=frozen,project=str(project),board_accessed=False,hardware_verified=False,
               c_verification=str(c),c_state_sha256=sha(c/'state.json'))
    def save():(out/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    try:
        env=os.environ.copy();env.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),U02_FP32_HLS_PROJECT=str(project))
        with (out/'synthesis.log').open('wb') as log:
            proc=subprocess.Popen(['powershell.exe','-NoProfile','-Command',
              "& 'D:/Develop/Xilink/Vivado/2018.3/bin/vivado_hls.bat' -f inference/hls/synth_fp32_output.tcl; exit $LASTEXITCODE"],
              cwd=U02,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid']=proc.pid;save();code=proc.wait()
        state.update(terminal=True,exit_code=code,log_sha256=sha(out/'synthesis.log'));save()
        assert code==0 and all(sha(Path(p))==value for p,value in frozen.items())
        assert 'ERROR:' not in (out/'synthesis.log').read_text(errors='replace'), 'HLS wrapper returned zero despite a Tcl failure'
        assert (project/'zu5ev/syn/report/u02_stream_csynth.xml').is_file()
        state.update(complete=True,scope='HLS synthesis only; requires independent resource/sharing audit, RTL, routed implementation and complete-board acceptance.')
    except Exception as error:state['error']=str(error);raise
    finally:save();print('FP32_OUTPUT_SYNTHESIS_RECORD',out,flush=True)

if __name__=='__main__':main()
