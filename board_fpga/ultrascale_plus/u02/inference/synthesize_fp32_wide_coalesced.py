"""Synthesize the complete C-verified wide-memory model without touching the board."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from synthesize_fp32_ooc import U02,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('record',type=Path);args=parser.parse_args()
    proof_path=args.record.resolve();assert proof_path.parent==U02/'reports'
    proof=json.loads((proof_path/'state.json').read_text())
    assert proof['complete'] and proof['terminal'] and proof['exit_code']==0
    assert sha(proof_path/'csim.log')==proof['log_sha256']
    assert all(sha(Path(path))==digest for path,digest in proof['input_sha256'].items())
    project=Path(proof['project']);assert project==U02/'inference/stream_hls_22_fp32_wide_coalesced'
    assert not (project/'zu5ev/syn/report/u02_stream_csynth.xml').exists()
    assert shutil.disk_usage(U02).free>10*1024**3
    script=U02/'inference/hls/synth_fp32_output.tcl'
    frozen=dict(proof['input_sha256']);frozen.update({str(script):sha(script),str(Path(__file__)):sha(Path(__file__))})
    out=U02/('reports/fp32_wide_coalesced_synthesis_'+str(time.time_ns()));out.mkdir()
    state=dict(complete=False,project=str(project),input_sha256=frozen,c_record=str(proof_path),
        board_accessed=False,hardware_verified=False,routed=False,formal_baseline='bdc3b7a')
    def save():(out/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save();started=time.monotonic()
    try:
        env=os.environ.copy();env.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),U02_FP32_HLS_PROJECT=str(project))
        with (out/'synthesis.log').open('wb') as log:
            process=subprocess.Popen(['powershell.exe','-NoProfile','-Command',
                "& 'D:/Develop/Xilink/Vivado/2018.3/bin/vivado_hls.bat' -f inference/hls/synth_fp32_output.tcl; exit $LASTEXITCODE"],
                cwd=U02,env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid']=process.pid;save();code=process.wait()
        state.update(terminal=True,exit_code=code,log_sha256=sha(out/'synthesis.log'),elapsed_wall_s=time.monotonic()-started);save()
        assert code==0 and all(sha(Path(path))==digest for path,digest in frozen.items())
        assert 'ERROR:' not in (out/'synthesis.log').read_text(errors='replace')
        assert (project/'zu5ev/syn/report/u02_stream_csynth.xml').is_file()
        state.update(complete=True,scope='HLS synthesis only: resource/sharing/interface audit, RTL, route and complete-board ARM comparison still required')
    except Exception as error:state['error']=repr(error);raise
    finally:save();print('FP32_WIDE_COALESCED_SYNTHESIS_RECORD',out,flush=True)


if __name__=='__main__':main()
