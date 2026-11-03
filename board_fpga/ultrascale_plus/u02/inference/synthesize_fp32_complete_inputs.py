"""Repair omitted HLS RAM initialization dependencies in a separate OOC build."""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from synthesize_fp32_ooc import U02, check_rtl, sha


def main():
    check_rtl(U02/'reports/fp32_output_cosim_1791800499799941500')
    previous=U02/'inference/fp32_output_ooc_inputs'
    preparation=json.loads((previous/'preparation.json').read_text())
    manifest=U02/'reports/stream_fp32_output_synthesis_manifest.json'
    assert sha(manifest)==preparation['manifest_sha256']
    source=Path(json.loads(manifest.read_text())['project'])/'zu5ev/syn/verilog'
    inputs=U02/'inference/fp32_complete_ooc_inputs'
    project=U02/'inference/fp32_complete_ooc_project'
    assert not inputs.exists() and not project.exists(), 'Preserve previous attempts'
    assert shutil.disk_usage(U02).free>10*1024**3
    inputs.mkdir()
    for name,record in preparation['files'].items():
        assert sha(previous/name)==record['staged_sha256']
        shutil.copyfile(previous/name,inputs/name)
    referenced=set()
    for path in inputs.glob('*.v'):
        referenced.update(re.findall(r'\$readmemh\("\./([^"/]+\.dat)"',path.read_text()))
    assert len(referenced)==3
    for name in sorted(referenced):
        assert (source/name).is_file()
        shutil.copyfile(source/name,inputs/name)
        assert sha(source/name)==sha(inputs/name)
    script=U02/'inference/fp32_ooc_complete_inputs.tcl'
    frozen={str(path):sha(path) for path in list(inputs.iterdir())+[script,Path(__file__),manifest]}
    output=U02/('reports/fp32_complete_ooc_'+str(time.time_ns()));output.mkdir()
    state=dict(complete=False,project=str(project),input_sha256=frozen,
        ram_initialization_dependencies=sorted(referenced),
        prior_attempt_preserved=True,reason='Prior OOC omitted three readmemh .dat dependencies and produced Synth 8-4445 critical warnings.',
        board_accessed=False,routed=False,hardware_verified=False)
    def save():
        (output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    try:
        variables=os.environ.copy();variables.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),
            U02_FP32_COMPLETE_INPUTS=str(inputs),U02_FP32_COMPLETE_PROJECT=str(project))
        with (output/'synthesis.log').open('wb') as log:
            command=[ 'powershell.exe','-NoProfile','-Command',
                "& 'D:/Develop/Xilink/Vivado/2018.3/bin/vivado.bat' -mode batch -source inference/fp32_ooc_complete_inputs.tcl -log '"+
                str(output/'vivado.log')+"' -journal '"+str(output/'vivado.jou')+"'; exit $LASTEXITCODE"]
            process=subprocess.Popen(command,cwd=U02,env=variables,stdout=log,stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid']=process.pid;save();code=process.wait()
        state.update(terminal=True,exit_code=code,log_sha256=sha(output/'synthesis.log'));save()
        text=(output/'synthesis.log').read_text(errors='replace')
        assert code==0 and 'FP32_COMPLETE_INPUTS_OOC_COMPLETE' in text
        assert not re.search(r'(?m)^\s*(?:ERROR:|CRITICAL WARNING:)',text)
        assert all(sha(Path(path))==digest for path,digest in frozen.items())
        state.update(complete=True,artifacts_sha256={name:sha(project/name) for name in
            ('utilization.rpt','timing.rpt','drc.rpt','fp32_output_ooc.dcp')},
            scope='OOC synthesis with all HLS initialization data dependencies; no top route or board acceptance.')
    except Exception as error:
        state['error']=str(error);raise
    finally:
        save();print('FP32_COMPLETE_INPUTS_OOC_RECORD',output,flush=True)


if __name__=='__main__':main()
