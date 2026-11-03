"""Gate an independent full PS/PL implementation on final post-opt proof."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from synthesize_fp32_ooc import U02,sha,check_rtl


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('post_opt_record',type=Path)
    parser.add_argument('--wait-for-existing-audit',action='store_true')
    args=parser.parse_args();audit=args.post_opt_record.resolve()
    assert audit.parent==U02/'reports'
    output=U02/('reports/fp32_top_implementation_'+str(time.time_ns()));output.mkdir()
    state=dict(complete=False,waiting_for_existing_audit=True,post_opt_record=str(audit),
        board_accessed=False,board_programmed=False,startup_files_modified=False,routed=False)
    def save():(output/'state.json').write_text(json.dumps(state,indent=2)+'\n')
    save();print('FP32_TOP_IMPLEMENTATION_RECORD',output,flush=True)
    try:
        deadline=time.monotonic()+2700
        while True:
            try:proof=json.loads((audit/'state.json').read_text())
            except json.JSONDecodeError:
                if time.monotonic()>=deadline:raise
                time.sleep(1);continue
            if proof.get('terminal'):break
            assert args.wait_for_existing_audit, 'Audit not terminal; do not start top-level build'
            if time.monotonic()>=deadline:raise RuntimeError('Audit wait expired; no build restarted')
            time.sleep(15)
        assert proof['complete'] and proof['exit_code']==0, 'Post-opt did not pass actual DRC and clock gates'
        assert sha(audit/'audit.log')==proof['log_sha256']
        assert all(sha(Path(path))==digest for path,digest in proof['input_sha256'].items())
        assert all(sha(audit/path)==digest for path,digest in proof['artifacts_sha256'].items())
        drc=(audit/'drc.rpt').read_text(errors='replace')
        assert not re.search(r'(?m)^\|\s*\S+\s*\|\s*Error\s*\|',drc)
        check_rtl(U02/'reports/fp32_output_cosim_1791800499799941500')
        package_record=U02/'reports/fp32_candidate_package_1791810889840246600'
        package_proof=json.loads((package_record/'state.json').read_text())
        assert package_proof['complete'] and package_proof['subcore_target_family_verified']
        package=Path(package_proof['package'])
        assert sha(package/'component.xml')==package_proof['component_sha256']
        assert all(sha(Path(path))==digest for path,digest in package_proof['input_sha256'].items())
        assert all(sha(package/path)==digest for path,digest in package_proof['subcore_xci_sha256'].items())
        assert shutil.disk_usage(U02).free>20*1024**3
        project=U02/'inference/fp32_top_project';assert not project.exists(), 'Preserve existing top-level attempt'
        sources=[Path(__file__),U02/'inference/create_fp32_top_project.tcl',U02/'inference/create_stream_project.tcl',
            U02.parents[1]/'bringup/axu5evb_e_boot/vendor/ps_config.tcl',audit/'state.json',package_record/'state.json']
        sources+=list((U02/'board').glob('*.tcl'))+list((U02/'board').glob('*.v'))+list((U02/'board').glob('*.xdc'))+list((U02/'board').glob('*.tsv'))
        sources+=[path for folder in ('hdl','subcore','constraints','bd') for path in (package/folder).rglob('*') if path.is_file()]
        sources+=[package/'component.xml']
        frozen={str(path):sha(path) for path in sources}
        state.update(waiting_for_existing_audit=False,project=str(project),input_sha256=frozen);save()
        variables=os.environ.copy();variables.update(TEMP=str(U02/'toolchain/tmp'),TMP=str(U02/'toolchain/tmp'),
            U02_FP32_IP_PACKAGE=str(package),U02_STREAM_PROJECT_DIR=str(project),U02_STREAM_REPORT_DIR=str(output))
        command=['powershell.exe','-NoProfile','-Command',
            "& 'D:/Develop/Xilink/Vivado/2018.3/bin/vivado.bat' -mode batch -source inference/create_fp32_top_project.tcl -log '"+
            str(output/'vivado.log')+"' -journal '"+str(output/'vivado.jou')+"'; exit $LASTEXITCODE"]
        with (output/'implementation.log').open('wb') as log:
            process=subprocess.Popen(command,cwd=U02,env=variables,stdout=log,stderr=subprocess.STDOUT,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid']=process.pid;save();code=process.wait()
        state.update(exit_code=code,log_sha256=sha(output/'implementation.log'));save()
        text=(output/'implementation.log').read_text(errors='replace')
        assert code==0 and 'U02_STREAM_FULL_NETWORK_BACKEND_BIT_READY' in text
        assert not re.search(r'(?m)^\s*ERROR:',text)
        assert all(sha(Path(path))==digest for path,digest in frozen.items())
        bit=project/'u02_stream.runs/impl_1/compute_wrapper.bit'
        assert bit.is_file() and bit.stat().st_size>7_000_000
        state.update(complete=True,routed=True,bitstream=str(bit),bitstream_sha256=sha(bit),
            reports_sha256={name:sha(output/name) for name in ('timing.rpt','utilization.rpt','drc.rpt')},
            scope='Top implementation finished; independent routed timing/resource/reset/address/IP audit and board numerical validation still required before acceptance.')
    except Exception as error:state['error']=str(error);raise
    finally:state['terminal']=True;save()


if __name__=='__main__':main()
