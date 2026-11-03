"""Run independent Vivado OOC synthesis only after completed numeric RTL proof."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

U02 = Path(__file__).resolve().parents[1]
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def check_rtl(record):
    state = json.loads((record / 'state.json').read_text())
    assert state.get('terminal') and state['complete'] and state['exit_code'] == 0, 'Numeric RTL proof incomplete'
    assert sha(record / 'cosim.log') == state['log_sha256']
    assert all(sha(Path(p)) == digest for p, digest in state['input_sha256'].items())
    project = Path(state['project'])
    report = project / 'zu5ev/sim/report/u02_stream_cosim.rpt'
    vectors = project / 'zu5ev/sim/tv/cdatafile/ref.tcl'
    assert sha(report) == state['cosim_report_sha256']
    assert sha(vectors) == state['vector_metadata_sha256']
    assert re.search(r'(?mi)^\s*\|\s*Verilog\s*\|\s*Pass\s*\|', report.read_text())
    assert re.search(r'(?m)^set trans_num\s+252\s*$', vectors.read_text())
    return state

def main():
    proof_record = U02 / 'reports/fp32_output_cosim_1791800499799941500'
    check_rtl(proof_record)
    inputs = U02 / 'inference/fp32_output_ooc_inputs'
    prep = json.loads((inputs / 'preparation.json').read_text())
    manifest = U02 / 'reports/stream_fp32_output_synthesis_manifest.json'
    assert sha(manifest) == prep['manifest_sha256']
    for filename, record in prep['files'].items():
        assert sha(inputs / filename) == record['staged_sha256']
    assert shutil.disk_usage(U02).free > 10 * 1024**3
    project = U02 / 'inference/fp32_output_ooc_project'
    assert not project.exists(), 'Preserve existing synthesis attempt'
    script = U02 / 'inference/fp32_output_ooc.tcl'
    frozen = {str(p): sha(p) for p in [script, Path(__file__), inputs / 'preparation.json', manifest]}
    frozen.update({str(inputs / name): entry['staged_sha256'] for name, entry in prep['files'].items()})
    output = U02 / ('reports/fp32_output_ooc_' + str(time.time_ns()))
    output.mkdir()
    state = dict(complete=False, project=str(project), input_sha256=frozen,
                 rtl_record=str(proof_record), rtl_state_sha256=sha(proof_record / 'state.json'),
                 board_accessed=False, hardware_verified=False, routed=False)
    def save():
        (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    try:
        environment = os.environ.copy()
        environment.update(TEMP=str(U02 / 'toolchain/tmp'), TMP=str(U02 / 'toolchain/tmp'),
                           U02_FP32_OOC_INPUTS=str(inputs), U02_FP32_OOC_PROJECT=str(project))
        with (output / 'synthesis.log').open('wb') as log:
            process = subprocess.Popen(['powershell.exe', '-NoProfile', '-Command',
                "& 'D:/Develop/Xilink/Vivado/2018.3/bin/vivado.bat' -mode batch -source inference/fp32_output_ooc.tcl -log " +
                "'" + str(output / 'vivado.log') + "' -journal '" + str(output / 'vivado.jou') + "'; exit $LASTEXITCODE"],
                cwd=U02, env=environment, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid'] = process.pid
            save()
            code = process.wait()
        state.update(terminal=True, exit_code=code, log_sha256=sha(output / 'synthesis.log'))
        save()
        text = (output / 'synthesis.log').read_text(errors='replace')
        assert code == 0 and not re.search(r'(?m)^\s*ERROR:', text)
        assert 'FP32_OUTPUT_OOC_SYNTHESIS_COMPLETE' in text
        assert all(sha(Path(p)) == digest for p, digest in frozen.items())
        artifacts = {name: sha(project / name) for name in ('utilization.rpt', 'timing.rpt', 'drc.rpt', 'fp32_output_ooc.dcp')}
        state.update(complete=True, artifacts_sha256=artifacts,
                     scope='Actual Vivado OOC synthesis only; top-level route, safe deployment and full-board inference still required.')
    except Exception as error:
        state['error'] = str(error)
        raise
    finally:
        save()
        print('FP32_OUTPUT_OOC_RECORD', output, flush=True)

if __name__ == '__main__':
    main()
