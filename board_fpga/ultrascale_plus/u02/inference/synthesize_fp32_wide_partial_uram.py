"""Synthesize repair round 2 in its existing C-verified project."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from synthesize_fp32_ooc import U02, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('record', type=Path)
    record = parser.parse_args().record.resolve()
    assert record.parent == U02 / 'reports'
    proof = json.loads((record / 'state.json').read_text())
    assert proof['complete'] and proof['terminal'] and proof['exit_code'] == 0
    assert sha(record / 'csim.log') == proof['log_sha256']
    assert all(sha(Path(path)) == digest for path, digest in proof['input_sha256'].items())
    project = Path(proof['project'])
    assert project == U02 / 'inference/stream_hls_23_fp32_wide_partial_uram'
    assert not (project / 'zu5ev/syn/report/u02_stream_csynth.xml').exists()
    assert shutil.disk_usage(U02).free > 10 * 1024**3
    script = U02 / 'inference/hls/synth_fp32_output.tcl'
    frozen = dict(proof['input_sha256'])
    frozen.update({str(script): sha(script), str(Path(__file__)): sha(Path(__file__))})
    output = U02 / ('reports/fp32_wide_partial_uram_synthesis_' + str(time.time_ns()))
    output.mkdir()
    state = dict(complete=False, project=str(project), input_sha256=frozen, c_record=str(record),
                 repair_round=2, board_accessed=False, hardware_verified=False, routed=False)
    def save():
        (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    print('FP32_PARTIAL_URAM_SYNTHESIS_RECORD', output, flush=True)
    started = time.monotonic()
    try:
        variables = os.environ.copy()
        variables.update(TEMP=str(U02 / 'toolchain/tmp'), TMP=str(U02 / 'toolchain/tmp'),
                         U02_FP32_HLS_PROJECT=str(project))
        with (output / 'synthesis.log').open('wb') as log:
            process = subprocess.Popen(['powershell.exe', '-NoProfile', '-Command',
                "& 'D:/Develop/Xilink/Vivado/2018.3/bin/vivado_hls.bat' -f inference/hls/synth_fp32_output.tcl; exit $LASTEXITCODE"],
                cwd=U02, env=variables, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid'] = process.pid
            save()
            code = process.wait()
        state.update(terminal=True, exit_code=code, log_sha256=sha(output / 'synthesis.log'),
                     elapsed_wall_s=time.monotonic() - started)
        save()
        assert code == 0 and all(sha(Path(path)) == digest for path, digest in frozen.items())
        assert 'ERROR:' not in (output / 'synthesis.log').read_text(errors='replace')
        assert (project / 'zu5ev/syn/report/u02_stream_csynth.xml').is_file()
        state.update(complete=True, scope='HLS synthesis only; numerical RTL, physical resources, route and full-network board A/B remain required')
    except Exception as error:
        state['error'] = repr(error)
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
