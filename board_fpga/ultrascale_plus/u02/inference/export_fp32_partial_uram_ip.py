"""Export the exact HLS23 solution after numeric RTL and independent OOC proof."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from synthesize_fp32_ooc import U02, sha, check_rtl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rtl_record', type=Path)
    parser.add_argument('ooc_record', type=Path)
    args = parser.parse_args()
    rtl_record, ooc_record = args.rtl_record.resolve(), args.ooc_record.resolve()
    assert rtl_record.parent == ooc_record.parent == U02 / 'reports'
    proof = check_rtl(rtl_record)
    project = Path(proof['project'])
    assert project == U02 / 'inference/stream_hls_23_fp32_wide_partial_uram'
    source = project / 'zu5ev/syn/verilog'
    assert {p.name: sha(p) for p in source.glob('*.v')} == proof['rtl_sha256']
    ooc = json.loads((ooc_record / 'state.json').read_text())
    assert ooc['complete'] and ooc['terminal'] and ooc['exit_code'] == 0
    assert Path(ooc['proof_record']) == Path(proof['synthesis_record'])
    assert sha(ooc_record / 'synthesis.log') == ooc['log_sha256']
    assert all(sha(Path(p)) == h for p, h in ooc['input_sha256'].items())
    assert all(sha(Path(ooc['project']) / n) == h for n, h in ooc['artifacts_sha256'].items())
    assert all(sha(source / n) == e['source_sha256'] for n, e in ooc['staged_files'].items())
    assert 'FP32_PARTIAL_URAM_DRC_ERROR_COUNT=0' in (ooc_record / 'synthesis.log').read_text()
    assert not (project / 'zu5ev/impl/ip').exists(), 'Preserve prior package attempt'
    script = U02 / 'inference/hls/export_stream.tcl'
    inputs = [p for p in source.iterdir() if p.is_file()]
    inputs += [rtl_record / 'state.json', ooc_record / 'state.json', script,
               Path(__file__), Path(__file__).with_name('synthesize_fp32_ooc.py')]
    frozen = {str(p): sha(p) for p in inputs}
    output = U02 / ('reports/fp32_wide_partial_uram_ip_export_' + str(time.time_ns()))
    output.mkdir()
    state = dict(complete=False, input_sha256=frozen, project=str(project),
                 rtl_record=str(rtl_record), ooc_record=str(ooc_record),
                 hls_synthesis_rerun=False, rtl_simulation_rerun=False,
                 board_accessed=False, routed=False, deployment_accepted=False)
    def save():
        (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        env = os.environ.copy()
        env.update(TEMP=str(U02 / 'toolchain/tmp'), TMP=str(U02 / 'toolchain/tmp'),
                   U02_STREAM_HLS_PROJECT=str(project))
        with (output / 'export.log').open('wb') as log:
            process = subprocess.Popen(['powershell.exe', '-NoProfile', '-Command',
                "& 'D:/Develop/Xilink/Vivado/2018.3/bin/vivado_hls.bat' -f inference/hls/export_stream.tcl; exit $LASTEXITCODE"],
                cwd=U02, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid'] = process.pid
            save()
            code = process.wait()
        state.update(terminal=True, exit_code=code, log_sha256=sha(output / 'export.log'))
        save()
        assert all(sha(Path(p)) == h for p, h in frozen.items())
        text = (output / 'export.log').read_text(errors='replace')
        assert code == 0 and not re.search(r'(?m)^\s*(?:ERROR|CRITICAL WARNING):', text), 'Preserve export failure for diagnosis; do not restart'
        component = project / 'zu5ev/impl/ip/component.xml'
        assert component.is_file()
        state.update(complete=True, component_sha256=sha(component),
                     scope='Fresh HLS23 package export only; isolated package dependency audit, full top route and board A/B remain')
    except Exception as error:
        state['error'] = repr(error)
        raise
    finally:
        save()
        print('FP32_PARTIAL_URAM_IP_EXPORT_RECORD', output, flush=True)


if __name__ == '__main__':
    main()
