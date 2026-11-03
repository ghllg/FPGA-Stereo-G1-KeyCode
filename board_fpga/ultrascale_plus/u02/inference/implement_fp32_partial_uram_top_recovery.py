"""Continue the preserved full top after PS simulation product generation failed."""
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from synthesize_fp32_ooc import U02, sha, check_rtl


def main():
    rtl_record = U02 / 'reports/fp32_wide_partial_uram_cosim_1791880536391743100'
    ooc_record = U02 / 'reports/fp32_wide_partial_uram_ooc_1791881122182526600'
    package_record = U02 / 'reports/fp32_wide_partial_uram_package_1791886718639335900'
    rtl = check_rtl(rtl_record)
    hls = Path(rtl['project'])
    assert hls == U02 / 'inference/stream_hls_23_fp32_wide_partial_uram'
    assert {p.name: sha(p) for p in (hls / 'zu5ev/syn/verilog').glob('*.v')} == rtl['rtl_sha256']
    ooc = json.loads((ooc_record / 'state.json').read_text())
    assert ooc['complete'] and ooc['terminal'] and ooc['exit_code'] == 0
    assert Path(ooc['proof_record']) == Path(rtl['synthesis_record'])
    assert sha(ooc_record / 'synthesis.log') == ooc['log_sha256']
    assert all(sha(Path(p)) == h for p, h in ooc['input_sha256'].items())
    ooc_project = Path(ooc['project'])
    assert all(sha(ooc_project / n) == h for n, h in ooc['artifacts_sha256'].items())
    assert 'FP32_PARTIAL_URAM_DRC_ERROR_COUNT=0' in (ooc_record / 'synthesis.log').read_text()
    assert not re.search(r'\|\s*(?:Error|Critical Warning)\s*\|', (ooc_project / 'drc.rpt').read_text())
    package_proof = json.loads((package_record / 'state.json').read_text())
    assert package_proof['complete'] and package_proof['terminal'] and package_proof['exit_code'] == 0
    assert package_proof['subcore_target_family_verified']
    package = Path(package_proof['package'])
    assert package == U02 / 'inference/fp32_wide_partial_uram_ip_package_zu5'
    assert sha(package / 'component.xml') == package_proof['component_sha256']
    assert sha(package_record / 'package.log') == package_proof['log_sha256']
    assert all(sha(Path(p)) == h for p, h in package_proof['input_sha256'].items())
    assert all(sha(package / n) == h for n, h in package_proof['subcore_xci_sha256'].items())
    for name, entry in ooc['staged_files'].items():
        folder = 'hdl/verilog' if name.endswith(('.v', '.dat')) else 'subcore'
        assert sha(package / folder / name) == entry['staged_sha256'], name
    project = U02 / 'inference/fp32_partial_uram_top_project'
    failed_record = U02 / 'reports/fp32_partial_uram_top_implementation_1791887300491943300'
    failed = json.loads((failed_record / 'state.json').read_text())
    assert failed['terminal'] and not failed['complete'] and len(failed['phases']) == 1
    assert failed['phases'][0]['exit_code'] == 1 and Path(failed['project']) == project
    assert sha(failed_record / 'synthesis.log') == failed['phases'][0]['log_sha256']
    assert all(sha(Path(p)) == h for p, h in failed['input_sha256'].items())
    assert "Failed to generate 'Verilog Simulation Wrapper'" in (failed_record / 'synthesis.log').read_text(errors='replace')
    assert project.exists() and not (project / 'u02_stream.runs/synth_1/runme.log').exists()
    assert not list(project.rglob('*.bit')), 'No implemented build may be resumed here'
    assert shutil.disk_usage(U02).free > 20 * 1024**3
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ('total_phys', 'avail_phys', 'total_commit',
                'avail_commit', 'total_virtual', 'avail_virtual', 'extended')]
    memory = MemoryStatus()
    memory.length = ctypes.sizeof(memory)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    # Each Vivado parent exits before the next phase. This removes the prior
    # multi-GiB synthesis/report heap from the implementation child's budget.
    # Keep the independently exercised 8 GiB commit floor for every phase.
    assert memory.avail_commit > 8 * 1024**3, ('Insufficient available commit', memory.avail_commit)
    script = U02 / 'inference/recover_fp32_partial_uram_top.tcl'
    sources = [Path(__file__), script, U02 / 'inference/create_stream_project.tcl',
               Path(__file__).with_name('synthesize_fp32_ooc.py'),
               U02.parents[1] / 'bringup/axu5evb_e_boot/vendor/ps_config.tcl',
               rtl_record / 'state.json', ooc_record / 'state.json', package_record / 'state.json']
    sources += [p for p in (U02 / 'board').iterdir() if p.suffix in ('.tcl', '.v', '.xdc', '.tsv')]
    sources += [p for folder in ('hdl', 'subcore', 'constraints', 'bd') for p in (package / folder).rglob('*') if p.is_file()]
    sources += [package / 'component.xml', failed_record / 'state.json',
                U02 / 'inference/create_fp32_partial_uram_top.tcl']
    frozen = {str(p): sha(p) for p in sources}
    output = U02 / ('reports/fp32_partial_uram_top_recovery_' + str(time.time_ns()))
    output.mkdir()
    state = dict(complete=False, project=str(project), input_sha256=frozen,
                 rtl_record=str(rtl_record), ooc_record=str(ooc_record), package_record=str(package_record),
                 available_commit_before=memory.avail_commit, max_threads=1,
                 phased_processes=True, phases=[], recovery_source_record=str(failed_record),
                 recovery_scope='Saved BD synthesis products only; no prior synth or impl run is restarted',
                 board_accessed=False, board_programmed=False, startup_files_modified=False,
                 routed=False, deployment_accepted=False, formal_project_preserved=True)
    def save():
        (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    print('FP32_PARTIAL_URAM_TOP_IMPLEMENTATION_RECORD', output, flush=True)
    try:
        env = os.environ.copy()
        env.update(TEMP=str(U02 / 'toolchain/tmp'), TMP=str(U02 / 'toolchain/tmp'),
                   U02_FP32_IP_PACKAGE=str(package), U02_STREAM_PROJECT_DIR=str(project),
                   U02_STREAM_REPORT_DIR=str(output))
        # An unrelated inherited recovery flag must not open an existing build.
        env.pop('U02_STREAM_RESUME_EMPTY', None)
        for phase in ('synthesis', 'implementation', 'reports'):
            assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
            assert memory.avail_commit > 8 * 1024**3, ('Phase commit preflight', phase, memory.avail_commit)
            entry = dict(phase=phase, complete=False, available_commit_before=memory.avail_commit)
            state['phases'].append(entry)
            state['current_phase'] = phase
            save()
            env['U02_STREAM_BUILD_PHASE'] = phase
            phase_log = output / (phase + '.log')
            with phase_log.open('wb') as log:
                process = subprocess.Popen(['powershell.exe', '-NoProfile', '-Command',
                    "& 'D:/Develop/Xilink/Vivado/2018.3/bin/vivado.bat' -mode batch -source inference/recover_fp32_partial_uram_top.tcl -log '" +
                    str(output / (phase + '_vivado.log')) + "' -journal '" + str(output / (phase + '_vivado.jou')) + "'; exit $LASTEXITCODE"],
                    cwd=U02, env=env, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                state['pid'] = entry['pid'] = process.pid
                save()
                code = process.wait()
            entry.update(terminal=True, exit_code=code, log_sha256=sha(phase_log))
            save()
            text = phase_log.read_text(errors='replace')
            assert code == 0 and not re.search(r'(?m)^\s*ERROR:', text), phase
            assert all(sha(Path(p)) == h for p, h in frozen.items())
            entry['complete'] = True
            save()
        with (output / 'implementation.log').open('wb') as combined:
            for phase in state['phases']:
                combined.write((output / (phase['phase'] + '.log')).read_bytes())
        state.update(exit_code=0, log_sha256=sha(output / 'implementation.log'))
        save()
        text = (output / 'implementation.log').read_text(errors='replace')
        assert code == 0 and 'U02_STREAM_FULL_NETWORK_BACKEND_BIT_READY' in text
        assert not re.search(r'(?m)^\s*ERROR:', text)
        assert all(sha(Path(p)) == h for p, h in frozen.items())
        bit = project / 'u02_stream.runs/impl_1/compute_wrapper.bit'
        assert bit.is_file() and bit.stat().st_size > 7_000_000
        state.update(complete=True, routed=True, bitstream=str(bit), bitstream_sha256=sha(bit),
            reports_sha256={n: sha(output / n) for n in ('timing.rpt', 'utilization.rpt', 'drc.rpt', 'compute_bd.tcl', 'compute.ltx')},
            scope='Full top build finished; independent routed clock/timing/resource/reset/address audit and whole-network board A/B still required')
    except Exception as error:
        state['error'] = repr(error)
        raise
    finally:
        state['terminal'] = True
        save()


if __name__ == '__main__':
    main()
