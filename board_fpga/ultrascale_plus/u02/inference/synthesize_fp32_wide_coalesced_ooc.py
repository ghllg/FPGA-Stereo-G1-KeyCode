"""Independent wide-memory mapping; diagnostic synthesis cannot authorize deployment."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from audit_fp32_wide_synthesis import audit, wide_register_map
from audit_fp32_synthesis import signed_leaves
from prepare_stream_ip import verify_compute_sharing
from synthesize_fp32_ooc import U02, sha, check_rtl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rtl_record', type=Path, nargs='?')
    parser.add_argument('--synthesis-record', type=Path,
                        help='Mapping diagnosis while numeric RTL is pending; no deployment acceptance')
    args = parser.parse_args()
    assert bool(args.rtl_record) != bool(args.synthesis_record)
    diagnostic = args.synthesis_record is not None
    record = (args.synthesis_record or args.rtl_record).resolve()
    assert record.parent == U02 / 'reports'
    if diagnostic:
        structure = audit(record)
        proof = json.loads((record / 'state.json').read_text())
    else:
        proof = check_rtl(record)
        structure = audit(Path(proof['synthesis_record']))
    hls = Path(proof['project'])
    assert hls == U02 / 'inference/stream_hls_22_fp32_wide_coalesced'
    assert structure['meets_clock_estimate']
    assert structure['fits_bram_estimate'] or (
        288 < structure['estimated_resources']['BRAM_18K'] <= 320 and
        structure['uram_staging_intent']['instances'] == 8)
    # OOC resolves the HLS BRAM/UltraRAM discrepancy; deployment still requires
    # actual resource and route acceptance, regardless of the HLS estimate.
    source = hls / 'zu5ev/syn/verilog'
    if not diagnostic:
        assert {p.name: sha(p) for p in source.glob('*.v')} == proof['rtl_sha256']
    assert shutil.disk_usage(U02).free > 10 * 1024**3
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ('total_phys', 'avail_phys', 'total_commit',
                                            'avail_commit', 'total_virtual', 'avail_virtual', 'extended')]
    memory = MemoryStatus(); memory.length = ctypes.sizeof(memory)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory))
    assert memory.avail_commit > 8 * 1024**3, ('Insufficient available commit', memory.avail_commit)
    inputs = U02 / 'inference/fp32_wide_coalesced_ooc_inputs'
    project = U02 / 'inference/fp32_wide_coalesced_ooc_project'
    assert not inputs.exists() and not project.exists(), 'Preserve previous attempts'
    # Include initialization files and floating IP scripts, not only Verilog.
    files = [p for p in source.iterdir() if p.suffix in ('.v', '.dat') or p.name.endswith('_ip.tcl')]
    assert len([p for p in files if p.name.endswith('_ip.tcl')]) == 4
    leaves = signed_leaves(source)
    inputs.mkdir()
    staged = {}
    for path in files:
        raw = path.read_bytes()
        if path.name in leaves:
            text = raw.decode('ascii')
            needle = 'module ' + leaves[path.name] + '('
            assert text.count(needle) == 1
            raw = text.replace(needle, '(* use_dsp = "yes" *)\n' + needle).encode('ascii')
        target = inputs / path.name
        target.write_bytes(raw)
        staged[path.name] = dict(source_sha256=sha(path), staged_sha256=sha(target))
    assert signed_leaves(inputs) == leaves
    assert wide_register_map(inputs) == structure['register_map']
    assert verify_compute_sharing(inputs, 'requant_words') == structure['compute_sharing']
    for path in inputs.glob('*.v'):
        for filename in re.findall(r'\$readmemh\("(?:\./)?([^"/]+)"', path.read_text()):
            assert (inputs / filename).is_file(), ('Missing RAM initialization', filename)
    script = U02 / 'inference/fp32_wide_ooc_single_thread.tcl'
    frozen = {str(p): sha(p) for p in [script, Path(__file__), record / 'state.json']}
    frozen.update({str(inputs / n): e['staged_sha256'] for n, e in staged.items()})
    out = U02 / ('reports/fp32_wide_coalesced_ooc_' + str(time.time_ns()))
    out.mkdir()
    state = dict(complete=False, max_threads=1, available_commit_before=memory.avail_commit, project=str(project), proof_record=str(record),
                 rtl_record=None if diagnostic else str(record),
                 numeric_rtl_verified=not diagnostic, diagnostic_only=diagnostic,
                 deployment_accepted=False,
                 structure=structure, staged_files=staged, input_sha256=frozen,
                 board_accessed=False, routed=False, hardware_verified=False)
    def save():
        (out / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        env = os.environ.copy()
        env.update(TEMP=str(U02 / 'toolchain/tmp'), TMP=str(U02 / 'toolchain/tmp'),
                   U02_FP32_COMPLETE_INPUTS=str(inputs), U02_FP32_COMPLETE_PROJECT=str(project))
        with (out / 'synthesis.log').open('wb') as log:
            process = subprocess.Popen(['powershell.exe', '-NoProfile', '-Command',
                "& 'D:/Develop/Xilink/Vivado/2018.3/bin/vivado.bat' -mode batch -source inference/fp32_wide_ooc_single_thread.tcl -log '" +
                str(out / 'vivado.log') + "' -journal '" + str(out / 'vivado.jou') + "'; exit $LASTEXITCODE"],
                cwd=U02, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid'] = process.pid
            save()
            code = process.wait()
        state.update(terminal=True, exit_code=code, log_sha256=sha(out / 'synthesis.log'))
        save()
        logtext = (out / 'synthesis.log').read_text(errors='replace')
        assert code == 0 and not re.search(r'(?m)^\s*ERROR:', logtext)
        assert 'FP32_COMPLETE_INPUTS_OOC_COMPLETE' in logtext
        assert all(sha(Path(p)) == h for p, h in frozen.items())
        assert all(sha(source / n) == e['source_sha256'] for n, e in staged.items())
        state.update(complete=True, artifacts_sha256={n: sha(project / n) for n in
                     ('utilization.rpt', 'timing.rpt', 'drc.rpt', 'fp32_output_ooc.dcp')},
                     scope='OOC synthesis only; actual resource audit, top route and same-board whole-model comparison remain')
    except Exception as error:
        state['error'] = repr(error)
        raise
    finally:
        save()
        print('FP32_WIDE_COALESCED_OOC_RECORD', out, flush=True)


if __name__ == '__main__':
    main()
