"""Prepare an isolated mapped IP; numerical acceptance remains a separate gate."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from audit_region_dual_ooc import sha

U02 = Path(__file__).resolve().parents[1]


def verify_hashes(entries, base=None):
    for name, digest in entries.items():
        path = base / name if base else Path(name)
        if not path.is_absolute():
            path = U02 / path
        if sha(path) != digest:
            raise ValueError('Evidence changed: ' + str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mapping-archive', required=True, type=Path)
    evidence = parser.add_mutually_exclusive_group(required=True)
    evidence.add_argument('--rtl-archive', type=Path)
    evidence.add_argument('--rtl-setup-record', type=Path,
                          help='Offline preparation while numeric RTL runs; never deployment acceptance')
    parser.add_argument('--project-dir', required=True, type=Path)
    args = parser.parse_args()
    mapped = args.mapping_archive.resolve()
    numeric = args.rtl_archive.resolve() if args.rtl_archive else None
    audit_path = mapped / 'audit.json'
    audit = json.loads(audit_path.read_text())
    if not audit.get('eligible_for_implementation_review'):
        raise ValueError('Mapped capacity, clock and DRC gates required')
    map_state_path = mapped / 'state.json'
    if sha(map_state_path) != audit['state_sha256']:
        raise ValueError('Mapping audit/state mismatch')
    mapping = json.loads(map_state_path.read_text())
    verify_hashes(mapping['input_sha256'])
    verify_hashes(mapping['artifacts_sha256'], Path(mapping['project']))
    evidence_paths = []
    if numeric:
        archive = json.loads((numeric / 'archive.json').read_text())
        verify_hashes(archive['archive_sha256'], numeric)
        rtl = json.loads((numeric / 'state.json').read_text())
        if not all(rtl.get(key) is True for key in ('terminal', 'complete', 'rtl_postcheck_verified')):
            raise ValueError('Completed numeric RTL gate required')
        verify_hashes(rtl['input_sha256'])
        verify_hashes(rtl['simulation_inputs_sha256'], Path(rtl['project']) / 'sim')
        verify_hashes(rtl['rtl_outputs_sha256'])
        if sha(Path(rtl['project']) / 'sim/verilog/xsim.dir/u02_region_dual/xsimk.exe') != rtl['snapshot_sha256']:
            raise ValueError('Numeric simulation snapshot changed')
        setup_path = Path(rtl['previous_record']) / 'state.json'
        evidence_paths.extend((numeric / 'archive.json', numeric / 'state.json'))
    else:
        setup_path = args.rtl_setup_record.resolve() / 'state.json'
    setup = json.loads(setup_path.read_text())
    if (not all(setup.get(key) is True for key in ('terminal', 'prepared', 'setup_succeeded')) or
            setup.get('expected_transactions') != 5 or setup.get('exit_code') != 0):
        raise ValueError('Five-transaction, source-bound simulation preparation required')
    verify_hashes(setup['input_sha256'])
    verify_hashes(setup['rtl_sha256'], Path(setup['project']) / 'zu5ev/syn/verilog')
    evidence_paths.append(setup_path)
    hls_record = Path(mapping['hls_record'])
    if Path(setup['hls_record']).resolve() != hls_record.resolve():
        raise ValueError('Numeric RTL and mapped RTL originate from different HLS solutions')
    project = args.project_dir.resolve()
    if project.exists() or project.drive.upper() not in ('D:', 'E:'):
        raise ValueError('Fresh D/E export project required')
    original = hls_record / 'hls_project'
    shutil.copytree(original, project,
                    ignore=shutil.ignore_patterns('impl', 'csim', 'sim', '.Xil'))
    raw = original / 'zu5ev/syn/verilog'
    frozen = {str(p): sha(p) for p in raw.iterdir() if p.is_file()}
    verify_hashes({p.name: sha(p) for p in raw.iterdir() if p.is_file()},
                  project / 'zu5ev/syn/verilog')
    script = Path(__file__).with_name('hls') / 'export_region_dual.tcl'
    for path in (script, Path(__file__), audit_path, map_state_path, *evidence_paths):
        frozen[str(path)] = sha(path)
    record = U02 / ('reports/region_dual_ip_export_' + str(time.time_ns()))
    record.mkdir()
    state = dict(complete=False, terminal=False, project=str(project),
                 input_sha256=frozen, mapping_archive=str(mapped),
                 rtl_archive=str(numeric) if numeric else None,
                 rtl_setup_record=str(setup_path.parent),
                 rtl_postcheck_verified=bool(numeric),
                 preparation_only=not bool(numeric),
                 board_accessed=False, deployment_accepted=False,
                 hls_synthesis_rerun=False, rtl_simulation_rerun=False,
                 scope='Package export only; exact staged DSP attributes, subcores, top route and board A/B remain gates.')
    def save():
        (record / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    try:
        env = os.environ.copy()
        env.update(TEMP=str(U02 / 'toolchain/tmp'), TMP=str(U02 / 'toolchain/tmp'),
                   U02_REGION_EXPORT_PROJECT=str(project))
        with (record / 'export.log').open('wb') as output:
            process = subprocess.Popen(
                ['D:/Develop/Xilink/Vivado/2018.3/bin/vivado_hls.bat', '-f', str(script)],
                cwd=record, env=env, stdout=output, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid'] = process.pid
            save()
            state['exit_code'] = process.wait()
        verify_hashes(frozen)
        verify_hashes({p.name: sha(p) for p in raw.iterdir() if p.is_file()},
                      project / 'zu5ev/syn/verilog')
        log = (record / 'export.log').read_text(errors='replace')
        if (state['exit_code'] or 'U02_REGION_DUAL_EXPORT_FINISHED' not in log or
                re.search(r'(?m)^\s*(?:ERROR|CRITICAL WARNING):', log)):
            raise ValueError('Export failed; preserve this project and diagnose before another attempt')
        component = project / 'zu5ev/impl/ip/component.xml'
        state.update(complete=True, component_sha256=sha(component),
                     package_dependencies_audited=False)
    except Exception as error:
        state['error'] = str(error)
        raise
    finally:
        state.update(terminal=True, log_sha256=sha(record / 'export.log'))
        save()
        print('U02_REGION_DUAL_IP_EXPORT_RECORD', record, flush=True)


if __name__ == '__main__':
    main()
