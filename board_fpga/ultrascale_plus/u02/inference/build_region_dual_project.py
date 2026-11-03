"""Build an isolated candidate PS/PL design; never access or deploy to a board."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
from export_region_dual_ip import U02, verify_hashes
from audit_region_dual_ooc import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-record', required=True, type=Path)
    parser.add_argument('--project-dir', required=True, type=Path)
    parser.add_argument('--report-dir', required=True, type=Path)
    args = parser.parse_args()
    package_record = args.package_record.resolve()
    proof_path = package_record / 'state.json'
    package = json.loads(proof_path.read_text())
    if (not package.get('terminal') or not package.get('complete') or
            package.get('exit_code') != 0 or not package.get('subcore_target_family_verified')):
        raise ValueError('Successful source-bound ZU5EV package required')
    verify_hashes(package['input_sha256'])
    ip = Path(package['package'])
    verify_hashes(package['subcore_xci_sha256'], ip)
    if len(package['subcore_xci_sha256']) != 13 or sha(ip / 'component.xml') != package['component_sha256']:
        raise ValueError('Incomplete candidate component or dependencies')
    project, reports = args.project_dir.resolve(), args.report_dir.resolve()
    for path in (project, reports):
        if path.exists() or path.drive.upper() not in ('D:', 'E:'):
            raise ValueError('Fresh D/E outputs required: ' + str(path))
    if project == reports or project in reports.parents or reports in project.parents:
        raise ValueError('Independent project and report directories required')
    script = Path(__file__).with_name('create_region_dual_project.tcl')
    board = U02 / 'board'
    sources = [script, Path(__file__), proof_path, ip / 'component.xml',
               Path(__file__).with_name('export_region_dual_ip.py'),
               Path(__file__).with_name('audit_region_dual_ooc.py'),
               board / 'por_reset.v', board / 'ps_config_2018.tcl',
               board / 'ps_required_baseline.tsv', board / 'add_clock_probe.tcl']
    frozen = {str(p): sha(p) for p in sources}
    frozen.update({str(p): sha(p) for p in ip.rglob('*') if p.is_file() and
                   p.parts[len(ip.parts)] in ('hdl', 'subcore', 'xgui')})
    record = U02 / ('reports/region_dual_full_build_' + str(time.time_ns()))
    record.mkdir()
    state = dict(complete=False, terminal=False, project=str(project), reports=str(reports),
                 package_record=str(package_record), input_sha256=frozen,
                 board_accessed=False, deployment_accepted=False,
                 rtl_postcheck_verified=package.get('rtl_postcheck_verified', False),
                 scope='Offline full PS/PL build; source-bound numeric RTL and complete-network board A/B remain mandatory deployment gates.')
    def save():
        (record / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    save()
    print('U02_REGION_DUAL_FULL_BUILD_RECORD', record, flush=True)
    try:
        env = os.environ.copy()
        env.update(TEMP=str(U02 / 'toolchain/tmp'), TMP=str(U02 / 'toolchain/tmp'),
                   U02_REGION_PROJECT_DIR=str(project), U02_REGION_REPORT_DIR=str(reports),
                   U02_REGION_IP_DIR=str(ip), U02_REGION_IP_VLNV='u02:inference:u02_region_dual:1.0')
        with (record / 'build.log').open('wb') as output:
            process = subprocess.Popen(
                ['D:/Develop/Xilink/Vivado/2018.3/bin/vivado.bat', '-mode', 'batch',
                 '-source', str(script), '-log', str(record / 'vivado.log'),
                 '-journal', str(record / 'vivado.jou')], cwd=record, env=env,
                stdout=output, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            state['pid'] = process.pid
            save()
            state['exit_code'] = process.wait()
        log = (record / 'build.log').read_text(errors='replace')
        if (state['exit_code'] or 'U02_REGION_DUAL_CANDIDATE_BIT_READY_NOT_DEPLOYED' not in log or
                re.search(r'(?m)^\s*ERROR:', log)):
            raise ValueError('Full candidate build failed; preserve outputs for diagnosis')
        verify_hashes(frozen)
        bit = project / 'u02_region_dual.runs/impl_1/compute_wrapper.bit'
        artifacts = [bit, *[reports / name for name in
                     ('timing.rpt', 'utilization.rpt', 'drc.rpt', 'compute_bd.tcl', 'compute.ltx')]]
        state.update(complete=True, bitstream_generated=True,
                     artifacts_sha256={str(p): sha(p) for p in artifacts},
                     routed_reports_audited=False)
    except Exception as error:
        state['error'] = str(error)
        raise
    finally:
        state.update(terminal=True, log_sha256=sha(record / 'build.log'))
        save()
        print('U02_REGION_DUAL_FULL_BUILD_DONE', record, flush=True)


if __name__ == '__main__':
    main()
