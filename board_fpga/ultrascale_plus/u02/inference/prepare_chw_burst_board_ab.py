"""Prepare a matched DW128/CHW/DW128 owner from existing verified safety helpers."""
import argparse
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_dw128_v2_board_ab import U02, sha, verify_frozen
from prepare_dw128_v2_hardware import render
from prepare_dw128_v2_ab_owner import generate as original_owner, compare_children
from prepare_dw128_v2_pair_child import generate as child_source

PREPARED = U02 / 'milestones/chw_burst_board_ab_prepared_20261018'


def hardware_tree():
    verify_frozen(PREPARED)
    descriptor = json.loads((PREPARED / 'descriptor.json').read_text())
    assert descriptor['schema'] == 'u02_chw_burst_hardware_tree_v1'
    result = {}
    for label in ('original', 'baseline', 'candidate'):
        item = descriptor['hardware'][label]
        project, reports = PREPARED / item['project'], PREPARED / item['reports']
        bit = project / 'u02_stream.runs/impl_1/compute_wrapper.bit'
        assert sha(bit) == item['bit_sha256']
        assert sha(reports / 'compute.ltx') == item['ltx_sha256']
        result[label] = dict(project=project, reports=reports, bit=bit)
    assert len({sha(item['bit']) for item in result.values()}) == 3
    return result


def prepared_children():
    hardware_tree()
    return {label: PREPARED / (label + '_pair_child.py') for label in ('baseline', 'candidate')}


def generate_owner():
    source, required = original_owner()
    before = 'from prepare_dw128_v2_ab_owner import hardware_tree,prepared_children,compare_children'
    assert source.count(before) == 1
    source = source.replace(before, 'from prepare_chw_burst_board_ab import hardware_tree,prepared_children,compare_children')
    before = "(('baseline_before','dw118'),('candidate','candidate'),('baseline_after','dw118'))"
    assert source.count(before) == 1
    source = source.replace(before, "(('baseline_before','baseline'),('candidate','candidate'),('baseline_after','baseline'))")
    source = source.replace('Owned DW118 / candidate / DW118 complete-network A/B',
                            'Owned DW128 / CHW burst / DW128 complete-network A/B')
    before = "owner_pid=os.getpid(),remote=remote,results=results,boot_id=prior['boot_id'],"
    assert source.count(before) == 1
    source = source.replace(before, before + "\n        candidate_kind='chw_burst',baseline_kind='dw128_write_v2',")
    # Keep the existing owner schema: the unchanged child checks the same safety contract.
    for p in (Path(__file__), U02 / 'inference/prepare_dw128_v2_hardware.py'):
        required[str(p.resolve())] = sha(p)
    assignment = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'REQUIRED_SOURCES' for t in n.targets))
    lines = source.splitlines(keepends=True)
    lines[assignment.lineno - 1:assignment.end_lineno] = ['REQUIRED_SOURCES=' + repr(required) + '\n']
    source = ''.join(lines)
    compile(source, '<chw-board-owner>', 'exec')
    # All clock, loading, ownership, rollback and recovery helpers remain identical.
    baseline, _ = original_owner()
    def helpers(text):
        main = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
        return {n.name: ast.dump(n, include_attributes=False) for n in main.body if isinstance(n, ast.FunctionDef)}
    assert helpers(source) == helpers(baseline)
    return source, required


def prepare():
    assert not PREPARED.exists()
    routed = U02 / 'milestones/chw_burst_routed_118m_20261018'
    verify_frozen(routed)
    summary = json.loads((routed / 'summary.json').read_text())
    assert summary['route_checks_passed'] and summary['effective_clock_guards_passed']
    assert not summary['board_verified'] and not summary['formal_promoted']
    old = U02 / 'milestones/dw128_v2_hardware_prepared_20261017'
    verify_frozen(old)
    prior = json.loads((old / 'descriptor.json').read_text())
    artifacts = {}
    for role, old_role in (('original', 'original'), ('baseline', 'candidate')):
        item = prior['hardware'][old_role]
        artifacts[role] = (old / item['project'] / 'u02_stream.runs/impl_1/compute_wrapper.bit',
                           old / item['reports'] / 'compute.ltx')
    artifacts['candidate'] = (routed / 'compute_wrapper.bit', routed / 'compute.ltx')
    assert sha(artifacts['baseline'][0]) == '9af64d2d9b135fba7c8d7f704b0bbdcccb8cf124e68f33cc1d9e80600e32e2f4'
    assert sha(artifacts['candidate'][0]) == summary['bit_sha256']
    source, required = generate_owner()
    PREPARED.mkdir()
    descriptor = dict(schema='u02_chw_burst_hardware_tree_v1', offline_prepared=True,
        board_accessed=False, hardware_executed=False, formal_promoted=False,
        routed_manifest_sha256=sha(routed / 'manifest.json'), hardware={})
    for label, (bit, ltx) in artifacts.items():
        project, reports = PREPARED / label / 'project', PREPARED / label / 'reports'
        target = project / 'u02_stream.runs/impl_1/compute_wrapper.bit'
        target.parent.mkdir(parents=True)
        reports.mkdir()
        shutil.copyfile(bit, target)
        shutil.copyfile(ltx, reports / 'compute.ltx')
        descriptor['hardware'][label] = dict(project=project.relative_to(PREPARED).as_posix(),
            reports=reports.relative_to(PREPARED).as_posix(), bit_sha256=sha(target), ltx_sha256=sha(reports / 'compute.ltx'))
        for kind in ('program', 'verify', 'idle_readonly'):
            template = 'program_stream.tcl' if kind == 'program' else 'verify_resident_interface.tcl'
            (PREPARED / label / (kind + '.tcl')).write_text(render((U02 / 'inference' / template).read_text(), project, reports, kind))
        if label != 'original':
            child = child_source((U02 / 'inference/validate_pair_network_dual.py').read_text(), project, reports, sha(target))
            (PREPARED / (label + '_pair_child.py')).write_text(child)
    (PREPARED / 'run_owned_ab.py').write_text(source)
    for name in ('run_owned_ab', 'baseline_pair_child', 'candidate_pair_child'):
        result = subprocess.run([sys.executable, str(PREPARED / (name + '.py')), '--help'], capture_output=True, text=True, check=True)
        (PREPARED / (name + '_help.log')).write_text(result.stdout + result.stderr)
    descriptor.update(helper_sources=required, sequence=['DW128 before', 'CHW candidate', 'DW128 after', 'original rollback'],
        scope='Prepared full-network pair A/B only; no camera or pipeline throughput claim')
    (PREPARED / 'descriptor.json').write_text(json.dumps(descriptor, indent=2) + '\n')
    shutil.copyfile(Path(__file__), PREPARED / 'executed_prepare.py')
    (PREPARED / '.gitattributes').write_bytes(b'* -text whitespace=cr-at-eol,-blank-at-eof\n')
    (PREPARED / '.gitignore').write_bytes(b'!*\n!*/\n')
    files = {p.relative_to(PREPARED).as_posix(): sha(p) for p in sorted(PREPARED.rglob('*')) if p.is_file()}
    (PREPARED / 'manifest.json').write_text(json.dumps(dict(files_sha256=files), indent=2) + '\n')
    hardware_tree()
    print('CHW_BOARD_AB_OFFLINE_PREPARED', PREPARED)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    prepare()
