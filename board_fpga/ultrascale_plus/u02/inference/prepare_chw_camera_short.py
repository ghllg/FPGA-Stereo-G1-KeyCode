"""Prepare existing 1080p camera owner/child with explicitly accepted CHW identity."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys

from prepare_dw128_v2_board_ab import U02, sha, verify_frozen
from prepare_pipeline_irq_board import replace_once
from chw_burst_camera_hardware import accepted_candidate

OUT = U02 / 'milestones/chw_camera_short_prepared_20261018'


def main():
    assert not OUT.exists()
    accepted_candidate()
    archive = U02 / 'milestones/stereo_half_frame_300s_verified_20261018'
    verify_frozen(archive)
    owner_path = archive / 'hardware_owner/executed_script.py'
    child_path = U02 / 'inference/validate_dual_stereo.py'
    owner, child = owner_path.read_text(), child_path.read_text()
    assert owner.count('from dw128_v2_stereo_hardware import') == 2
    assert child.count('from dw128_v2_stereo_hardware import') == 2
    owner = owner.replace('from dw128_v2_stereo_hardware import', 'from chw_burst_camera_hardware import')
    child = child.replace('from dw128_v2_stereo_hardware import', 'from chw_burst_camera_hardware import')
    owner = replace_once(owner, "state['hardware_variant']='dw128_v2' if args.dw128_v2 else 'dw118'",
        "state['hardware_variant']='chw_burst' if args.dw128_v2 else 'dw118'")
    child = replace_once(child, "('dw128_v2' if dw128 else 'dw118')", "('chw_burst' if dw128 else 'dw118')")
    owner = replace_once(owner, "implementation=U02/'milestones/dw128_v2_board_ab_verified_20261017'",
        "implementation=U02/'milestones/chw_pipeline_board_verified_20261018'")
    owner = replace_once(owner, "child_source=U02/'inference/validate_dual_stereo.py'",
        'child_source=Path(' + repr(str(OUT / 'validate_chw_stereo_child.py')) + ')')
    owner = replace_once(owner, "parser.add_argument('--dw128-v2',action='store_true'",
        "parser.add_argument('--chw-burst',dest='dw128_v2',action='store_true'")
    owner = owner.replace('Experimental board-A/B-verified DW128 hardware; no formal promotion',
        'Accepted CHW hardware; temporary camera test only')
    child = replace_once(child, 'from retest_original_after_region import',
        'import sys\nsys.path.insert(0,' + repr(str(U02 / 'inference')) + ')\nfrom retest_original_after_region import')
    paths = [Path(__file__), U02 / 'inference/chw_burst_camera_hardware.py',
        U02 / 'inference/freeze_chw_pipeline_board.py', owner_path, child_path]
    required = {str(p.resolve()): sha(p) for p in paths}
    for label, before, after in (('owner', owner_path.read_text(), owner), ('child', child_path.read_text(), child)):
        def helpers(text):
            main = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'main')
            return {n.name: ast.dump(n, include_attributes=False) for n in main.body if isinstance(n, ast.FunctionDef)}
        assert helpers(before) == helpers(after), label
    OUT.mkdir()
    for name, text in (('run_owned_chw_stereo.py', owner), ('validate_chw_stereo_child.py', child)):
        compile(text, name, 'exec')
        path = OUT / name
        path.write_text(text, encoding='utf-8')
        result = subprocess.run([sys.executable, str(path), '--help'], capture_output=True, check=True)
        (OUT / (name + '.help.log')).write_bytes(result.stdout + result.stderr)
    result = subprocess.run([sys.executable, str(U02 / 'inference/test_stereo_owner_observation.py'),
        str(OUT / 'run_owned_chw_stereo.py'), '--output', str(OUT / 'observation_tests.json')], capture_output=True, check=True)
    (OUT / 'observation_tests.log').write_bytes(result.stdout + result.stderr)
    shutil.copyfile(Path(__file__), OUT / 'executed_prepare.py')
    (OUT / 'preparation.json').write_text(json.dumps(dict(complete=True, board_accessed=False,
        required_sources=required, helpers_unchanged=True, seconds=20,
        scope='Same accepted half-frame 1080p Host and camera path; CHW hardware identity only'), indent=2) + '\n')
    (OUT / '.gitattributes').write_bytes(b'* -text whitespace=cr-at-eol,-blank-at-eof\n')
    (OUT / '.gitignore').write_bytes(b'!*\n!*/\n')
    files = {p.name: sha(p) for p in OUT.iterdir() if p.is_file()}
    (OUT / 'manifest.json').write_text(json.dumps(dict(files_sha256=files), indent=2) + '\n')
    verify_frozen(OUT)
    print('CHW_CAMERA_SHORT_PREPARED', OUT)


if __name__ == '__main__':
    main()
