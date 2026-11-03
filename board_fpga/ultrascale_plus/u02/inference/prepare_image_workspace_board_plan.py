"""Prepare local-only preflight evidence and explicit A/B/A commands.

Never executes the board tests. The existing owned camera runner is called only
with --preflight-only, which returns before UART/network/JTAG initialization.
The saved boot/session must be revalidated when the board becomes available.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import sys

from prepare_dw128_v2_board_ab import U02, sha, verify_frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('candidate_archive', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    archive, out = args.candidate_archive.resolve(), args.destination.resolve()
    verify_frozen(archive)
    result = json.loads((archive / 'result.json').read_text())
    assert result['reuse_level'] == 0 and not result['promoted'] and not result['board_accessed']
    assert not out.exists()
    owner = U02 / 'milestones/chw_camera_short_prepared_20261018/run_owned_chw_stereo.py'
    verify_frozen(owner.parent)
    baseline = U02 / 'milestones/stereo_half_frame_prepared_20261018/binaries/pose_stereo_pipeline_half_frame'
    candidate = archive / 'binaries/pose_stereo_pipeline_image_sampler'
    assert sha(baseline) == result['accepted_host_sha256']
    assert sha(candidate) == result['candidate_host']['sha256']
    restored = U02 / 'reports/formal_dw_dual_stereo_1792341526335480800'
    common = [sys.executable, str(owner), '--execute', '--chw-burst', '--seconds', '20',
        '--threads', '1', '--source-mode', '1080p', '--uvc1080-module',
        str(U02 / 'milestones/uvc1080_metadata_fix_prepared_20261017'),
        '--uvc-hwtimestamps', '1', '--spin-count', '300000', '--pl-poll-spin-us', '1000',
        '--strict-image-reference', '--bounded-clock-check', '--clock-capture-samples', '13',
        '--portable-hardware', str(U02 / 'milestones/portable_hardware_prepared_20261015/runtime_tree'),
        '--pipeline-pairs', '2', '--pair-limit-us', '16667', '--restored-record', str(restored)]
    out.mkdir(parents=True)
    stages = []
    for label, host in (('baseline_before', baseline), ('sampler_candidate', candidate), ('baseline_after', baseline)):
        argv = common + ['--candidate-host', str(host), '--candidate-host-sha256', sha(host)]
        preflight = subprocess.run(argv + ['--preflight-only'], capture_output=True)
        (out / (label + '.stdout')).write_bytes(preflight.stdout)
        (out / (label + '.stderr')).write_bytes(preflight.stderr)
        assert preflight.returncode == 0, label
        proof = json.loads(preflight.stdout)
        assert proof['board_accessed'] is False and proof['host_sha256'] == sha(host)
        assert proof['bit_sha256'] == result['required_unchanged_bit_sha256']
        stages.append(dict(label=label, argv=argv, host_sha256=sha(host), preflight=proof))
    plan = dict(scope='Prepared only; no UART/JTAG/board access', board_accessed=False,
        requires_board_ownership_and_live_session_revalidation=True,
        saved_restored_state_sha256=sha(restored / 'state.json'),
        owner_sha256=sha(owner), stages=stages,
        next_steps=['Run frozen AArch64 image test with the identical eight JPEGs and PNG.',
            'Revalidate current boot and restored ownership before any board operation.',
            'Run owned 20s A/B/A with fixed hardware, driver, clocks and pairing; compare latency, throughput and numerical references.',
            'Promote only after actual board correctness/performance acceptance; otherwise keep accepted Host.'])
    (out / 'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    (out / 'executed_prepare.py').write_bytes(Path(__file__).read_bytes())
    (out / '.gitattributes').write_bytes(b'* -text whitespace=cr-at-eol,-blank-at-eof\n')
    (out / '.gitignore').write_bytes(b'!*\n!*/\n')
    manifest = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
    (out / 'manifest.json').write_text(json.dumps(dict(files_sha256=manifest), indent=2), encoding='utf-8')
    verify_frozen(out)
    print(json.dumps(dict(prepared=str(out), board_accessed=False, stages=len(stages)), indent=2))


if __name__ == '__main__':
    main()
