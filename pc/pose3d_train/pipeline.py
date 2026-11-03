"""Reproducible saved-pose -> strict geometry -> learned completion -> checked BVH."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from .bvh import read_bvh
from .geometry import Stereo
from .infer import infer, load_pose
from .model import load_model
from mocap.io import read_json, write_json
from mocap.reconstruction import reconstruct
from mocap.lafan import export_lafan, NAMES


def replay(pose, calibration, checkpoint, output, max_sample_interval_s=None, single_frame_gap_s=0.):
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a new directory.')
    stereo = Stereo(calibration)
    _, model = load_model(checkpoint)
    if model['calibration_hash'] != stereo.hash:
        raise ValueError('Checkpoint/calibration mismatch; do not bypass fingerprint checks.')
    meta, rows = load_pose(pose, stereo)
    output.mkdir(parents=True)
    report = dict(schema='pose3d_delivery_run/1.0', state='incomplete', stage='geometry',
                  experimental=True, production_accepted=False, real_accuracy_validated=False,
                  provisional_calibration=bool(stereo.calibration.get('diagnostic_only', False)),
                  source_pose=str(Path(pose).resolve()), source_sha256=hashlib.sha256(Path(pose).read_bytes()).hexdigest(),
                  calibration_hash=stereo.hash, checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                  exposure_sync_verified=False, hardware_accessed=False)
    report.update(source_2d_backend=meta.get('source_2d_backend'),
                  input_hardware_binding=meta.get('hardware_inference_binding'),
                  live_acquisition_validated=meta.get('live_acquisition_validated', False))
    try:
        strict = output/'strict.jsonl'
        hybrid = output/'skeleton3d.jsonl'
        reconstruct(pose, calibration, strict, max_delta_ms=10., max_reprojection_px=5., tau=0.)
        report['stage'] = 'learned_completion'
        infer(pose, calibration, checkpoint, hybrid, tau=0., max_delta_ms=10.)
        a = [json.loads(s) for s in strict.read_text(encoding='utf-8').splitlines()]
        b = [json.loads(s) for s in hybrid.read_text(encoding='utf-8').splitlines()]
        preserved, added = 0, 0
        missing = {}
        for old, new in zip(a,b):
            if old['timestamp_ns'] != new['timestamp_ns']:
                raise ValueError('Source timeline changed.')
            for index, (x,y) in enumerate(zip(old['joints'],new['joints'])):
                if x['valid']:
                    if not y['valid'] or not np.allclose(x['xyz'],y['xyz'],atol=1e-10,rtol=0):
                        raise ValueError('A reliable geometric observation was moved.')
                    preserved += 1
                elif y['valid']:
                    if not y.get('estimated') or y.get('reason') != 'learned_body_completion':
                        raise ValueError('Missing completion provenance.')
                    added += 1
                if index >= 5 and not y['valid']:
                    key = rows[0]['joint_names'][index]+':'+y['reason']
                    missing[key] = missing.get(key,0)+1
        mask_a = np.array([[j['valid'] for j in p['joints'][5:]] for p in a])
        mask_b = np.array([[j['valid'] for j in p['joints'][5:]] for p in b])
        report.update(frames=len(b), geometry_joints_preserved=preserved, learned_added=added,
                      body_coverage_before=float(mask_a.mean()), body_coverage_after=float(mask_b.mean()),
                      complete_body_frames=int(mask_b.all(axis=1).sum()), remaining_missing=missing)
        report['stage'] = 'bvh_export'
        try:
            export = export_lafan(hybrid, output/'bvh', fps=30., max_gap_s=.15,
                                  max_sample_interval_s=max_sample_interval_s, single_frame_gap_s=single_frame_gap_s)
        except ValueError as exc:
            if '没有可导出的连续全身片段' not in str(exc):
                raise
            report.update(quality_status='no_exportable_segment', bvh_error=str(exc), bvh_files=[])
        else:
            checks = []
            for item in export['segments']:
                path = output/'bvh'/item['file']
                names, parents, xyz, dt = read_bvh(path.read_text(encoding='ascii'))
                with np.load(path.with_suffix('.npz')) as saved:
                    error = float(np.max(np.abs(xyz*100-saved['global_positions_cm'])))
                if names != list(NAMES) or abs(dt-1/30) > 1e-8 or error > .001:
                    raise ValueError('BVH round-trip check failed.')
                checks.append(dict(file=item['file'], frames=len(xyz), max_fk_error_cm=error))
            report.update(quality_status=export['quality_status'], exported_duration_s=export['exported_duration_s'],
                          bvh_coverage=export['coverage_ratio'], bvh_files=checks,
                          bvh_missing_joint_gap_s=.15, bvh_max_sample_interval_s=export['max_sample_interval_s'],
                          bvh_single_frame_gap_s=single_frame_gap_s,
                          bvh_single_frame_filled_joint_count=export['single_frame_filled_joint_count'])
        report.update(state='complete', stage='done')
    except Exception as exc:
        report.update(state='failed', error=str(exc))
        raise
    finally:
        write_json(output/'pipeline.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['pose','calibration','checkpoint','output']:
        parser.add_argument('--'+flag, required=True)
    args = parser.parse_args()
    replay(args.pose, args.calibration, args.checkpoint, args.output)
