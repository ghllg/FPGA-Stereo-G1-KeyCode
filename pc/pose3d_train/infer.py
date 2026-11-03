"""Paired COCO17 pixel JSONL -> hybrid 3D JSONL, with explicit learned provenance."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import cv2
import numpy as np
import torch
from .geometry import BODY, Stereo
from .model import load_model
from mocap.io import COCO17, frame_packet, read_json, write_json
from mocap.reconstruction import CausalFilter, triangulate


def complete_missing(stereo, prediction, base, observed, pixels, confidence, single_hip_anchor=False, single_view_ray=False, conflict_model=None):
    """Never move a reliable triangulation. Completion is an experimental estimate."""
    if single_view_ray:
        prediction = prediction.copy()
        visible = confidence >= .35
        cam0 = (prediction-stereo.w[:3,3])@stereo.w[:3,:3]
        for side in range(2):
            mask = visible[:,side] & ~visible[:,1-side] & ~observed
            camera = cam0 if side == 0 else cam0@stereo.r.T+stereo.t
            normalized = cv2.undistortPoints(pixels[:,side].reshape(-1,1,2).astype(float),
                                             stereo.k[side],stereo.d[side]).reshape(-1,12,2)
            adjusted = np.concatenate([normalized,np.ones((*normalized.shape[:-1],1))],axis=-1)*camera[...,2:3]
            if side:adjusted = (adjusted-stereo.t)@stereo.r
            world = adjusted@stereo.w[:3,:3].T+stereo.w[:3,3]
            # Keep the learned depth; a single measured ray cannot determine true 3D depth.
            mask &= (camera[...,2] > .1) & (np.linalg.norm(world-prediction,axis=-1) <= .2)
            prediction[mask] = world[mask]
    projected, depth = stereo.project(prediction)
    visible = confidence >= .35
    errors = np.where(visible, np.linalg.norm(projected-pixels, axis=-1), 0.)
    root = base[:, 6:8].mean(axis=1)
    hip_anchor = observed[:,6:8].any(axis=1) if single_hip_anchor else observed[:,6:8].all(axis=1)
    anchored = hip_anchor & (observed.sum(axis=1) >= 6)
    accepted = (anchored[:, None] & np.isfinite(prediction).all(axis=-1)
                & (depth > .1).all(axis=1) & (np.linalg.norm(prediction-root[:, None], axis=-1) < 2.)
                & (errors.max(axis=1) <= 20.))
    output = np.where(observed[..., None], base, prediction)
    valid, completed = observed | accepted, accepted & ~observed
    if conflict_model is not None:
        # Retry only failed, conflicting stereo joints. Original observations stay intact.
        conflict = ~valid & visible.all(axis=1)
        if conflict.any():
            masked = confidence.copy()
            frames,joints = np.where(conflict)
            sides = np.argmin(confidence[frames,:,joints],axis=1)
            masked[frames,sides,joints] = 0.
            x,b,good,_ = stereo.features(pixels,masked,candidate_base=True)
            with torch.inference_mode():
                retry = torch.cat([conflict_model(a,bb) for a,bb in zip(torch.from_numpy(x).split(2048),torch.from_numpy(b).split(2048))]).numpy()
            alternative,accepted_retry,_ = complete_missing(stereo,retry,b,good,pixels,masked,single_hip_anchor,single_view_ray)
            use = conflict & accepted_retry
            output[use] = alternative[use];valid |= use;completed |= use
    return output, valid, completed


def retry_wrist_conflicts(stereo, model, joints, observed, pixels, confidence,
                         length_history, candidate_base=False, single_hip_anchor=False, single_view_ray=False):
    """Retry a stretched wrist estimate using past reliable forearm lengths.

    The 0.2 heatmap-score margin is an experimental selection rule, not an
    accuracy probability. Reliable geometry and non-wrist joints stay intact.
    The complete original joint is retained when a retry is accepted.
    Call after preserving the original output in raw_xyz.
    """
    for body_index, coco_index in ((4, 9), (5, 10)):
        joint, elbow = joints[coco_index], joints[coco_index-2]
        history = length_history[coco_index]
        if (joint['valid'] and elbow['valid'] and joint['reason'] == elbow['reason'] == 'observed'
                and confidence[0, :, [body_index, body_index-2]].min() >= .6):
            history.append(float(np.linalg.norm(np.array(joint['xyz'])-elbow['xyz'])))
            del history[:-30]
        if not elbow['valid'] or elbow['reason'] != 'observed' or len(history) < 5:
            continue
        reference = float(np.median(history))
        scores = confidence[0, :, body_index]
        if (observed[0, body_index] or not joint['valid'] or joint['reason'] != 'learned_body_completion'
                or scores.min() < .35 or scores.max() - scores.min() < .2):
            continue
        length = float(np.linalg.norm(np.array(joint['xyz'])-elbow['xyz']))
        if length <= 1.5*reference or length-reference <= .1:
            continue
        side = int(np.argmin(scores))
        # Retry one wrist at a time, so another suspect joint cannot change its features.
        masked = confidence.copy()
        masked[0, side, body_index] = 0.
        x, base, good, _ = stereo.features(pixels, masked, candidate_base=candidate_base)
        with torch.inference_mode():
            prediction = model(torch.from_numpy(x), torch.from_numpy(base)).numpy()
        alternative, valid, _ = complete_missing(stereo, prediction, base, good, pixels, masked,
                                                single_hip_anchor, single_view_ray, model)
        retry_length = float(np.linalg.norm(alternative[0, body_index]-elbow['xyz']))
        if valid[0, body_index] and .75*reference <= retry_length <= 1.25*reference:
            original = dict(joint)
            joint.update(xyz=alternative[0, body_index].astype(float).tolist(), estimated=True,
                         confidence=.2, reason='learned_body_completion',
                         correction='wrist_conflict_retry_v1',
                         completion_view_rejected=('left', 'right')[side],
                         forearm_reference_m=reference, uncorrected_joint=original)


def load_pose(path, stereo):
    path = Path(path)
    meta = read_json(path.with_suffix('.meta.json'))
    cal = stereo.calibration
    if meta.get('state') != 'complete' or meta.get('camera_ids') != cal['camera_ids']:
        raise ValueError('Complete pose data with matching camera order is required.')
    if meta.get('mirrored') is not False or meta.get('unit') != 'original_image_pixels':
        raise ValueError('Unmirrored original-image pixels are required; undo letterbox/cropping first.')
    if meta.get('calibration_id', cal['calibration_id']) != cal['calibration_id']:
        raise ValueError('Pose metadata calibration id mismatch.')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    previous = -1
    for row in rows:
        if row.get('schema') != 'pose2d/1.0' or row.get('joint_names') != list(COCO17) or len(row.get('views', [])) != 2:
            raise ValueError('Expected two COCO17 views per pose2d/1.0 frame.')
        timestamp = row.get('timestamp_ns')
        if type(timestamp) is not int or timestamp <= previous:
            raise ValueError('Pose timestamps must strictly increase.')
        previous = timestamp
        if row.get('model_id') != meta.get('model_id'):
            raise ValueError('Do not mix 2D model versions within a file.')
        for i, view in enumerate(row['views']):
            if (view.get('camera_id') != cal['camera_ids'][i] or view.get('image_size') != cal['image_size']
                    or len(view.get('joints', [])) != 17 or type(view.get('timestamp_ns')) is not int):
                raise ValueError('Camera identity, resolution, joint count or timestamp mismatch.')
            if view.get('person_count', 1) != 1:
                raise ValueError('This first model supports one identified person in each view.')
            for joint in view['joints']:
                confidence = joint.get('confidence')
                if type(confidence) not in (int, float) or not np.isfinite(confidence) or not 0 <= confidence <= 1:
                    raise ValueError('Invalid 2D confidence.')
                if joint.get('valid') and (np.shape(joint.get('xy')) != (2,) or not np.isfinite(joint['xy']).all()):
                    raise ValueError('Valid 2D coordinates must contain finite x,y.')
    if not rows or ('rows' in meta and meta['rows'] != len(rows)):
        raise ValueError('Empty or incomplete pose file.')
    return meta, rows


def infer(pose, calibration, checkpoint, output, tau=0., max_delta_ms=10.):
    if tau < 0 or max_delta_ms < 0:
        raise ValueError('Filter time and maximum pairing delta must be nonnegative.')
    stereo = Stereo(calibration)
    model, saved = load_model(checkpoint)
    if saved['calibration_hash'] != stereo.hash:
        raise ValueError('This model was trained for a different calibration; regenerate data and retrain.')
    meta, rows = load_pose(pose, stereo)
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; choose a new file.')
    output.parent.mkdir(parents=True, exist_ok=True)
    model_hash = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    report = dict(schema='skeleton3d_manifest/1.0', state='incomplete', method='stereo_geometry_mlp_completion',
                  source_pose2d=str(Path(pose).resolve()), source_2d_model=meta['model_id'],
                  calibration_id=stereo.calibration['calibration_id'], calibration_hash=stereo.hash,
                  model_id=f'pose3d_mlp:{model_hash[:16]}', checkpoint_sha256=model_hash,
                  frames=0, learned_joint_count=0, valid_body_joint_count=0, body_joint_count=len(rows)*12,
                  filter_tau_s=tau, max_delta_ms=max_delta_ms, temporal_alignment=meta.get('temporal_alignment'),
                  uncertainty='Completion confidence 0.2 is a provenance flag, not calibrated accuracy.',
                  experimental=True, independent_real_3d_ground_truth=False)
    report.update(provisional_calibration=saved.get('provisional_calibration', False),
                  single_view_ray=saved.get('single_view_ray',False),
                  reject_conflicting_view=saved.get('reject_conflicting_view',False),
                  calibration_accepted_for_training=saved.get('calibration_accepted_for_training'),
                  pose_ensemble=meta.get('pose_ensemble'),
                  source_roi_xywh=meta.get('roi_xywh'))
    write_json(output.with_suffix('.meta.json'), report)
    torch.set_num_threads(1)
    filt = CausalFilter(tau)
    timings, model_timings = [], []
    with torch.inference_mode():
        for _ in range(10):
            model(torch.zeros(1, 147), torch.zeros(1, 12, 3))
        with output.open('x', encoding='utf-8') as stream:
            for number, row in enumerate(rows):
                started = time.perf_counter()
                left, right = row['views']
                delta = (right['timestamp_ns']-left['timestamp_ns'])/1e6
                if abs(delta) > max_delta_ms:
                    joints = [dict(xyz=None, valid=False, confidence=0., estimated=False, reason='pair_time_mismatch') for _ in COCO17]
                else:
                    pixels = np.array([[[joint['xy'] if joint.get('valid') else [0., 0.]
                                         for joint in view['joints'][5:]] for view in row['views']]])
                    confidence = np.array([[[joint['confidence'] if joint.get('valid') else 0.
                                             for joint in view['joints'][5:]] for view in row['views']]])
                    x, base, observed, _ = stereo.features(pixels, confidence, candidate_base=saved.get('feature_policy')=='geometry_candidate_v1')
                    model_start = time.perf_counter()
                    prediction = model(torch.from_numpy(x), torch.from_numpy(base)).numpy()
                    model_timings.append((time.perf_counter()-model_start)*1000.)
                    hybrid, valid, completed = complete_missing(stereo, prediction, base, observed, pixels, confidence,
                                                               single_hip_anchor=saved.get('single_hip_anchor',False),
                                                               single_view_ray=saved.get('single_view_ray',False),
                                                               conflict_model=model if saved.get('reject_conflicting_view',False) else None)
                    joints = triangulate(left['joints'], right['joints'], stereo.calibration)
                    for body_index, coco_index in enumerate(BODY):
                        if completed[0, body_index]:
                            joints[coco_index] = dict(xyz=hybrid[0, body_index].astype(float).tolist(),
                                                     valid=True, estimated=True, confidence=.2, reason='learned_body_completion')
                            report['learned_joint_count'] += 1
                    report['valid_body_joint_count'] += sum(j['valid'] for j in joints[5:])
                filt.apply(joints, row['timestamp_ns'])
                packet = frame_packet(number, row['timestamp_ns'], joints,
                                      session_id=Path(meta['source_session']).name, calibration_id=report['calibration_id'],
                                      model_id=report['model_id'], coordinate_system=stereo.calibration['world_coordinate_system'],
                                      clock_id=meta['clock_id'])
                packet.update(source='pose3d_train', source_2d_model=meta['model_id'], source_seq=row.get('seq'),
                              pair_delta_ms=delta, source_frames=[view['frame_index'] for view in row['views']],
                              filter=dict(type='causal_exponential', tau_s=tau), experimental=True)
                if row.get('temporal_alignment'):
                    packet['temporal_alignment'] = row['temporal_alignment']
                timings.append((time.perf_counter()-started)*1000.)
                stream.write(json.dumps(packet, ensure_ascii=False, allow_nan=False)+'\n')
                report['frames'] += 1
    report.update(state='complete', valid_body_ratio=report['valid_body_joint_count']/report['body_joint_count'],
                  pipeline_compute_median_ms=float(np.median(timings)), pipeline_compute_p95_ms=float(np.percentile(timings, 95)),
                  model_only_median_ms=float(np.median(model_timings)) if model_timings else None,
                  model_only_p95_ms=float(np.percentile(model_timings, 95)) if model_timings else None,
                  latency_excludes='File read/write, network, 2D inference and BVH export.')
    write_json(output.with_suffix('.meta.json'), report)
    print(json.dumps(report, ensure_ascii=False), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pose', required=True)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--tau', type=float, default=0.)
    args = parser.parse_args()
    infer(args.pose, args.calibration, args.checkpoint, args.output, args.tau)
