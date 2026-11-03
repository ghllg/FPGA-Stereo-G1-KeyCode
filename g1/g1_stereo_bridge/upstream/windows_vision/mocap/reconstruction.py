from __future__ import annotations
import json
from pathlib import Path
import cv2
import numpy as np
from .io import COCO17, read_json, write_json, frame_packet
from .calibration import load_calibration


def triangulate(a, b, calibration, min_confidence=.35, max_reprojection_px=5., min_parallax_deg=1.):
    k = [np.array(x) for x in calibration['K']]
    d = [np.array(x) for x in calibration['distortion']]
    r = np.array(calibration['R_cam0_to_cam1']); t = np.array(calibration['t_cam0_to_cam1_m'])
    p0 = np.column_stack([np.eye(3), np.zeros(3)])
    p1 = np.column_stack([r, t])
    result = []
    for left, right in zip(a, b):
        item = dict(xyz=None, valid=False, estimated=False, confidence=0., reason='missing_2d', reprojection_px=None)
        result.append(item)
        if not left.get('valid') or not right.get('valid') or min(left['confidence'], right['confidence']) < min_confidence: continue
        xy = [np.array(x['xy'], np.float64).reshape(1, 1, 2) for x in (left, right)]
        if not all(np.isfinite(x).all() for x in xy): continue
        und = [cv2.undistortPoints(x, ki, di).reshape(2) for x, ki, di in zip(xy, k, d)]
        ray0 = np.r_[und[0], 1.]; ray1 = r.T @ np.r_[und[1], 1.]
        angle = np.degrees(np.arccos(np.clip(np.dot(ray0, ray1)/np.linalg.norm(ray0)/np.linalg.norm(ray1), -1, 1)))
        if angle < min_parallax_deg: item['reason'] = 'low_parallax'; continue
        homogeneous = cv2.triangulatePoints(p0, p1, und[0].reshape(2, 1), und[1].reshape(2, 1)).ravel()
        if abs(homogeneous[3]) < 1e-9: item['reason'] = 'point_at_infinity'; continue
        point = homogeneous[:3] / homogeneous[3]
        if not np.isfinite(point).all() or point[2] <= 0 or (r @ point + t)[2] <= 0 or np.linalg.norm(point) > 20:
            item['reason'] = 'invalid_depth'; continue
        errors = []
        for ki, di, ri, ti, pixel in zip(k, d, [np.zeros(3), cv2.Rodrigues(r)[0]], [np.zeros(3), t], xy):
            proj, _ = cv2.projectPoints(point.reshape(1, 3), ri, ti, ki, di)
            errors.append(float(np.linalg.norm(proj - pixel)))
        item['reprojection_px'] = errors
        if max(errors) > max_reprojection_px: item['reason'] = 'reprojection_error'; continue
        world = calibration.get('world_from_cam0')
        if world is not None: point = (np.array(world) @ np.r_[point, 1.])[:3]
        estimated = bool(left.get('estimated') or right.get('estimated'))
        item.update(xyz=point.tolist(), valid=True, confidence=float(min(left['confidence'], right['confidence'])),
                    estimated=estimated, reason='offline_temporal_interpolation' if estimated else 'observed',
                    parallax_deg=float(angle))
        if left.get('pose_candidate_source') or right.get('pose_candidate_source'):
            item['pose_candidate_sources'] = [left.get('pose_candidate_source'), right.get('pose_candidate_source')]
    return result


class CausalFilter:
    def __init__(self, tau=.04):
        self.tau = tau; self.previous = [None]*17; self.times = [None]*17

    def apply(self, joints, timestamp_ns):
        for i, joint in enumerate(joints):
            if not joint['valid']:
                self.previous[i] = self.times[i] = None
                continue
            point = np.array(joint['xyz']); joint['raw_xyz'] = point.tolist()
            dt = (timestamp_ns - self.times[i])/1e9 if self.times[i] is not None else None
            if self.tau > 0 and dt is not None and 0 < dt < .2:
                alpha = 1 - np.exp(-dt/self.tau)
                point = self.previous[i] + alpha * (point - self.previous[i])
            self.previous[i], self.times[i] = point, timestamp_ns
            joint['xyz'] = point.tolist()
        return joints


def reconstruct(pose_file, calibration_file, output, max_delta_ms=10., max_reprojection_px=5., tau=.04):
    pose_file, output = Path(pose_file), Path(output)
    meta = read_json(pose_file.with_suffix('.meta.json'))
    if meta.get('state') != 'complete': raise ValueError('二维关键点文件未完整生成。')
    calibration = load_calibration(calibration_file)
    if meta['camera_ids'] != calibration['camera_ids']: raise ValueError('二维点相机顺序与标定不一致。')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): raise ValueError('三维骨架文件已存在，请选择新文件。')
    report = dict(schema='skeleton3d_manifest/1.0', state='incomplete', frames=0,
                  source_pose2d=str(pose_file.resolve()), calibration_id=calibration['calibration_id'],
                  model_id=meta['model_id'], filter_tau_s=tau, max_reprojection_px=max_reprojection_px,
                  max_delta_ms=max_delta_ms, valid_joint_counts=[0]*17)
    if meta.get('temporal_alignment'):
        report['temporal_alignment'] = meta['temporal_alignment']
    write_json(output.with_suffix('.meta.json'), report)
    filt = CausalFilter(tau); previous = -1; errors = []; frame_count = 0
    try:
        with pose_file.open(encoding='utf-8') as src, output.open('x', encoding='utf-8') as dst:
            for line in src:
                row = json.loads(line)
                if row.get('joint_names') != list(COCO17) or len(row['views']) != 2:
                    raise ValueError('二维关键点格式必须为两路 COCO17。')
                timestamp = row['timestamp_ns']
                if timestamp <= previous: raise ValueError('二维点时间戳必须严格递增。')
                previous = timestamp
                for i, view in enumerate(row['views']):
                    if view['camera_id'] != calibration['camera_ids'][i] or view['image_size'] != calibration['image_size'] or len(view['joints']) != 17:
                        raise ValueError('二维点相机、分辨率或关节数与标定不一致。')
                delta = (row['views'][1]['timestamp_ns']-row['views'][0]['timestamp_ns'])/1e6
                if abs(delta) > max_delta_ms:
                    joints = [dict(xyz=None, valid=False, estimated=False, confidence=0., reason='pair_time_mismatch') for _ in range(17)]
                else:
                    joints = triangulate(row['views'][0]['joints'], row['views'][1]['joints'], calibration,
                                         max_reprojection_px=max_reprojection_px)
                filt.apply(joints, timestamp)
                packet = frame_packet(frame_count, timestamp, joints,
                        session_id=Path(meta['source_session']).name, calibration_id=calibration['calibration_id'],
                        model_id=meta['model_id'], coordinate_system=calibration.get('world_coordinate_system') or calibration['coordinate_system'],
                        clock_id=meta['clock_id'])
                packet.update(pair_delta_ms=delta, source_frames=[v['frame_index'] for v in row['views']],
                              filter=dict(type='causal_exponential', tau_s=tau))
                if row.get('temporal_alignment'):
                    packet['temporal_alignment'] = row['temporal_alignment']
                dst.write(json.dumps(packet, ensure_ascii=False, allow_nan=False)+'\n')
                frame_count += 1
                for i, joint in enumerate(joints):
                    if joint['valid']:
                        report['valid_joint_counts'][i] += 1
                        errors.extend(joint['reprojection_px'])
        if not frame_count: raise ValueError('二维关键点文件为空。')
        report.update(state='complete', frames=frame_count,
                      valid_joint_ratios=[n/frame_count for n in report['valid_joint_counts']],
                      reprojection_median_px=float(np.median(errors)) if errors else None)
    finally:
        report['frames'] = frame_count
        write_json(output.with_suffix('.meta.json'), report)
    print(f'三维骨架已保存：{output}；{frame_count} 帧', flush=True)
