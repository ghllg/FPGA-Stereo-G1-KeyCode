"""Offline articulated fitting to BOTH image observations, with explicit estimated provenance."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from .calibration import load_calibration
from .io import read_json, write_json, frame_packet
from .reconstruction import triangulate

JOINTS = (0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
# Shoulder-to-hip distance is NOT a rigid bone: the spine bends during a squat.
# Only limb segments and soft pelvis/shoulder widths receive length priors.
BONES = ((5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
         (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))


def fit_segment(initial, pixels, confidence, times, calibration, lengths, max_nfev=30):
    """Fixed cameras, soft bone lengths, robust image residuals and bounded acceleration."""
    n, joints, _ = initial.shape
    k = [np.array(a) for a in calibration['K']]; d = [np.array(a) for a in calibration['distortion']]
    r = np.array(calibration['R_cam0_to_cam1']); t = np.array(calibration['t_cam0_to_cam1_m'])
    rotation = cv2.Rodrigues(r)[0]
    edges = np.array([(JOINTS.index(a), JOINTS.index(b)) for a, b in BONES])
    pixel_mask = np.isfinite(pixels).all(axis=-1)
    target_pixels = np.nan_to_num(pixels)
    weight = np.sqrt(confidence)*pixel_mask
    dt = np.diff(times)
    if n > 2:
        acceleration_dt = (dt[1:]+dt[:-1])*.5

    def project(x):
        points = x.reshape(-1, 3)
        return np.array([cv2.projectPoints(points, rv, tv, ki, di)[0].reshape(n, joints, 2)
                         for rv, tv, ki, di in zip([np.zeros(3), rotation], [np.zeros(3), t], k, d)])

    def residual(flat):
        x = flat.reshape(n, joints, 3)
        image = ((project(x)-target_pixels)*weight[:, :, :, None]/3.).ravel()
        bone = ((np.linalg.norm(x[:, edges[:, 0]]-x[:, edges[:, 1]], axis=-1)-lengths)/.02).ravel()
        if n > 2:
            velocity = np.diff(x, axis=0)/dt[:, None, None]
            acceleration = np.diff(velocity, axis=0)/acceleration_dt[:, None, None]
            temporal = (acceleration/20.).ravel()
        else: temporal = np.empty(0)
        return np.r_[image, bone, temporal]

    image_size = 2*n*joints*2; bone_size = n*len(edges)
    total = image_size+bone_size+max(0, n-2)*joints*3
    sparsity = lil_matrix((total, n*joints*3), dtype=np.int8)
    columns = np.arange(n*joints*3).reshape(n, joints, 3)
    for cam in range(2):
        for f in range(n):
            for j in range(joints):
                for axis in range(2): sparsity[((cam*n+f)*joints+j)*2+axis, columns[f, j]] = 1
    for f in range(n):
        for edge, (a, b) in enumerate(edges):
            sparsity[image_size+f*len(edges)+edge, np.r_[columns[f, a], columns[f, b]]] = 1
    for f in range(max(0, n-2)):
        for j in range(joints):
            for axis in range(3):
                sparsity[image_size+bone_size+(f*joints+j)*3+axis, columns[f:f+3, j, axis]] = 1
    result = least_squares(residual, initial.ravel(), jac_sparsity=sparsity.tocsr(), x_scale='jac',
                           loss='soft_l1', f_scale=1., max_nfev=max_nfev,
                           ftol=1e-4, xtol=1e-5, gtol=1e-4)
    fitted = result.x.reshape(n, joints, 3)
    errors = np.linalg.norm(project(fitted)-target_pixels, axis=-1)
    return fitted, errors, dict(converged=bool(result.success), evaluations=int(result.nfev),
                               status=int(result.status), cost=float(result.cost), message=str(result.message))


def recover(pose_file, calibration_file, output, max_input_reprojection_px=20., max_gap_s=.15):
    source, output = Path(pose_file), Path(output)
    if output.exists(): raise ValueError('恢复骨架输出已存在，请选择新文件。')
    if not 5 <= max_input_reprojection_px <= 25 or not 0 <= max_gap_s <= .5:
        raise ValueError('恢复误差上限应为 5–25 px；短缺口上限应为 0–0.5 秒。')
    meta = read_json(source.with_suffix('.meta.json')); cal = load_calibration(calibration_file)
    if meta.get('state') != 'complete' or meta['camera_ids'] != cal['camera_ids']:
        raise ValueError('需要已完整保存且相机身份一致的二维点。')
    if not meta.get('temporal_alignment'): raise ValueError('离线约束恢复需要先核对/校正两路内容时间差。')
    rows = [json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]
    if len(rows) < 3: raise ValueError('至少需要三帧。')
    if any(len(row['views']) != 2 or any(v['camera_id'] != cal['camera_ids'][cam] or
           v['image_size'] != cal['image_size'] for cam, v in enumerate(row['views'])) for row in rows):
        raise ValueError('相机身份、顺序或分辨率与标定不一致。')
    times = np.array([(row['timestamp_ns']-rows[0]['timestamp_ns'])/1e9 for row in rows])
    if np.any(np.diff(times) <= 0): raise ValueError('二维点时间戳必须严格递增。')
    optical = dict(cal, world_from_cam0=None)
    packets = [triangulate(row['views'][0]['joints'], row['views'][1]['joints'], optical,
                           max_reprojection_px=max_input_reprojection_px) for row in rows]
    raw = np.array([[j['xyz'] if j['valid'] else [np.nan]*3 for j in joints] for joints in packets])[:, JOINTS]
    filled = raw.copy(); gap_filled = np.zeros(raw.shape[:2], bool)
    for j in range(len(JOINTS)):
        valid = np.flatnonzero(np.isfinite(raw[:, j]).all(axis=1))
        for a, b in zip(valid[:-1], valid[1:]):
            if b > a+1 and times[b]-times[a] <= max_gap_s:
                alpha = (times[a+1:b]-times[a])/(times[b]-times[a])
                filled[a+1:b, j] = raw[a, j]+alpha[:, None]*(raw[b, j]-raw[a, j]); gap_filled[a+1:b, j] = True
    lengths = []
    for a, b in BONES:
        distance = np.linalg.norm(raw[:, JOINTS.index(a)]-raw[:, JOINTS.index(b)], axis=1)
        finite = distance[np.isfinite(distance)]
        if len(finite) < 10: raise ValueError('可靠骨长观测不足，不能进行全身约束恢复。')
        lengths.append(float(np.median(finite)))
    lengths = np.array(lengths)
    if np.any(lengths < .05) or np.any(lengths > .85): raise ValueError('人体骨长异常，拒绝恢复。')
    pixels = np.array([[[v['joints'][j]['xy'] if v['joints'][j]['valid'] else [np.nan]*2
                         for j in JOINTS] for v in row['views']] for row in rows]).transpose(1, 0, 2, 3)
    conf = np.array([[[v['joints'][j]['confidence'] if v['joints'][j]['valid'] else 0.
                       for j in JOINTS] for v in row['views']] for row in rows]).transpose(1, 0, 2)
    # Do not allow rejected input observations to pull an interpolated seed toward an outlier.
    conf *= np.isfinite(raw).all(axis=2)[None, :, :]
    usable = np.isfinite(filled).all(axis=(1, 2))
    groups = []; group = []
    for i, ok in enumerate(usable):
        if not ok or (group and times[i]-times[group[-1]] > max_gap_s):
            if len(group) >= 3: groups.append(group)
            group = []
        if ok: group.append(i)
    if len(group) >= 3: groups.append(group)
    if not groups: raise ValueError('没有可恢复的全身连续区间；不补造长时间缺失动作。')
    world = np.array(cal['world_from_cam0']) if cal.get('world_from_cam0') is not None else np.eye(4)
    fitted_world = np.full_like(filled, np.nan); error_all = np.full((2, len(rows), len(JOINTS)), np.nan)
    segments = []
    for group in groups:
        print(f'离线骨架约束：处理 {len(group)} 帧；固定双目参数、拟合两路像素、骨长和时序…', flush=True)
        fitted, errors, status = fit_segment(filled[group], pixels[:, group], conf[:, group], times[group], cal, lengths)
        if not np.isfinite(fitted).all(): raise ValueError('离线拟合出现非有限值。')
        camera1 = fitted@np.array(cal['R_cam0_to_cam1']).T+np.array(cal['t_cam0_to_cam1_m'])
        depth_ok = (fitted[:, :, 2] > 0) & (camera1[:, :, 2] > 0) & (np.linalg.norm(fitted, axis=2) < 20)
        observation_ok = (np.max(errors, axis=0) <= max_input_reprojection_px) | gap_filled[group]
        good = depth_ok & observation_ok
        if not status['converged']:
            # Finite iteration-limited candidates still require the same physical and observation gates.
            status['acceptance'] = 'iteration_limited_candidate_with_per_joint_geometry_gates'
        fitted_world[group] = fitted@world[:3, :3].T+world[:3, 3]
        for index, row_index in enumerate(group): fitted_world[row_index, ~good[index]] = np.nan
        error_all[:, group] = errors
        segments.append(dict(first_seq=group[0], last_seq=group[-1], **status))
    report = dict(schema='skeleton3d_manifest/1.0', state='incomplete', frames=0, source_pose2d=str(source.resolve()),
                  calibration_id=cal['calibration_id'], model_id=meta['model_id'], temporal_alignment=meta['temporal_alignment'],
                  method='offline_articulated_soft_l1', estimated=True, max_input_reprojection_px=max_input_reprojection_px,
                  max_gap_s=max_gap_s, pixel_sigma_px=3., bone_sigma_m=.02, acceleration_sigma_m_s2=20.,
                  bone_pairs=[list(b) for b in BONES], median_bone_lengths_m=lengths.tolist(), fit_segments=segments,
                  note='Estimated offline poses fitted to both camera observations. No changes to camera calibration. Short bounded gaps only; no long-gap extrapolation. Not measured ground truth.')
    if meta.get('pose_ensemble'): report['pose_ensemble'] = meta['pose_ensemble']
    output.parent.mkdir(parents=True, exist_ok=True); write_json(output.with_suffix('.meta.json'), report)
    counts = [0]*17
    with output.open('x', encoding='utf-8') as dst:
        for i, row in enumerate(rows):
            joints = copy.deepcopy(packets[i])
            for j, joint in enumerate(joints):
                if joint['valid']:
                    joint['xyz'] = (world@np.r_[joint['xyz'], 1.])[:3].tolist()
            for slot, coco in enumerate(JOINTS):
                point = fitted_world[i, slot]
                if not np.isfinite(point).all():
                    joints[coco] = dict(xyz=None, valid=False, estimated=False, confidence=0., reason='offline_fit_unavailable')
                    continue
                errors = error_all[:, i, slot]
                joints[coco].update(xyz=point.tolist(), valid=True, estimated=True,
                    reason='offline_articulated_short_gap' if gap_filled[i, slot] else 'offline_articulated_fit',
                    confidence=float(np.min(conf[:, i, slot])*np.exp(-np.max(errors)/15.)),
                    input_reprojection_px=packets[i][coco].get('reprojection_px'),
                    reprojection_px=errors.tolist(), source_strict_valid=bool(packets[i][coco]['valid'] and
                    max(packets[i][coco].get('reprojection_px') or [float('inf')]) <= 5.))
            packet = frame_packet(i, row['timestamp_ns'], joints, session_id=Path(meta['source_session']).name,
                calibration_id=cal['calibration_id'], model_id=meta['model_id'], clock_id=meta['clock_id'],
                coordinate_system=cal.get('world_coordinate_system') or cal['coordinate_system'])
            packet.update(temporal_alignment=row['temporal_alignment'],
                          source_frames=[v['frame_index'] for v in row['views']], filter=dict(type=report['method']))
            dst.write(json.dumps(packet, ensure_ascii=False, allow_nan=False)+'\n')
            for j, joint in enumerate(joints): counts[j] += int(joint['valid'])
    report.update(state='complete', frames=len(rows), valid_joint_counts=counts,
                  valid_joint_ratios=[c/len(rows) for c in counts]); write_json(output.with_suffix('.meta.json'), report)
    print(f'离线估计骨架已保存：{output}；{len(rows)} 帧；请与严格筛选版及原视频对照。', flush=True)
    return report
