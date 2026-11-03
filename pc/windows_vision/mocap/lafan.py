"""22-joint LAFAN1-topology BVH with fixed bones, local rotations and honest provenance."""
from __future__ import annotations
import json
import warnings
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .io import write_json
from shared.skeleton_protocol import decode

NAMES = ('Hips', 'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToe',
         'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToe',
         'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
         'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
         'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand')
PARENTS = np.array([-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20])
CORE = (5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)
WORLD_TO_BVH = np.array([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]])


def normalize(x):
    length = np.linalg.norm(x)
    if length < 1e-7: raise ValueError('骨架退化：骨长或身体方向为零。')
    return x / length


def body_basis(points):
    lateral = normalize(points[11]-points[12])
    up = normalize((points[5]+points[6]-points[11]-points[12]) * .5)
    front = normalize(np.cross(lateral, up))
    up = normalize(np.cross(front, lateral))
    return np.column_stack([lateral, up, front])


def extend_coco(points, torso_length=None):
    """Return Y-up metre targets. Spine/head/toes use explicit geometric assumptions."""
    p = np.array(points) @ WORLD_TO_BVH.T
    basis = body_basis(p); up, front = basis[:, 1], basis[:, 2]
    hips = (p[11]+p[12])*.5; shoulders = (p[5]+p[6])*.5
    torso = np.linalg.norm(shoulders-hips)
    head = p[0] + front * -.06 if np.isfinite(p[0]).all() else shoulders + up * torso * .55
    q = np.zeros((22, 3))
    q[0] = hips
    q[1:4] = p[[11, 13, 15]]; q[5:8] = p[[12, 14, 16]]
    # Toe targets are assumed, not measured COCO ankle-to-toe contact.
    q[4] = q[3] + front * np.linalg.norm(q[3]-q[2]) * .45
    q[8] = q[7] + front * np.linalg.norm(q[7]-q[6]) * .45
    q[9] = hips + (shoulders-hips)*.33
    q[10] = hips + (shoulders-hips)*.66
    q[11] = shoulders
    if torso_length is not None and torso < torso_length:
        # A bent torso must bend the estimated spine, not shrink fixed BVH bones.
        # Three equal links with symmetric bend preserve hip and shoulder targets.
        link = torso_length/3.
        cosine = np.clip((torso/link-1)/2, 0., 1.)
        sine = np.sqrt(1-cosine*cosine)
        direction = normalize(shoulders-hips)
        bend = normalize(np.cross(basis[:, 0], direction))
        q[9] = hips + direction*link*cosine - bend*link*sine
        q[10] = q[9] + direction*link
        lateral = basis[:, 0]
        spine_up = normalize(q[9]-hips)
        spine_front = normalize(np.cross(lateral, spine_up))
        basis = np.column_stack([lateral, normalize(np.cross(spine_front, lateral)), spine_front])
    q[12] = shoulders + (head-shoulders)*.45
    q[13] = head
    q[14] = shoulders + (p[5]-shoulders)*.35
    q[15:18] = p[[5, 7, 9]]
    q[18] = shoulders + (p[6]-shoulders)*.35
    q[19:22] = p[[6, 8, 10]]
    return q, basis


def rotation_between(a, b):
    a, b = normalize(a), normalize(b)
    cross = np.cross(a, b); dot = np.clip(np.dot(a, b), -1, 1)
    if dot > .999999: return np.eye(3)
    if dot < -.999999:
        axis = normalize(np.cross(a, np.eye(3)[np.argmin(abs(a))]))
        return Rotation.from_rotvec(axis*np.pi).as_matrix()
    return Rotation.from_quat(np.r_[cross, 1+dot]/np.linalg.norm(np.r_[cross, 1+dot])).as_matrix()


def fit_rotations(targets, offsets, root_basis):
    global_r = np.tile(np.eye(3), (22, 1, 1)); local_r = global_r.copy()
    fk = np.zeros((22, 3)); fk[0] = targets[0]
    global_r[0] = local_r[0] = root_basis
    for joint in range(1, 22):
        parent = PARENTS[joint]
        fk[joint] = fk[parent] + global_r[parent] @ offsets[joint]
        children = np.flatnonzero(PARENTS == joint)
        if len(children):
            src = np.array([offsets[c] for c in children])
            dst = np.array([global_r[parent].T @ (targets[c]-targets[joint]) for c in children])
            if len(children) == 1:
                local_r[joint] = rotation_between(src[0], dst[0])
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    local_r[joint] = Rotation.align_vectors(dst/np.linalg.norm(dst, axis=1)[:, None],
                                                           src/np.linalg.norm(src, axis=1)[:, None])[0].as_matrix()
        global_r[joint] = global_r[parent] @ local_r[joint]
    return local_r, fk


def offsets_from_targets(targets):
    lengths = np.median(np.linalg.norm(targets[:, 1:]-targets[:, PARENTS[1:]], axis=2), axis=0)
    if np.any(lengths < .005) or np.any(lengths > 1.5):
        raise ValueError('重建骨长异常，检查尺度、左右身份、关键点与标定。')
    directions = np.zeros((22, 3))
    directions[1] = [1, 0, 0]; directions[5] = [-1, 0, 0]
    directions[[2, 3, 6, 7]] = [0, -1, 0]
    directions[[4, 8]] = [0, 0, 1]
    directions[9:14] = [0, 1, 0]
    directions[14:18] = [1, 0, 0]; directions[18:22] = [-1, 0, 0]
    directions[1:] *= lengths[:, None]
    return directions


def write_bvh(path, offsets, roots, rotations, fps):
    lines = ['HIERARCHY']
    def node(j, level):
        indent = '  '*level
        lines.append(indent + ('ROOT ' if j == 0 else 'JOINT ') + NAMES[j])
        lines.append(indent+'{')
        lines.append(indent+'  OFFSET '+' '.join(f'{v*100:.8f}' for v in offsets[j]))
        lines.append(indent+'  '+('CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation' if j == 0
                                  else 'CHANNELS 3 Zrotation Yrotation Xrotation'))
        children = np.flatnonzero(PARENTS == j)
        for child in children: node(int(child), level+1)
        if not len(children):
            lines.extend([indent+'  End Site', indent+'  {', indent+'    OFFSET 0 0 0', indent+'  }'])
        lines.append(indent+'}')
    node(0, 0)
    lines.extend(['MOTION', f'Frames: {len(roots)}', f'Frame Time: {1/fps:.9f}'])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        # Uppercase ZYX is intrinsic; matches Ubisoft qZ*qY*qX reader.
        euler = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_euler('ZYX', degrees=True).reshape(-1, 22, 3)
    euler = np.rad2deg(np.unwrap(np.deg2rad(euler), axis=0))
    for root, angles in zip(roots, euler):
        lines.append(' '.join(f'{x:.8f}' for x in np.r_[root*100, angles.ravel()]))
    Path(path).write_text('\n'.join(lines)+'\n', encoding='ascii')


def export_lafan(skeleton_file, output_dir, fps=30., max_gap_s=.15, max_sample_interval_s=None, single_frame_gap_s=0.):
    if not 1 <= fps <= 240 or not 0 <= max_gap_s <= 1: raise ValueError('帧率或缺口阈值不合理。')
    sample_limit = max_gap_s if max_sample_interval_s is None else max_sample_interval_s
    if not np.isfinite(sample_limit) or not 0 < sample_limit <= 1:
        raise ValueError('采样间隔阈值不合理。')
    if not np.isfinite(single_frame_gap_s) or not 0 <= single_frame_gap_s <= 1:
        raise ValueError('单帧缺口阈值不合理。')
    source = Path(skeleton_file)
    source_meta = {}
    if source.with_suffix('.meta.json').exists():
        from .io import read_json
        source_meta = read_json(source.with_suffix('.meta.json'))
        if source_meta.get('state') != 'complete':
            raise ValueError('三维骨架未完整生成。')
    packets = [decode(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(packets) < 2: raise ValueError('至少需要两帧骨架。')
    if any(p['coordinate_system'] != 'world_right_handed_X_forward_Y_left_Z_up' for p in packets):
        raise ValueError('BVH 导出需要地面世界坐标。请先运行 world 完成地面标定。')
    reference = packets[0]
    if any(any(p[k] != reference[k] for k in ('session_id', 'clock_id', 'calibration_id', 'model_id')) for p in packets):
        raise ValueError('不能混合不同会话、时钟、标定或模型的骨架。')
    origin = packets[0]['timestamp_ns']
    times = np.array([(p['timestamp_ns']-origin)/1e9 for p in packets])
    if np.any(np.diff(times) <= 0): raise ValueError('时间戳必须严格递增。')
    raw = np.full((len(packets), 17, 3), np.nan)
    for i, packet in enumerate(packets):
        for j, joint in enumerate(packet['joints']):
            if joint['valid']: raw[i, j] = joint['xyz']
    # Interpolate at source times only over short, bounded missing runs. Never extrapolate.
    filled = raw.copy(); was_filled = np.zeros(raw.shape[:2], bool)
    single_frame_filled = np.zeros(raw.shape[:2], bool)
    for j in range(17):
        valid = np.flatnonzero(np.isfinite(raw[:, j]).all(axis=1))
        for a, b in zip(valid[:-1], valid[1:]):
            single = (b == a+2 and times[b]-times[a] <= single_frame_gap_s
                      and np.max(np.diff(times[a:b+1])) <= sample_limit)
            if b > a+1 and (times[b]-times[a] <= max_gap_s or single):
                fractions = (times[a+1:b]-times[a])/(times[b]-times[a])
                filled[a+1:b, j] = raw[a, j] + fractions[:, None]*(raw[b, j]-raw[a, j])
                was_filled[a+1:b, j] = True
                if single: single_frame_filled[a+1:b, j] = True
    samples = np.arange(int(np.floor(times[-1]*fps))+1)/fps
    torso_distances = np.linalg.norm((filled[:, 5]+filled[:, 6]-filled[:, 11]-filled[:, 12])*.5, axis=1)
    finite_torsos = torso_distances[np.isfinite(torso_distances)]
    torso_length = float(np.percentile(finite_torsos, 90)) if len(finite_torsos) else None
    resampled = []; provenance = []; chunks = []; chunk = []
    for sample_index, t in enumerate(samples):
        hi = int(np.searchsorted(times, t)); hi = min(hi, len(times)-1)
        if abs(times[hi]-t) < 1e-8: lo = hi; alpha = 0.
        else: lo = max(0, hi-1); alpha = (t-times[lo])/(times[hi]-times[lo]) if hi != lo else 0.
        usable = np.isfinite(filled[[lo, hi]][:, CORE]).all() and (lo == hi or times[hi]-times[lo] <= sample_limit)
        if usable:
            p = filled[lo] if lo == hi else filled[lo]*(1-alpha)+filled[hi]*alpha
            try: q, basis = extend_coco(p, torso_length)
            except ValueError: usable = False
        if not usable:
            if chunk: chunks.append(chunk); chunk = []
            continue
        index = len(resampled); resampled.append((q, basis))
        provenance.append(dict(timestamp_ns=origin+round(t*1e9), sample_index=sample_index,
                               source_seq=[packets[lo]['seq'], packets[hi]['seq']], interpolation_alpha=float(alpha),
                               source_estimated_coco_indices=[j for j in range(17) if packets[lo]['joints'][j]['estimated'] or packets[hi]['joints'][j]['estimated']],
                               source_temporal_alignment=[packets[lo].get('temporal_alignment'), packets[hi].get('temporal_alignment')],
                               source_model_candidates=[joint.get('pose_candidate_sources') for joint in packets[lo]['joints']],
                               gap_filled_coco_indices=np.flatnonzero(was_filled[lo] | was_filled[hi]).tolist(),
                               single_frame_filled_coco_indices=np.flatnonzero(single_frame_filled[lo] | single_frame_filled[hi]).tolist()))
        chunk.append(index)
    if chunk: chunks.append(chunk)
    chunks = [chunk for chunk in chunks if len(chunk) >= 2]
    if not chunks: raise ValueError('没有可导出的连续全身片段。检查遮挡、标定和识别质量。')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()): raise ValueError('BVH 输出目录非空，请使用新目录。')
    output.mkdir(parents=True, exist_ok=True)
    all_targets = np.array([resampled[i][0] for chunk in chunks for i in chunk])
    offsets = offsets_from_targets(all_targets)
    exported_frames = sum(len(chunk) for chunk in chunks)
    dropped_samples = sorted(set(range(len(samples))) - {provenance[i]['sample_index'] for chunk in chunks for i in chunk})
    report = dict(schema='lafan1_export/1.0', state='incomplete', source_skeleton=str(source.resolve()),
                  topology='LAFAN1 22 joints; subject-specific fixed bone lengths', unit='cm', up_axis='Y',
                  rotation_channels='intrinsic ZYX degrees', joint_names=list(NAMES), parents=PARENTS.tolist(),
                  fps=fps, max_gap_s=max_gap_s, max_sample_interval_s=sample_limit,
                  single_frame_gap_s=single_frame_gap_s, single_frame_filled_joint_count=int(single_frame_filled.sum()),
                  world_to_bvh_rotation=WORLD_TO_BVH.tolist(),
                  calibration_id=reference['calibration_id'], model_id=reference['model_id'],
                  source_reconstruction_method=source_meta.get('method', 'strict_triangulation'),
                  temporal_alignment=source_meta.get('temporal_alignment'),
                  clock_id=reference['clock_id'], input_frames=len(packets), segments=[],
                  source_duration_s=float(times[-1]), resampling_grid_frames=len(samples),
                  exported_frames=exported_frames, exported_duration_s=exported_frames/fps,
                  coverage_ratio=exported_frames/len(samples), dropped_sample_indices=dropped_samples,
                  quality_status='low_coverage' if exported_frames/len(samples) < .9 else 'coverage_ok_requires_visual_review',
                  estimated_spine_method='symmetric_three_link_bend', estimated_torso_length_m=torso_length,
                  estimated_joints=['Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head', 'LeftShoulder', 'RightShoulder', 'LeftToe', 'RightToe'],
                  assumptions=['Spine/clavicle/head/toe positions use geometric estimates, not direct measurements.',
                               'Spine bend uses a symmetric three-link geometric estimate to preserve torso length; not measured vertebral motion.',
                               'Axial twist is unobservable: minimal-swing rotations; leaf wrist/head/toe local rotation defaults to identity.',
                               'BVH bone lengths are constant median lengths; FK positions can differ from measured targets.',
                               'Short gaps use offline interpolation, not a causal live estimate; long gaps split clips.',
                               'This is generated LAFAN1-topology motion, not original Ubisoft ground truth or a trained model.'])
    write_json(output/'export_report.json', report)
    try:
        for number, chunk in enumerate(chunks, 1):
            targets = np.array([resampled[i][0] for i in chunk])
            rotations = []; fk = []
            for i in chunk:
                rot, pos = fit_rotations(*[resampled[i][0], offsets, resampled[i][1]])
                rotations.append(rot); fk.append(pos)
            rotations, fk = np.array(rotations), np.array(fk)
            name = f'motion_{number:03d}'
            write_bvh(output/(name+'.bvh'), offsets, targets[:, 0], rotations, fps)
            xyzw = Rotation.from_matrix(rotations.reshape(-1, 3, 3)).as_quat().reshape(-1, 22, 4)
            quats = xyzw[:, :, [3, 0, 1, 2]]
            for f in range(1, len(quats)):
                flip = np.sum(quats[f]*quats[f-1], axis=1) < 0
                quats[f, flip] *= -1
            timestamps = np.array([provenance[i]['timestamp_ns'] for i in chunk], dtype=np.int64)
            np.savez_compressed(output/(name+'.npz'), root_positions_cm=targets[:, 0]*100,
                                local_quaternions_wxyz=quats, offsets_cm=offsets*100,
                                global_positions_cm=fk*100, target_positions_cm=targets*100,
                                joint_names=np.array(NAMES), parents=PARENTS, fps=np.array(fps), timestamps_ns=timestamps)
            fit_error = np.linalg.norm(fk-targets, axis=2)
            sidecar = dict(frames=[provenance[i] for i in chunk],
                           fk_fit_median_m=float(np.median(fit_error)), fk_fit_p95_m=float(np.percentile(fit_error, 95)))
            write_json(output/(name+'.quality.json'), sidecar)
            report['segments'].append(dict(file=name+'.bvh', frames=len(chunk), start_ns=int(timestamps[0]),
                                          end_ns=int(timestamps[-1]), fk_fit_p95_m=sidecar['fk_fit_p95_m'],
                                          fit_warning=bool(sidecar['fk_fit_p95_m'] > .08)))
        report['state'] = 'complete'
    finally: write_json(output/'export_report.json', report)
    print(f'LAFAN1 BVH 已导出：{output}；{len(chunks)} 个片段；{exported_frames} 帧，'
          f'{exported_frames/fps:.2f} 秒；时间覆盖率 {report["coverage_ratio"]:.1%}', flush=True)
    if report['coverage_ratio'] < .9:
        print('质量提示：动作时间覆盖不足 90%，流程完成不代表完整恢复。请查看 export_report.json 的丢失时间点。', flush=True)
    return report
