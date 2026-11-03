"""Offline content alignment. Driver timestamps are retained, never presented as exposure times."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import cv2
import numpy as np
from .calibration import load_calibration
from .io import read_json, write_json


def estimate_offset(rows, calibration, max_offset_s=2.):
    if not 0 < max_offset_s <= 10:
        raise ValueError('内容同步搜索范围必须在 0 到 10 秒之间。')
    origin = rows[0]['timestamp_ns']
    times = [np.array([(row['views'][cam]['timestamp_ns']-origin)/1e9 for row in rows]) for cam in range(2)]
    if any(np.any(np.diff(t) <= 0) for t in times):
        raise ValueError('每路二维点的原始时间戳必须严格递增。')
    pixels = []
    for cam in range(2):
        x = np.array([[j['xy'] if j['valid'] else [np.nan, np.nan]
                       for j in row['views'][cam]['joints']] for row in rows])
        pixels.append(cv2.undistortPoints(x.reshape(-1, 1, 2), np.array(calibration['K'][cam]),
                      np.array(calibration['distortion'][cam]), P=np.array(calibration['K'][cam])).reshape(-1, 17, 2))
    r = np.array(calibration['R_cam0_to_cam1']); t = np.array(calibration['t_cam0_to_cam1_m'])
    cross = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    fundamental = np.linalg.inv(calibration['K'][1]).T @ cross @ r @ np.linalg.inv(calibration['K'][0])
    dynamic_joints = [0, 5, 6, 7, 8, 9, 10, 11, 12]
    movement = np.nanmedian(np.linalg.norm(np.diff(pixels[0][:, dynamic_joints], axis=0), axis=2), axis=1)
    moving = np.r_[False, movement > .7]
    # Alternating half-second blocks provide separate fit and validation observations.
    fit = moving & ((np.floor(times[0]*2).astype(int) % 2) == 0)
    held = moving & ~fit
    if min(np.sum(fit), np.sum(held)) < 12:
        raise ValueError('动作变化不足，无法可靠自动估计内容时间差；请使用有动作的录像或明确给出已核对的时间偏移。')
    homogeneous_left = np.concatenate([pixels[0], np.ones((len(rows), 17, 1))], axis=2)
    lines_right = homogeneous_left @ fundamental.T

    def residual(offset):
        query = times[0]+offset
        hi = np.searchsorted(times[1], query).clip(1, len(rows)-1); lo = hi-1
        alpha = (query-times[1][lo])/(times[1][hi]-times[1][lo])
        right = pixels[1][lo]*(1-alpha[:, None, None])+pixels[1][hi]*alpha[:, None, None]
        hright = np.concatenate([right, np.ones((len(rows), 17, 1))], axis=2)
        lines_left = hright @ fundamental
        numerator = np.abs(np.sum(lines_right*hright, axis=2))
        error = numerator * np.sqrt(.5/(np.sum(lines_right[:, :, :2]**2, axis=2)) +
                                    .5/(np.sum(lines_left[:, :, :2]**2, axis=2)))
        available = (query >= times[1][0]) & (query <= times[1][-1]) & (times[1][hi]-times[1][lo] <= .12)
        error[~available] = np.nan
        return error[:, dynamic_joints]

    def score(offset, mask):
        errors = residual(offset)[mask]
        finite = errors[np.isfinite(errors)]
        return float(np.median(finite)) if len(finite) >= 50 else float('inf')

    grid = np.linspace(-max_offset_s, max_offset_s, int(max_offset_s*240)+1)
    scores = [(score(float(o), fit), float(o)) for o in grid]
    _, coarse = min(scores)
    best_score, offset = min((score(float(o), fit), float(o)) for o in
                             np.linspace(max(-max_offset_s, coarse-.012), min(max_offset_s, coarse+.012), 25))
    before = score(0., held); after = score(offset, held)
    distinct = min(s for s, o in scores if abs(o-offset) > .15) if max_offset_s > .15 else float('inf')
    if abs(offset) > max_offset_s-.02 or not np.isfinite(after) or after > 15:
        raise ValueError('内容同步未通过几何验证，不能自动修改配对。请查看二维识别或标定质量。')
    if abs(offset) > .03 and (after > before*.75 or distinct < best_score*1.1):
        raise ValueError('时间偏移不够明确，禁止自动配对；需要人工检查画面。')
    report = dict(method='offline_pose_epipolar_constant_offset', right_timestamp_subtract_s=offset,
                  search_range_s=max_offset_s, fit_median_epipolar_px=best_score,
                  heldout_before_median_epipolar_px=before, heldout_after_median_epipolar_px=after,
                  fit_motion_frames=int(np.sum(fit)), heldout_motion_frames=int(np.sum(held)),
                  alternative_offset_best_score_px=distinct if np.isfinite(distinct) else None,
                  calibration_modified=False, hardware_sync=False, exposure_timestamps_verified=False,
                  note='Content-derived constant offset, not hardware synchronization. Interpolated 2D joints are estimates; raw timestamps and contributing frames are retained.')
    return offset, report


def align_pose(pose_file, calibration_file, output, max_offset_s=2., offset_s=None):
    source, output = Path(pose_file), Path(output)
    if output.exists(): raise ValueError('内容同步输出已存在，请选择新文件。')
    meta = read_json(source.with_suffix('.meta.json'))
    if meta.get('state') != 'complete': raise ValueError('二维点文件未完整生成。')
    cal = load_calibration(calibration_file)
    if meta['camera_ids'] != cal['camera_ids']: raise ValueError('相机身份/顺序与标定不一致。')
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    if len(rows) < 30: raise ValueError('内容同步至少需要 30 对二维点。')
    if any(len(row['views']) != 2 or any(v['camera_id'] != cal['camera_ids'][i] or
           v['image_size'] != cal['image_size'] for i, v in enumerate(row['views'])) for row in rows):
        raise ValueError('二维点相机身份、顺序或分辨率与标定不一致。')
    if any('temporal_alignment' in row for row in rows): raise ValueError('禁止重复校正已同步的二维点。')
    if offset_s is None: offset, report = estimate_offset(rows, cal, max_offset_s)
    else:
        if not np.isfinite(offset_s) or abs(offset_s) > 10: raise ValueError('手动内容偏移必须为有限数且绝对值不超过 10 秒。')
        offset = float(offset_s)
        report = dict(method='manual_content_offset', right_timestamp_subtract_s=offset,
                      calibration_modified=False, hardware_sync=False, exposure_timestamps_verified=False)
    right_times = np.array([row['views'][1]['timestamp_ns'] for row in rows], dtype=np.int64)
    if np.any(np.diff(right_times) <= 0): raise ValueError('右路原始时间戳必须严格递增。')
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(meta, state='incomplete', source_pose2d=str(source.resolve()), temporal_alignment=report, rows=0)
    write_json(output.with_suffix('.meta.json'), manifest)
    count = 0
    try:
        with output.open('x', encoding='utf-8') as dst:
            for source_seq, raw in enumerate(rows):
                query = raw['views'][0]['timestamp_ns']+round(offset*1e9)
                hi = int(np.searchsorted(right_times, query))
                if hi == len(rows) or query < right_times[0]: continue
                lo = max(0, hi-1)
                if right_times[hi]-right_times[lo] > 120_000_000: continue
                alpha = (query-int(right_times[lo]))/int(right_times[hi]-right_times[lo]) if hi != lo else 0.
                row = copy.deepcopy(raw); right = row['views'][1]
                a, b = rows[lo]['views'][1], rows[hi]['views'][1]
                for j, (left_joint, right_joint) in enumerate(zip(a['joints'], b['joints'])):
                    valid = left_joint['valid'] and right_joint['valid']
                    xy = ((1-alpha)*np.array(left_joint['xy'])+alpha*np.array(right_joint['xy'])).tolist() if valid else None
                    right['joints'][j] = dict(xy=xy, valid=bool(valid),
                        confidence=float(min(left_joint['confidence'], right_joint['confidence'])),
                        estimated=bool(0 < alpha < 1))
                provenance = dict(source_seq=source_seq, right_source_frames=[a['frame_index'], b['frame_index']],
                                  right_original_timestamps_ns=[a['timestamp_ns'], b['timestamp_ns']],
                                  right_interpolation_alpha=float(alpha), right_timestamp_subtract_s=offset,
                                  method=report['method'])
                right.update(frame_index=a['frame_index'] if alpha < .5 else b['frame_index'],
                             timestamp_ns=raw['views'][0]['timestamp_ns'], temporal_alignment=provenance)
                row.update(seq=count, timestamp_ns=raw['views'][0]['timestamp_ns'], pair_delta_ms=0.,
                           temporal_alignment=provenance)
                dst.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); count += 1
        if not count: raise ValueError('同步校正后没有共同时间区间。')
        manifest.update(state='complete', rows=count, dropped_boundary_or_gap_rows=len(rows)-count)
    finally: write_json(output.with_suffix('.meta.json'), manifest)
    print(f'内容同步：右路原始时间减去 {offset*1000:.1f} ms；保留 {count}/{len(rows)} 对；{output}', flush=True)
    return manifest
