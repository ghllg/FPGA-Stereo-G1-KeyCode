"""Auditable selection between full-image and ROI predictions; no coordinate invention."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import numpy as np
from .calibration import load_calibration
from .io import read_json, write_json
from .reconstruction import triangulate


def fuse_pose(full_file, roi_file, calibration_file, output, agreement_px=40.):
    full_file, roi_file, output = map(Path, (full_file, roi_file, output))
    if output.exists(): raise ValueError('模型候选融合输出已存在。')
    full_meta, roi_meta = [read_json(p.with_suffix('.meta.json')) for p in (full_file, roi_file)]
    keys = ('state', 'source_session', 'camera_ids', 'model_id', 'clock_id')
    if full_meta['state'] != 'complete' or any(full_meta[k] != roi_meta[k] for k in keys):
        raise ValueError('只能融合同一会话、相机、模型及时钟的完整二维点。')
    if not all(m.get('temporal_alignment') for m in (full_meta, roi_meta)):
        raise ValueError('两个候选文件均需先校正内容同步。')
    offsets = [m['temporal_alignment']['right_timestamp_subtract_s'] for m in (full_meta, roi_meta)]
    if abs(offsets[0]-offsets[1]) > .01: raise ValueError('两个候选的内容同步偏移不一致。')
    cal = load_calibration(calibration_file)
    if cal['camera_ids'] != full_meta['camera_ids']: raise ValueError('标定相机身份不一致。')
    arrays = [[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
              for p in (full_file, roi_file)]
    if len(arrays[0]) != len(arrays[1]): raise ValueError('候选帧数不一致。')
    selected = [0, 0, 0, 0]; previous = [0]*17
    output.parent.mkdir(parents=True, exist_ok=True)
    lineage = dict(method='bounded_model_candidate_selection', full_source=str(full_file.resolve()),
                   roi_source=str(roi_file.resolve()), maximum_candidate_agreement_px=agreement_px,
                   non_full_candidate_penalty_px=2., switching_penalty_px=.5,
                   note='Only existing model predictions are selected; full/ROI candidates must agree within the pixel bound. Selection by camera consistency is not independent accuracy validation.')
    manifest = dict(full_meta, state='incomplete', pose_ensemble=lineage)
    write_json(output.with_suffix('.meta.json'), manifest)
    with output.open('x', encoding='utf-8') as dst:
        for full, roi in zip(*arrays):
            if full['timestamp_ns'] != roi['timestamp_ns']: raise ValueError('候选参考时间不一致。')
            row = copy.deepcopy(full)
            candidates = []
            modes = ((0, 0), (1, 1), (0, 1), (1, 0))
            for mode in modes:
                left = (full, roi)[mode[0]]['views'][0]['joints']
                right = (full, roi)[mode[1]]['views'][1]['joints']
                result = triangulate(left, right, cal, max_reprojection_px=25.)
                candidates.append(result)
            for j in range(17):
                agreement = all(full['views'][cam]['joints'][j]['valid'] and roi['views'][cam]['joints'][j]['valid'] and
                    np.linalg.norm(np.array(full['views'][cam]['joints'][j]['xy'])-
                                   np.array(roi['views'][cam]['joints'][j]['xy'])) <= agreement_px for cam in range(2))
                eligible = range(4) if agreement else (0,)
                def score(mode):
                    item = candidates[mode][j]
                    return ((max(item.get('reprojection_px') or [1e6]) if item['valid'] else 1e6) +
                            (2. if mode else 0.) + (.5 if previous[j] != mode else 0.))
                # Keep the trusted whole-frame prediction unless a bounded candidate improves consistency.
                chosen = min(eligible, key=score); previous[j] = chosen; selected[chosen] += 1
                for cam in range(2):
                    candidate = modes[chosen][cam]
                    joint = copy.deepcopy((full, roi)[candidate]['views'][cam]['joints'][j])
                    joint['pose_candidate_source'] = 'person_roi' if candidate else 'full_frame'
                    if candidate: joint['candidate_temporal_alignment'] = roi['temporal_alignment']
                    row['views'][cam]['joints'][j] = joint
            row['pose_ensemble'] = lineage
            dst.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    manifest.update(state='complete', rows=len(arrays[0]), candidate_pair_counts=selected)
    write_json(output.with_suffix('.meta.json'), manifest)
    print(f'模型候选融合：全图/裁剪/混合候选计数 {selected}；{output}', flush=True)
    return manifest
