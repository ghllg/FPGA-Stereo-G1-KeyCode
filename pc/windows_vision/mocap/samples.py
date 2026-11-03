"""Explicit synthetic contract fixture, not camera validation or training ground truth."""
import json
from pathlib import Path
import numpy as np
from .io import frame_packet, write_json
from .lafan import export_lafan


def standing_coco():
    return np.array([[0, 0, 1.72], [0, .035, 1.76], [0, -.035, 1.76], [0, .075, 1.73], [0, -.075, 1.73],
                     [0, .20, 1.45], [0, -.20, 1.45], [0, .49, 1.45], [0, -.49, 1.45],
                     [0, .75, 1.45], [0, -.75, 1.45], [0, .09, .94], [0, -.09, .94],
                     [0, .09, .52], [0, -.09, .52], [0, .09, .10], [0, -.09, .10]])


def generate_sample(output):
    output = Path(output)
    if output.exists(): raise ValueError('人工样例输出目录已存在。')
    output.mkdir(parents=True)
    path = output/'synthetic_skeleton3d.jsonl'
    with path.open('x', encoding='utf-8') as f:
        for i in range(91):
            points = standing_coco(); angle = np.sin(i/90*np.pi)*.9
            points[7] = points[5] + [0, .29*np.cos(angle), .29*np.sin(angle)]
            points[9] = points[7] + [0, .26*np.cos(angle), .26*np.sin(angle)]
            joints = [dict(xyz=p.tolist(), confidence=1., valid=True, estimated=True, reason='synthetic_fixture') for p in points]
            packet = frame_packet(i, 1_000_000_000 + round(i/30*1e9), joints, session_id='SYNTHETIC_NOT_MEASURED',
                                  calibration_id='synthetic', model_id='synthetic',
                                  coordinate_system='world_right_handed_X_forward_Y_left_Z_up', clock_id='synthetic:1')
            f.write(json.dumps(packet)+'\n')
    write_json(path.with_suffix('.meta.json'), dict(state='complete', synthetic=True))
    export_lafan(path, output/'lafan1')
    print('人工样例已生成；此样例只验证接口和导出，不代表双摄精度。', flush=True)

