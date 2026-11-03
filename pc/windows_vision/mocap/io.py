from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.skeleton_protocol import COCO17, VERSION, validate


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def pairs(session, max_delta_ms=10):
    sys.path.insert(0, str(ROOT / 'windows_camera'))
    from training_data import iter_training_pairs
    yield from iter_training_pairs(session, max_delta_ms)


def session_info(session):
    meta = read_json(Path(session) / 'session.json')
    if meta.get('state') != 'complete' or meta.get('camera_count', len(meta.get('devices', []))) != 2:
        raise ValueError('需要完整保存的双摄录像。')
    return meta


def camera_ids(meta):
    return [d['identifier'] for d in meta['devices']]


def calibration_samples(session, max_delta_ms, every, lookahead=6):
    """Sample saved pairs and nearby frames, retaining native pixels and a bounded cache.

    Future images only check motion independently in each camera; they are not
    substituted into the timestamp-matched calibration pair.
    """
    import csv
    import av
    folder = Path(session)
    if every < 1 or lookahead < 1: raise ValueError('抽样间隔和停稳检查帧间隔至少为 1。')
    if not max_delta_ms >= 0: raise ValueError('最大时间差不能为负数或 NaN。')
    session_info(folder)
    class BufferedReader:
        def __init__(self, container):
            self.frames = iter(container.decode(video=0)); self.index = -1; self.cache = {}
        def at(self, index):
            while self.index < index:
                try: frame = next(self.frames)
                except StopIteration: return None
                self.index += 1; self.cache[self.index] = frame
                self.cache.pop(self.index-lookahead-1, None)
            if index not in self.cache: raise ValueError('标定抽样帧顺序无效。')
            return self.cache[index].to_ndarray(format='rgb24')
    with (folder/'pairs.csv').open(newline='', encoding='utf-8') as f, av.open(str(folder/'left.mkv')) as left, av.open(str(folder/'right.mkv')) as right:
        readers = [BufferedReader(left), BufferedReader(right)]; number = -1; previous = [-1, -1]
        for row in csv.DictReader(f):
            times = [int(row['left_time_ns']), int(row['right_time_ns'])]
            if abs(times[1]-times[0]) > max_delta_ms*1_000_000: continue
            indices = [int(row['left_frame']), int(row['right_frame'])]
            if any(i <= last for i, last in zip(indices, previous)):
                raise ValueError('配对表帧号必须严格递增，不能重复使用帧。')
            previous = indices
            number += 1
            if number % every: continue
            rgb = [r.at(i) for r, i in zip(readers, indices)]
            later = [r.at(i+lookahead) for r, i in zip(readers, indices)]
            if any(x is None for x in rgb+later): break
            yield dict(pair_number=number, left_rgb=rgb[0], right_rgb=rgb[1],
                       left_frame_index=indices[0], right_frame_index=indices[1],
                       left_source_ns=times[0], right_source_ns=times[1],
                       stability_rgb=later)


def frame_packet(seq, timestamp, joints, *, session_id, calibration_id, model_id, coordinate_system, clock_id):
    return validate(dict(schema=VERSION, unit='m', joint_names=list(COCO17), seq=int(seq),
                         timestamp_ns=int(timestamp), session_id=session_id, source='windows_vision',
                         clock_id=clock_id, coordinate_system=coordinate_system,
                         calibration_id=calibration_id, model_id=model_id, joints=joints))
