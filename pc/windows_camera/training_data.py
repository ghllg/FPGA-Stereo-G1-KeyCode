"""Read timestamp-matched RGB pairs without assuming equal video frame numbers."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import av


class VideoReader:
    def __init__(self, container):
        self.frames = iter(container.decode(video=0))
        self.index = -1

    def at(self, target):
        if target <= self.index:
            raise ValueError('配对表帧号必须严格递增，不能重复使用帧。')
        frame = None
        while self.index < target:
            try:
                frame = next(self.frames)
            except StopIteration:
                raise ValueError(f'视频帧数不足，无法读取帧 {target}。') from None
            self.index += 1
        return frame.to_ndarray(format='rgb24')


def iter_training_pairs(folder, max_delta_ms=5.0):
    """Yield RGB arrays and precise stored timestamps, filtering saved pairs.

    max_delta_ms filters pairs.csv; it does not establish exposure accuracy.
    Videos are decoded sequentially. Each RGB array has shape (H, W, 3).
    Incomplete takes are rejected so interrupted recordings are not silently used.
    """
    if max_delta_ms < 0:
        raise ValueError('最大时间差不能为负数。')
    folder = Path(folder)
    metadata = json.loads((folder / 'session.json').read_text(encoding='utf-8'))
    if metadata.get('state') != 'complete':
        raise ValueError('该段录像未完整保存，请先检查 session.json。')
    if metadata.get('camera_count', len(metadata.get('devices', []))) != 2:
        raise ValueError('训练配对需要双路录像，当前是单相机数据。')
    origin = int(metadata['session_origin_source_ns'])
    threshold = round(max_delta_ms * 1_000_000)
    with (folder / 'pairs.csv').open(newline='', encoding='utf-8') as file, \
         av.open(str(folder / 'left.mkv')) as left, \
         av.open(str(folder / 'right.mkv')) as right:
        readers = [VideoReader(left), VideoReader(right)]
        for row in csv.DictReader(file):
            left_ns, right_ns = int(row['left_time_ns']), int(row['right_time_ns'])
            delta = right_ns - left_ns
            if abs(delta) > threshold:
                continue
            left_index, right_index = int(row['left_frame']), int(row['right_frame'])
            yield {
                'left_rgb': readers[0].at(left_index),
                'right_rgb': readers[1].at(right_index),
                'left_frame_index': left_index,
                'right_frame_index': right_index,
                'left_source_ns': left_ns,
                'right_source_ns': right_ns,
                'left_session_ns': left_ns - origin,
                'right_session_ns': right_ns - origin,
                'right_minus_left_ms': delta / 1_000_000,
            }
