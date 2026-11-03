"""Recorded JPEG replay through the live engine, with auditable pacing assumptions."""
import csv
import json
import threading
import time
from pathlib import Path
import av
from .source import jpeg_pixel_format
from .sync import Frame


class ReplaySource:
    def __init__(self, session, config, on_frame, on_error, *, pacing='original_receive', right_offset_ns=0):
        if pacing not in ('original_receive', 'content_aligned'): raise ValueError('未知录像回放节奏。')
        self.session = Path(session); self.config = config; self.on_frame = on_frame; self.on_error = on_error
        self.pacing, self.offset = pacing, right_offset_ns
        self.stop_event = threading.Event(); self.done = threading.Event(); self.lock = threading.Lock()
        self.counts = [0, 0]; self.finished = 0; self.pixel_formats = [None, None]; self.stderr = [[], []]
        self.tables = []
        for side in ('left', 'right'):
            with (self.session/f'{side}_timestamps.csv').open(encoding='utf-8', newline='') as f:
                rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)]
            if not rows: raise ValueError('录像时间表为空。')
            if any(row['frame_index'] != i for i, row in enumerate(rows)): raise ValueError('录像索引不连续。')
            times = [row['source_timestamp_ns'] for row in rows]
            if any(a >= b for a, b in zip(times, times[1:])): raise ValueError('录像源时间回退。')
            self.tables.append(rows)
        if pacing == 'original_receive':
            self.origin = min(t[0]['host_receive_monotonic_ns'] for t in self.tables)
        else:
            self.origin = max(self.tables[c][0]['source_timestamp_ns']-(self.offset if c else 0) for c in range(2))
        self.elapsed = [0., 0.]

    def start(self):
        self.started = time.perf_counter(); self.threads = []
        for c in range(2):
            t = threading.Thread(target=self._run, args=(c,), daemon=True); self.threads.append(t); t.start()

    def _run(self, c):
        try:
            side = ('left', 'right')[c]; count = 0
            with av.open(str(self.session/f'{side}.mkv')) as video:
                for packet in video.demux(video=0):
                    if self.stop_event.is_set(): return
                    if not packet.size: continue
                    if count >= len(self.tables[c]): raise ValueError('视频帧多于时间表。')
                    row = self.tables[c][count]; count += 1
                    timestamp = row['source_timestamp_ns']
                    delivery = row['host_receive_monotonic_ns'] if self.pacing == 'original_receive' else timestamp-(self.offset if c else 0)
                    relative = (delivery-self.origin)/1e9
                    if relative < 0: continue
                    delay = self.started+relative-time.perf_counter()
                    while delay > 0 and not self.stop_event.is_set():
                        time.sleep(min(delay, .01)); delay = self.started+relative-time.perf_counter()
                    if self.stop_event.is_set(): return
                    payload = bytes(packet)
                    if self.pixel_formats[c] is None: self.pixel_formats[c] = jpeg_pixel_format(payload)
                    self.on_frame(Frame(c, row['frame_index'], timestamp, time.perf_counter_ns(), payload))
                    self.counts[c] += 1; self.elapsed[c] = relative
                if count != len(self.tables[c]): raise ValueError(f'{side} 视频与时间表帧数不同：{count}/{len(self.tables[c])}')
        except Exception as exc: self.on_error(f'录像回放 cam{c}: {exc}')
        finally:
            with self.lock:
                self.finished += 1
                if self.finished == 2: self.done.set()

    def stop(self):
        self.stop_event.set()
        for t in self.threads: t.join(timeout=3)


def run_replay(session, calibration, output, *, pacing='original_receive', right_offset_ns=0, model=None, inference_size=640, refine_person=False):
    from windows_camera.capture import Config, Device
    from mocap.io import read_json, write_json
    from .engine import Engine
    session, output = Path(session), Path(output)
    if output.exists(): raise ValueError('录像测试输出已存在，请使用新目录。')
    meta = read_json(session/'session.json'); mode = meta['requested_mode']
    config = Config(tuple(Device(**d) for d in meta['devices']), width=mode['width'], height=mode['height'],
                    fps=mode['fps'], timestamp_mode=meta.get('timestamp_mode', 'driver'))
    alignment = dict(method='prior_offline_content_estimate_for_replay' if right_offset_ns else 'unverified_raw_driver_clock_baseline',
                     right_offset_ns=right_offset_ns, hardware_synchronized=False, exposure_timestamp_verified=False,
                     pacing=pacing, verified_by_flash=False,
                     note='QA-only replay, not a live synchronization measurement. content_aligned pacing uses previously recorded future packets.')
    factory = lambda cfg, receive, fail: ReplaySource(session, cfg, receive, fail, pacing=pacing, right_offset_ns=right_offset_ns)
    engine = Engine(config, str(calibration), model=model, source_factory=factory,
                    replay_alignment=alignment, validation_record=True, inference_size=inference_size, refine_person=refine_person)
    engine.start(); started = time.perf_counter(); requested = False; progress = 0
    try:
        while engine.thread.is_alive():
            if engine.source:
                with engine.lock: frames_ready = all(engine.latest_frames)
                if frames_ready and not requested:
                    engine.request('record', str(output)); requested = True
                if engine.source.done.is_set(): time.sleep(.15); break
            if time.perf_counter()-started > 180: raise TimeoutError('录像验证超时。')
            if time.perf_counter()-progress > 3:
                snap = engine.read(); print('回放', pacing, '采集',snap.get('capture_counts'), '骨架', engine.poses,
                                            '状态',snap.get('status'), flush=True); progress = time.perf_counter()
            time.sleep(.02)
    finally:
        engine.stop(); engine.thread.join(timeout=60)
    report = dict(session=str(session.resolve()), pacing=pacing, right_offset_ns=right_offset_ns,
                  error=engine.error, actual_causal_poses=engine.poses, elapsed_wall_s=time.perf_counter()-started,
                  input_frames=engine.source.counts if engine.source else None, dropped_frames=list(engine.buffer.dropped),
                  valid_initialization=engine.offsets is not None, external_accuracy_validated=False)
    if output.exists(): write_json(output/'replay_report.json', report)
    if engine.error or engine.thread.is_alive(): raise RuntimeError(engine.error or '回放停止超时')
    return report


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='使用原录像对实时引擎做有时间节奏的测试')
    parser.add_argument('--session', required=True); parser.add_argument('--calibration', required=True)
    parser.add_argument('--output', required=True); parser.add_argument('--pacing', choices=['original_receive','content_aligned'], default='original_receive')
    parser.add_argument('--right-offset-ms', type=float, default=0); parser.add_argument('--model')
    parser.add_argument('--image-size', type=int, default=640)
    parser.add_argument('--refine-person', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run_replay(args.session, args.calibration, args.output, pacing=args.pacing,
                               right_offset_ns=round(args.right_offset_ms*1e6), model=args.model,
                               inference_size=args.image_size, refine_person=args.refine_person), ensure_ascii=False, indent=2))
