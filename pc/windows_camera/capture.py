"""MJPEG capture without re-encoding; sample-time alignment, not exposure sync."""
from __future__ import annotations

import csv
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import av
import imageio_ffmpeg
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

LOG = logging.getLogger(__name__)
NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
NS = Fraction(1, 1_000_000_000)


class ReadPipe:
    """Expose only read: Windows anonymous pipes must not advertise seek()."""
    def __init__(self, pipe):
        self.pipe = pipe

    def read(self, size):
        return self.pipe.read(size)


def ffmpeg_path():
    return os.environ.get('STEREO_FFMPEG') or imageio_ffmpeg.get_ffmpeg_exe()


@dataclass(frozen=True)
class Device:
    name: str
    identifier: str


def parse_devices(text):
    devices = []
    name = None
    for line in text.splitlines():
        match = re.search(r'"(.*?)"\s+\(video\)', line)
        if match:
            name = match.group(1)
        elif '(audio)' in line:
            name = None
        elif name:
            match = re.search(r'Alternative name "(.*?)"', line)
            if match:
                devices.append(Device(name, match.group(1)))
                name = None
    return devices


def list_devices():
    result = subprocess.run(
        [ffmpeg_path(), '-hide_banner', '-list_devices', 'true', '-f', 'dshow', '-i', 'dummy'],
        capture_output=True, timeout=15, creationflags=NO_WINDOW,
    )
    text = result.stderr.decode('utf-8', errors='replace')
    LOG.info('Device enumeration:\n%s', text)
    devices = parse_devices(text)
    if not devices and 'Unknown input format' in text:
        raise RuntimeError('FFmpeg 缺少 DirectShow 支持。')
    return devices


@dataclass(frozen=True)
class Config:
    devices: tuple[Device, ...]
    width: int = 1920
    height: int = 1080
    fps: int = 60
    tolerance_ms: float = 10.0
    demo: bool = False
    timestamp_mode: str = 'driver'
    pixel_formats: tuple[str, ...] = ()


def capture_command(config):
    cmd = [ffmpeg_path(), '-hide_banner', '-loglevel', 'warning', '-nostdin', '-copyts']
    for i, device in enumerate(config.devices):
        if config.demo:
            cmd += ['-re', '-f', 'lavfi', '-i',
                    f'testsrc2=size={config.width}x{config.height}:rate={config.fps}']
        else:
            cmd += ['-thread_queue_size', '128', '-f', 'dshow', '-rtbufsize', '64M',
                    '-video_size', f'{config.width}x{config.height}',
                    # Integer 60 fails at the H200S descriptor boundary. A
                    # 59.94 request is accepted and negotiates its ~60 fps mode.
                    '-framerate', '60000/1001' if config.fps == 60 else str(config.fps), '-vcodec', 'mjpeg',
                    '-use_video_device_timestamps', 'true' if config.timestamp_mode == 'driver' else 'false',
                    '-i', f'video={device.identifier}']
    for i in range(len(config.devices)):
        cmd += ['-map', f'{i}:v:0']
    cmd += ['-an']
    if config.demo:
        cmd += ['-c:v', 'mjpeg', '-q:v', '3', '-threads', '2']
    else:
        cmd += ['-c:v', 'copy']
    return cmd + ['-avoid_negative_ts', 'disabled', '-max_interleave_delta', '100000',
                  '-flush_packets', '1', '-f', 'nut', 'pipe:1']


@dataclass(frozen=True)
class Sample:
    camera: int
    timestamp_ns: int
    receive_ns: int
    payload: bytes


@dataclass(frozen=True)
class PreviewInfo:
    capture_index: int
    timestamp_ns: int
    recording_index: int | None
    recording_origin_ns: int | None


def format_time_ns(value):
    """Readable session time; six fractional digits are units, not accuracy."""
    sign = '-' if value < 0 else ''
    microseconds = abs(value) // 1000
    seconds, fraction = divmod(microseconds, 1_000_000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{sign}{hours:02d}:{minutes:02d}:{seconds:02d}.{fraction:06d}'


def srt_time(value_ns):
    milliseconds = max(0, value_ns // 1_000_000)
    seconds, fraction = divmod(milliseconds, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d},{fraction:03d}'


def write_subtitle(file, camera, index, source_ns, origin_ns, end_ns):
    relative = source_ns - origin_ns
    end = max(end_ns - origin_ns, relative + 1_000_000)
    file.write(f'{index + 1}\n{srt_time(relative)} --> {srt_time(end)}\n'
               f'{"AB"[camera]} | frame={index} | t={format_time_ns(relative)}\n'
               f'source_ns={source_ns}\n\n')


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def align_csv(folder, tolerance_ms):
    """Streaming monotonic one-to-one pairing; unmatched frames remain in videos."""
    counts = [0, 0]
    matched = 0
    total_delta = 0
    max_delta = 0
    histogram = [0] * 1001
    threshold = round(tolerance_ms * 1_000_000)
    with (folder / 'left_timestamps.csv').open(newline='', encoding='utf-8') as left, \
         (folder / 'right_timestamps.csv').open(newline='', encoding='utf-8') as right, \
         (folder / 'pairs.csv').open('w', newline='', encoding='utf-8') as output:
        readers = [iter(csv.DictReader(left)), iter(csv.DictReader(right))]
        rows = [next(readers[0], None), next(readers[1], None)]
        writer = csv.writer(output)
        writer.writerow(['left_frame', 'right_frame', 'left_time_ns', 'right_time_ns', 'right_minus_left_ms'])
        while all(row is not None for row in rows):
            times = [int(row['source_timestamp_ns']) for row in rows]
            delta = times[1] - times[0]
            if abs(delta) <= threshold:
                writer.writerow([rows[0]['frame_index'], rows[1]['frame_index'], *times, f'{delta / 1e6:.6f}'])
                matched += 1
                absolute = abs(delta)
                total_delta += absolute
                max_delta = max(max_delta, absolute)
                histogram[min(1000, int(absolute / 100_000))] += 1
                advance = [0, 1]
            else:
                advance = [0 if delta > 0 else 1]
            for camera in advance:
                counts[camera] += 1
                rows[camera] = next(readers[camera], None)
        for camera in range(2):
            if rows[camera] is not None:
                counts[camera] += 1 + sum(1 for _ in readers[camera])
    target = max(1, (matched * 95 + 99) // 100)
    cumulative = 0
    p95 = None
    for index, number in enumerate(histogram):
        cumulative += number
        if cumulative >= target:
            p95 = (index + 1) / 10
            break
    return {'paired_frames': matched, 'unpaired_frames': [n - matched for n in counts],
            'paired_ratio_per_camera': [matched / n if n else 0 for n in counts],
            'mean_abs_sample_delta_ms': total_delta / matched / 1e6 if matched else None,
            'max_abs_sample_delta_ms': max_delta / 1e6 if matched else None,
            'p95_abs_sample_delta_ms_upper_bound': p95,
            'pair_tolerance_ms': tolerance_ms,
            'pairing': 'monotonic one-to-one threshold; does not measure exposure synchronization'}


class Recording:
    """A single writer drains accepted packets before closing either video."""
    def __init__(self, folder, config, origin_ns):
        self.folder = Path(folder)
        self.config = config
        self.camera_count = len(config.devices)
        self.origin_ns = origin_ns
        self.lock = threading.Lock()
        self.pending = queue.Queue(maxsize=512)
        self.pending_bytes = 0
        self.accepting = True
        self.done = threading.Event()
        self.ready = threading.Event()
        self.error = ''
        self.reason = ''
        self.frames = [0] * self.camera_count
        self.accepted_frames = [0] * self.camera_count
        self.bytes_written = 0
        self.metadata = {
            'version': 1, 'state': 'recording', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'devices': [vars(device) for device in config.devices],
            'camera_count': self.camera_count,
            'requested_mode': {'width': config.width, 'height': config.height, 'fps': config.fps, 'codec': 'MJPEG'},
            'driver_framerate_request': '60000/1001' if config.fps == 60 else str(config.fps),
            'negotiated_pixel_formats': config.pixel_formats,
            'timestamp_source_requested': ('synthetic_demo' if config.demo else
                                           'DirectShow driver sample time plus graph start reference time' if config.timestamp_mode == 'driver' else
                                           'DirectShow graph clock at sample callback'),
            'timestamp_mode': config.timestamp_mode,
            'timestamp_fallback_note': 'FFmpeg may fall back to graph callback time if driver sample time is absent; individual fallback frames are not identified.',
            'hardware_synchronized': False, 'exposure_timestamp_available': False,
            'session_origin_source_ns': origin_ns,
            'video_timebase': 'MKV milliseconds; CSV retains source timestamps in nanosecond units, not nanosecond accuracy',
            'timestamp_subtitles': 'left.srt / right.srt: optional playback overlay, not burned into training video',
            'frame_index': 'zero-based index in each saved video; use pairs.csv rather than matching equal frame indices',
            'note': 'Rolling shutter. Driver sample time is not a confirmed sensor exposure timestamp. Callback timing includes USB/driver latency. Validate visually with a shared flash/clap.',
        }
        self.folder.mkdir(parents=True, exist_ok=False)
        write_json(self.folder / 'session.json', self.metadata)
        self.thread = threading.Thread(target=self._run, name='video-writer', daemon=True)
        self.thread.start()

    def submit(self, sample):
        with self.lock:
            if not self.accepting or sample.timestamp_ns < self.origin_ns:
                return
            if self.pending_bytes + len(sample.payload) > 128 * 1024 * 1024:
                self._fail('录像写入积压超过 128 MB，已停止采集并保存已有帧。')
                return
            try:
                self.pending.put_nowait(sample)
                self.pending_bytes += len(sample.payload)
                index = self.accepted_frames[sample.camera]
                self.accepted_frames[sample.camera] += 1
                return index
            except queue.Full:
                self._fail('录像队列已满，已停止采集并保存已有帧。')

    def _fail(self, message):
        self.error = message
        self.accepting = False

    def stop(self, reason=''):
        with self.lock:
            self.accepting = False
            if reason:
                self.reason = reason

    def _run(self):
        containers, files, streams, writers, subtitles = [], [], [], [], []
        last = [None] * self.camera_count
        first = [None] * self.camera_count
        gaps = [0] * self.camera_count
        try:
            for camera, side in enumerate(('left', 'right')[:self.camera_count]):
                output = av.open(str(self.folder / f'{side}.mkv'), 'w', format='matroska')
                containers.append(output)
                stream = output.add_stream('mjpeg', rate=self.config.fps)
                stream.width = self.config.width
                stream.height = self.config.height
                stream.pix_fmt = self.config.pixel_formats[camera] if self.config.pixel_formats else 'yuvj420p'
                stream.time_base = NS
                streams.append(stream)
                file = (self.folder / f'{side}_timestamps.csv').open('w', newline='', encoding='utf-8')
                files.append(file)
                writer = csv.writer(file)
                writer.writerow(['frame_index', 'source_timestamp_ns', 'session_time_ns', 'host_receive_monotonic_ns', 'packet_bytes'])
                writers.append(writer)
                subtitles.append((self.folder / f'{side}.srt').open('w', encoding='utf-8'))
            self.ready.set()
            while True:
                try:
                    sample = self.pending.get(timeout=.1)
                except queue.Empty:
                    with self.lock:
                        if not self.accepting and self.pending.empty():
                            break
                    continue
                with self.lock:
                    self.pending_bytes -= len(sample.payload)
                camera = sample.camera
                if last[camera] is not None:
                    if sample.timestamp_ns <= last[camera]:
                        raise RuntimeError('相机时间戳重复或回退，本次数据已标记异常。')
                    if sample.timestamp_ns - last[camera] > 1.8e9 / self.config.fps:
                        gaps[camera] += 1
                if first[camera] is None:
                    first[camera] = sample.timestamp_ns
                packet = av.Packet(sample.payload)
                packet.pts = packet.dts = sample.timestamp_ns - self.origin_ns
                packet.time_base = NS
                packet.duration = round(1e9 / self.config.fps)
                packet.is_keyframe = True
                packet.stream = streams[camera]
                containers[camera].mux(packet)
                if last[camera] is not None:
                    write_subtitle(subtitles[camera], camera, self.frames[camera] - 1,
                                   last[camera], self.origin_ns, sample.timestamp_ns)
                writers[camera].writerow([self.frames[camera], sample.timestamp_ns,
                                         sample.timestamp_ns - self.origin_ns,
                                         sample.receive_ns, len(sample.payload)])
                self.frames[camera] += 1
                self.bytes_written += len(sample.payload)
                last[camera] = sample.timestamp_ns
                if self.frames[camera] % self.config.fps == 0:
                    files[camera].flush()
        except Exception as exc:
            LOG.exception('Recording failed')
            with self.lock:
                self._fail(str(exc))
        finally:
            for camera, file in enumerate(subtitles):
                try:
                    if self.frames[camera]:
                        write_subtitle(file, camera, self.frames[camera] - 1,
                                       last[camera], self.origin_ns, last[camera] + round(1e9 / self.config.fps))
                    file.close()
                except Exception as exc:
                    self.error = self.error or str(exc)
            for output in containers:
                try:
                    output.close()
                except Exception as exc:
                    self.error = self.error or str(exc)
            for file in files:
                try:
                    file.close()
                except Exception as exc:
                    self.error = self.error or str(exc)
            stats = {}
            try:
                stats = (align_csv(self.folder, self.config.tolerance_ms) if self.camera_count == 2
                         else {'pairing': 'not applicable: single camera; no stereo synchronization'})
            except Exception as exc:
                LOG.exception('Alignment failed')
                self.error = self.error or str(exc)
            if not all(self.frames):
                self.error = self.error or '至少一路没有录到帧，请重新采集。'
            measured_fps = [((self.frames[i] - 1) * 1e9 / (last[i] - first[i]))
                            if self.frames[i] > 1 else None for i in range(self.camera_count)]
            valid = not self.error and not self.reason and all(self.frames)
            self.metadata.update({'state': 'complete' if valid else 'incomplete',
                                  'error': self.error, 'stop_reason': self.reason,
                                  'frames': self.frames, 'first_source_ns': first, 'last_source_ns': last,
                                  'measured_sample_fps': measured_fps,
                                  'timestamp_gap_events': gaps, 'synchronization': stats,
                                  'finished_utc': datetime.now(timezone.utc).isoformat()})
            try:
                write_json(self.folder / 'session.json', self.metadata)
            except Exception as exc:
                LOG.exception('Metadata save failed')
                self.error = self.error or str(exc)
            self.ready.set()
            self.done.set()


class Capture(QThread):
    failed = Signal(str)
    initialized = Signal()

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.camera_count = len(config.devices)
        self.lock = threading.Lock()
        self.process = None
        self.stopping = threading.Event()
        self.recording = None
        self.images = [None] * self.camera_count
        self.preview_info = [None] * self.camera_count
        self.preview_origin_ns = None
        self.latest = [None] * self.camera_count
        self.received = [None] * self.camera_count
        self.recent = [deque(maxlen=180) for _ in config.devices]
        self.counts = [0] * self.camera_count
        self.errors = deque(maxlen=30)
        self.stream_ready = False

    def attach_recording(self, recording):
        with self.lock:
            self.recording = recording
            if recording:
                recording.metadata['capture_warnings'] = list(self.errors)

    def snapshot(self):
        with self.lock:
            return (self.images.copy(), self.latest.copy(), self.received.copy(),
                    [list(values) for values in self.recent], self.counts.copy())

    def preview_snapshot(self):
        with self.lock:
            return self.images.copy(), self.preview_info.copy(), self.preview_origin_ns

    def _stderr(self, process):
        for raw in iter(process.stderr.readline, b''):
            line = raw.decode('utf-8', errors='replace').strip()
            self.errors.append(line)
            LOG.warning('FFmpeg: %s', line)
            with self.lock:
                recording = self.recording
            if recording:
                with recording.lock:
                    warnings = recording.metadata.setdefault('capture_warnings', [])
                    if len(warnings) < 100:
                        warnings.append(line)
            if 'buffer' in line.lower() and ('too full' in line.lower() or 'dropp' in line.lower()):
                self.failed.emit('USB/采集缓冲区丢帧：' + line)

    def stop(self):
        self.stopping.set()
        with self.lock:
            process = self.process
        if process and process.poll() is None:
            process.terminate()

    def run(self):
        process = None
        container = None
        try:
            command = capture_command(self.config)
            LOG.info('Capture command: %r', command)
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, creationflags=NO_WINDOW, bufsize=0)
            with self.lock:
                self.process = process
            if self.stopping.is_set():
                process.terminate()
                return
            stderr = threading.Thread(target=self._stderr, args=(process,), daemon=True)
            stderr.start()
            container = av.open(ReadPipe(process.stdout), 'r', format='nut', options={'probesize': '32768', 'analyzeduration': '0'})
            formats = tuple(stream.codec_context.format.name if stream.codec_context.format else 'yuvj422p'
                            for stream in container.streams.video)
            self.config = replace(self.config, pixel_formats=formats)
            decoders = [av.CodecContext.create('mjpeg', 'r') for _ in self.config.devices]
            preview_at = [0] * self.camera_count
            for packet in container.demux(video=tuple(range(self.camera_count))):
                if self.stopping.is_set():
                    break
                if not packet.size or packet.pts is None:
                    continue
                camera = packet.stream.index
                if camera not in range(self.camera_count):
                    continue
                if packet.stream.codec_context.width != self.config.width or packet.stream.codec_context.height != self.config.height:
                    raise RuntimeError('实际分辨率与所选模式不一致，请换一个模式重试。')
                timestamp = int(packet.pts * packet.time_base * 1_000_000_000)
                now = time.perf_counter_ns()
                sample = Sample(camera, timestamp, now, bytes(packet))
                with self.lock:
                    if self.latest[camera] is not None and timestamp <= self.latest[camera]:
                        raise RuntimeError('采集时间戳重复或回退。')
                    self.latest[camera] = timestamp
                    self.received[camera] = now
                    self.recent[camera].append(timestamp)
                    self.counts[camera] += 1
                    recording = self.recording
                    ready = not self.stream_ready and all(value is not None for value in self.latest)
                    if ready:
                        self.stream_ready = True
                        self.preview_origin_ns = min(self.latest)
                    capture_index = self.counts[camera] - 1
                recording_index = None
                if recording:
                    recording_index = recording.submit(sample)
                if ready:
                    self.initialized.emit()
                if now - preview_at[camera] > 100_000_000:
                    preview_at[camera] = now
                    frames = decoders[camera].decode(av.Packet(sample.payload))
                    if frames:
                        frame = frames[0].reformat(width=640, height=round(640 * self.config.height / self.config.width), format='rgb24')
                        plane = frame.planes[0]
                        picture = QImage(bytes(plane), frame.width, frame.height, plane.line_size, QImage.Format.Format_RGB888).copy()
                        with self.lock:
                            self.images[camera] = picture
                            self.preview_info[camera] = PreviewInfo(capture_index, timestamp, recording_index,
                                                                   recording.origin_ns if recording_index is not None else None)
            if not self.stopping.is_set():
                raise RuntimeError('相机采集结束，请检查 USB 连接。\n' + '\n'.join(self.errors))
        except Exception as exc:
            if not self.stopping.is_set():
                LOG.exception('Capture failed')
                self.failed.emit(str(exc) + '\n' + '\n'.join(self.errors)[-2000:])
        finally:
            if process:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if container:
                container.close()
            if process:
                process.stdout.close()
                process.stderr.close()
