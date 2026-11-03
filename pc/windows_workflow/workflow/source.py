"""One compressed FFmpeg pipe per camera. Capture threads never decode JPEG."""
import subprocess
import threading
import time
from dataclasses import replace
import av
import cv2
import numpy as np
from windows_camera.capture import capture_command, NO_WINDOW, ReadPipe
from .sync import Frame


def jpeg_pixel_format(data):
    """Read JPEG component sampling without decoding pixels or guessing chroma."""
    if data[:2] != b'\xff\xd8': raise ValueError('压缩包不是完整 JPEG。')
    pos = 2
    while pos+4 < len(data):
        if data[pos] != 255: raise ValueError('JPEG 标记损坏。')
        while pos < len(data) and data[pos] == 255: pos += 1
        marker = data[pos]; pos += 1
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7: continue
        size = int.from_bytes(data[pos:pos+2], 'big')
        if size < 2 or pos+size > len(data): raise ValueError('JPEG 段长度异常。')
        segment = data[pos+2:pos+size]
        if marker in (0xC0, 0xC1, 0xC2):
            if len(segment) < 9: raise ValueError('JPEG SOF 不完整。')
            count = segment[5]
            if count == 1: return 'gray'
            if count != 3 or len(segment) < 15 or segment[10] != 0x11 or segment[13] != 0x11:
                raise ValueError('JPEG 色彩采样不支持，停止以避免错误封装。')
            format_name = {0x22: 'yuvj420p', 0x21: 'yuvj422p', 0x11: 'yuvj444p',
                           0x12: 'yuvj440p', 0x41: 'yuvj411p'}.get(segment[7])
            if format_name: return format_name
            raise ValueError('JPEG 色彩采样未知。')
        if marker == 0xDA: break
        pos += size
    raise ValueError('JPEG 缺少 SOF。')


class CameraSource:
    def __init__(self, config, on_frame, on_error):
        self.config, self.on_frame, self.on_error = config, on_frame, on_error
        self.stop_event = threading.Event(); self.processes = []; self.threads = []
        self.pixel_formats = [None, None]; self.counts = [0, 0]
        self.stderr = [[], []]

    def start(self):
        for c in range(2):
            t = threading.Thread(target=self._read, args=(c,), daemon=True); self.threads.append(t); t.start()

    def _read(self, c):
        p = None
        try:
            if self.stop_event.is_set(): return
            command = capture_command(replace(self.config, devices=(self.config.devices[c],)))
            command[command.index('-thread_queue_size')+1] = '4'
            command[command.index('-rtbufsize')+1] = '8M'
            p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=NO_WINDOW)
            self.processes.append(p)
            if self.stop_event.is_set(): p.terminate(); return
            def log():
                for line in iter(p.stderr.readline, b''):
                    message = line.decode('utf-8', errors='replace').strip()
                    self.stderr[c].append(message); self.stderr[c] = self.stderr[c][-20:]
                    if 'buffer' in message.lower() and 'too full' in message.lower():
                        self.on_error(f'cam{c} 驱动缓冲溢出，请降低采集模式并重新校时。')
            threading.Thread(target=log, daemon=True).start()
            with av.open(ReadPipe(p.stdout), 'r', format='nut') as container:
                stream = container.streams.video[0]
                size = [stream.codec_context.width, stream.codec_context.height]
                if size != [self.config.width, self.config.height]: raise ValueError(f'cam{c} 实际分辨率 {size} 不符标定。')
                self.pixel_formats[c] = stream.codec_context.format.name if stream.codec_context.format else None
                for packet in container.demux(stream):
                    if self.stop_event.is_set(): break
                    if packet.pts is None or not packet.size: continue
                    timestamp = int(packet.pts*packet.time_base*1_000_000_000)
                    if timestamp < 0: raise ValueError('源时间戳为负，请切换时钟模式后重新校时。')
                    received = time.perf_counter_ns(); payload = bytes(packet)
                    if self.pixel_formats[c] is None: self.pixel_formats[c] = jpeg_pixel_format(payload)
                    frame = Frame(c, self.counts[c], timestamp, received, payload)
                    self.counts[c] += 1; self.on_frame(frame)
            if not self.stop_event.is_set(): raise RuntimeError('摄像头流结束。'+' / '.join(self.stderr[c][-3:]))
        except Exception as exc:
            if not self.stop_event.is_set(): self.on_error(f'cam{c}: {exc}')
        finally:
            if p and p.poll() is None: p.terminate()

    def stop(self):
        self.stop_event.set()
        for p in list(self.processes):
            if p.poll() is None: p.terminate()
        for t in self.threads: t.join(timeout=3)
        for p in list(self.processes):
            if p.poll() is None: p.kill()


def demo_calibration():
    k = [[350., 0, 320.], [0, 350., 180.], [0, 0, 1.]]
    return dict(schema='stereo_calibration/1.0', unit='m', calibration_id='SYNTHETIC_NOT_MEASURED',
                camera_ids=['demo-left', 'demo-right'], image_size=[640, 360], K=[k, k],
                distortion=[[0]*5, [0]*5], R_cam0_to_cam1=np.eye(3).tolist(),
                t_cam0_to_cam1_m=[-.6, 0, 0],
                world_from_cam0=[[0, 0, 1, -3], [-1, 0, 0, 0], [0, -1, 0, 1.4], [0, 0, 0, 1]])


def demo_views(timestamp_ns, cal):
    from mocap.samples import standing_coco
    p = standing_coco(); angle = np.sin(timestamp_ns/1e9*1.5)*.6
    p[7] = p[5]+[0, .29*np.cos(angle), .29*np.sin(angle)]
    p[9] = p[7]+[0, .26*np.cos(angle), .26*np.sin(angle)]
    camera = (np.linalg.inv(cal['world_from_cam0']) @ np.column_stack([p, np.ones(17)]).T).T[:, :3]
    views = []
    for c in range(2):
        xyz = camera + (np.array(cal['t_cam0_to_cam1_m']) if c else 0)
        xy = xyz @ np.array(cal['K'][c]).T; xy = xy[:, :2]/xy[:, 2:]
        views.append(dict(image_size=[640, 360], person_count=1, status='synthetic',
                          joints=[dict(xy=a.tolist(), confidence=1., valid=True, estimated=True) for a in xy]))
    return views


class DemoSource:
    def __init__(self, config, on_frame, on_error):
        self.config, self.on_frame, self.on_error = config, on_frame, on_error
        self.stop_event = threading.Event(); self.counts = [0, 0]; self.pixel_formats = ['yuvj420p']*2
        self.stderr = [[], []]

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()

    def _run(self):
        try:
            cal = demo_calibration(); start = time.perf_counter(); index = 0
            while not self.stop_event.is_set():
                elapsed = index/self.config.fps; timestamp = 1_000_000_000+round(elapsed*1e9)
                views = demo_views(timestamp, cal)
                flash = any(a <= elapsed % 5 < a+.12 for a in (.6, 1.5, 2.8))
                for c in range(2):
                    image = np.full((360, 640, 3), 220 if flash else 35, np.uint8)
                    for joint in views[c]['joints']: cv2.circle(image, tuple(np.int32(joint['xy'])), 4, (0, 230, 0), -1)
                    payload = cv2.imencode('.jpg', image)[1].tobytes()
                    self.on_frame(Frame(c, index, timestamp+c*300_000_000, time.perf_counter_ns(), payload))
                    self.counts[c] += 1
                index += 1
                time.sleep(max(0, start+index/self.config.fps-time.perf_counter()))
        except Exception as exc: self.on_error(str(exc))

    def stop(self):
        self.stop_event.set(); self.thread.join(timeout=3)
