from __future__ import annotations

import bisect
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / 'demo_station'
ARCHIVE = ROOT / 'pose3d_train/handoff_20261023'
RUNS = ARCHIVE / 'workspace/outputs/full_pipeline_exec_20261022/live_runs'
TRAIN = ARCHIVE / 'workspace/outputs/training_1080p_20261021'
COCO_EDGES = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),
              (11,12),(11,13),(13,15),(12,14),(14,16),(0,1),(0,2),(1,3),(2,4)]


if os.name=='nt':
    # Reuse native wrappers: capture defers cyclic GC, so per-call ctypes types accumulate.
    import ctypes
    from ctypes import wintypes
    import msvcrt
    _kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    _create_file=_kernel.CreateFileW
    _create_file.argtypes=(wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,
                     wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE)
    _create_file.restype=wintypes.HANDLE
    _rename_file=_kernel.SetFileInformationByHandle
    _rename_file.argtypes=(wintypes.HANDLE,ctypes.c_int,wintypes.LPVOID,wintypes.DWORD)
    _rename_file.restype=wintypes.BOOL
    _close_handle=_kernel.CloseHandle
    _close_handle.argtypes=(wintypes.HANDLE,);_close_handle.restype=wintypes.BOOL
    class _RenameInfo(ctypes.Structure):
        _fields_=[('Flags',wintypes.DWORD),('RootDirectory',wintypes.HANDLE),
                  ('FileNameLength',wintypes.DWORD),('FileName',wintypes.WCHAR*1)]


def _open_state_reader(path):
    if os.name != 'nt':return Path(path).open('r',encoding='utf-8-sig')
    # READ | WRITE | DELETE sharing lets an atomic publisher replace the path
    # while this reader finishes reading the previous complete file.
    handle=_create_file(str(Path(path).absolute()),0x80000000,7,None,3,0x80,None)
    if handle == wintypes.HANDLE(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:descriptor=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    except Exception:
        _close_handle(handle)
        raise
    return os.fdopen(descriptor,'r',encoding='utf-8-sig')


def read_json(path):
    for attempt in range(3):
        try:
            with _open_state_reader(path) as stream:return json.load(stream)
        except (PermissionError,FileNotFoundError):
            if attempt==2:raise
            time.sleep(.005)


def _replace_state_file(temporary,path):
    if os.name!='nt':return temporary.replace(path)
    name=str(Path(path).absolute()).encode('utf-16-le')
    buffer=ctypes.create_string_buffer(ctypes.sizeof(_RenameInfo)+len(name))
    info=_RenameInfo.from_buffer(buffer);info.Flags=3;info.FileNameLength=len(name)
    ctypes.memmove(ctypes.addressof(buffer)+_RenameInfo.FileName.offset,name,len(name))
    handle=_create_file(str(Path(temporary).absolute()),0x10000,7,None,3,0x80,None)
    if handle==wintypes.HANDLE(-1).value:raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not _rename_file(handle,22,buffer,len(buffer)):
            raise ctypes.WinError(ctypes.get_last_error())
    finally:_close_handle(handle)

def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    descriptor, name = tempfile.mkstemp(prefix=path.name+'.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(content)
        for attempt in range(5):
            try:
                _replace_state_file(temporary,path)
                break
            except PermissionError:
                if attempt == 4: raise
                time.sleep(.005 * (attempt + 1))
    finally:
        try:
            if path.exists():temporary.unlink(missing_ok=True)
        except OSError: pass  # Cleanup must not hide a failed state publication.


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    result = []
    with path.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    result.append(json.loads(line))
                except ValueError as e:
                    raise ValueError(f'{path.name} 第 {number} 行损坏') from e
    return result


class JsonlTail:
    """Keep incomplete trailing bytes until the producer finishes its line."""
    def __init__(self, path, *, persistent=False):
        self.path, self.offset, self.pending = Path(path), 0, b''
        self.persistent=persistent;self._reader=None;self._identity=None

    def close(self):
        if self._reader:self._reader.close();self._reader=None;self._identity=None

    def poll(self):
        try:info=self.path.stat();size=info.st_size
        except FileNotFoundError:
            return []
        if size < self.offset:
            raise ValueError('数据文件被截断；请开始新会话')
        if self._reader and self._identity!=(info.st_dev,info.st_ino):
            raise ValueError('数据文件被替换；请开始新会话')
        if size == self.offset:return []
        if self.persistent:
            if self._reader is None:
                self._reader=self.path.open('rb',buffering=0);self._reader.seek(self.offset)
                opened=os.fstat(self._reader.fileno());self._identity=(opened.st_dev,opened.st_ino)
                if self._identity!=(info.st_dev,info.st_ino):
                    self.close();raise ValueError('数据文件被替换；请开始新会话')
            data=self._reader.read(min(8*1024*1024,size-self.offset));self.offset+=len(data)
        else:
            with self.path.open('rb') as f:
                f.seek(self.offset)
                data = f.read(min(8*1024*1024,size-self.offset))
                self.offset = f.tell()
        parts = (self.pending+data).split(b'\n')
        self.pending = parts.pop()
        if len(self.pending) > 2*1024*1024:
            raise ValueError('单条记录超过限制')
        return [json.loads(x) for x in parts if x.strip()]


@dataclass
class Track:
    times: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def add(self, stamp, value):
        stamp = float(stamp)
        if not math.isfinite(stamp):
            raise ValueError('非有限时间戳')
        if self.times and stamp < self.times[-1]:
            raise ValueError('时间戳倒退，不能拼接为同一会话')
        self.times.append(stamp)
        self.values.append(value)

    def at(self, stamp, max_age=.25):
        i = bisect.bisect_right(self.times, stamp+1e-9)-1
        if i < 0 or stamp-self.times[i] > max_age:
            return None
        return self.values[i]


class Timeline:
    def __init__(self):
        self.position = 0.
        self.revision = 0
        self.duration = 0.
        self.speed = 1.
        self.playing = False
        self.last = time.monotonic()

    def seek(self, value):
        self.position = min(self.duration, max(0., float(value)))
        self.revision += 1
        self.last = time.monotonic()

    def tick(self):
        now = time.monotonic()
        if self.playing:
            self.position = min(self.duration, self.position+(now-self.last)*self.speed)
            if self.position >= self.duration:
                self.playing = False
        self.last = now
        return self.position


class Session:
    def __init__(self, path):
        self.path = Path(path).resolve()
        if self.path.is_file():
            self.path = self.path.parent
        self.manifest = read_json(self.path/'session.json') if (self.path/'session.json').exists() else {}
        self.name = self.manifest.get('name', self.path.name)
        self.paired, self.skeleton, self.robot = Track(), Track(), Track()
        self.refined = Track()
        self.benchmarks = {'cpu': Track(), 'pl-stream': Track()}
        self.images = {'left': Track(), 'right': Track(), 'overview': Track()}
        self.origin_ns = None
        self.issues = []
        self.comparison = None
        self.duration = 0.
        self._load()

    def resolve(self, value):
        p = Path(value)
        return p if p.is_absolute() else self.path/p

    def stream_path(self, key, default):
        return self.resolve(self.manifest.get('streams', {}).get(key, default))

    def _load(self):
        received = read_lines(self.stream_path('paired', 'pc_receiver/received.ndjson'))
        if received:
            self.origin_ns = received[0]['message']['cameras'][0]['capture_timestamp_us']*1000
        if self.manifest.get('origin_ns') is not None:
            self.origin_ns = int(self.manifest['origin_ns'])
        skeletons = read_lines(self.stream_path('skeleton', 'online_3d/skeleton3d.jsonl'))
        if self.origin_ns is None and skeletons:
            self.origin_ns = skeletons[0]['timestamp_ns']
        for row in received:
            stamp = row['message']['cameras'][0]['capture_timestamp_us']*1000
            self.paired.add((stamp-self.origin_ns)/1e9, row)
        for row in skeletons:
            self.skeleton.add((row['timestamp_ns']-self.origin_ns)/1e9, row)
        if self.manifest.get('hybrid_refinement'):
            for row in read_lines(self.stream_path('refined','online_3d/pose2d.jsonl')):
                self.refined.add((row['timestamp_ns']-self.origin_ns)/1e9,row)
        report_path = self.stream_path('robot_report', 'g1/report.json')
        report = read_json(report_path) if report_path.exists() else {}
        if not report and (self.path/'g1/profile.json').exists():
            report=read_json(self.path/'g1/profile.json')
        robot_origin = report.get('source_origin_ns', self.manifest.get('robot_source_origin_ns'))
        robots = read_lines(self.stream_path('robot', 'g1/poses.ndjson'))
        if robots and (robot_origin is None or self.origin_ns is None):
            self.issues.append('G1 缺少源时间锚点，未与视频强行对齐')
        else:
            for row in robots:
                self.robot.add((robot_origin-self.origin_ns)/1e9+row['source_time'], row)
        self.issues.extend(report.get('errors', []))
        for side in ('left', 'right', 'overview'):
            entries = self.manifest.get('media', {}).get(side, {})
            if entries.get('index'):
                for row in read_lines(self.resolve(entries['index'])):
                    self.images[side].add(row['time_s'], row)
        for backend,track in self.benchmarks.items():
            for row in read_lines(self.path/(backend+'.ndjson')):
                stamp=row['cameras'][0]['capture_timestamp_us']*1000
                if self.origin_ns is None:self.origin_ns=stamp
                track.add((stamp-self.origin_ns)/1e9,row)
        cmp_path = self.path/'comparison.json'
        if cmp_path.exists():
            self.comparison = read_json(cmp_path)
        tracks = [self.paired, self.skeleton, self.robot, *self.images.values(),*self.benchmarks.values()]
        self.duration = max([t.times[-1] for t in tracks if t.times]+[float(self.manifest.get('duration_s',0))])

    def frame(self, stamp, backend='pl-stream'):
        return {'paired': self.paired.at(stamp), 'skeleton': self.skeleton.at(stamp), 'refined': self.refined.at(stamp),
                'robot': self.robot.at(stamp, .12), 'benchmark':self.benchmarks[backend].at(stamp),
                'images': {k:v.at(stamp,.2) for k,v in self.images.items()}}

    def verify_assets(self):
        results = []
        for name, item in self.manifest.get('assets', {}).items():
            p = self.resolve(item['path'])
            ok = p.is_file() and digest(p)==item.get('sha256')
            results.append((name, ok, str(p)))
        return results


def refined_display_cameras(row):
    """A UI view of PC-refined points; never a substitute board wire message."""
    return [dict(camera_id=view['camera_id'],source_size=view['image_size'],roi_xywh=view.get('roi_xywh'),
        roi_label='FPGA 初定位范围',source_2d_model=row['model_id'],
        keypoints=[[*((j['xy'] or [0.,0.])),j['confidence'],j['valid']] for j in view['joints']])
        for view in row['views']]


def milliseconds(value):
    return '—' if value is None or not math.isfinite(value) else f'{value:.1f} ms'


class StageTimingAverage:
    """Publish two-second means of distinct records sampled by the UI."""
    WINDOW_S = 2.
    KEYS = ('inference', 'reconstruction', 'mapping')

    def __init__(self):
        self.context = object()
        self.running = False
        self.values = dict.fromkeys(self.KEYS)
        self.last_records = {}

    def begin_window(self, now):
        self.started = now
        self.sums = dict.fromkeys(self.KEYS, 0.)
        self.counts = dict.fromkeys(self.KEYS, 0)

    def update(self, frame, metrics, context, running, now=None):
        now = time.monotonic() if now is None else now
        if context != self.context:
            self.context = context
            self.values = dict.fromkeys(self.KEYS)
            self.running = False
        if not running:
            self.running = False
            return self.values
        if not self.running:
            self.begin_window(now)
            self.last_records = {}
            self.running = True
        if now-self.started >= self.WINDOW_S:
            elapsed_windows = int((now-self.started)//self.WINDOW_S)
            self.values = {key: self.sums[key]/self.counts[key] if elapsed_windows == 1 and self.counts[key] else None for key in self.KEYS}
            self.begin_window(self.started+elapsed_windows*self.WINDOW_S)
        records = (frame.get('paired') or frame.get('benchmark'), frame.get('skeleton'), frame.get('robot'))
        for key, record in zip(self.KEYS, records):
            if record is None or record is self.last_records.get(key):continue
            self.last_records[key] = record
            value = metrics.get(key)
            if value is not None and math.isfinite(value) and value >= 0:
                self.sums[key] += value
                self.counts[key] += 1
        return self.values


def stage_metrics(frame,display_ns=None):
    paired, skeleton, robot = (frame.get(k) for k in ('paired','skeleton','robot'))
    result = dict(inference=None, reconstruction=None, mapping=None, board_age=None, e2e_upper=None)
    if paired:
        m = paired['message']; cams = m['cameras']
        result['inference'] = (max(c['infer_end_ns'] for c in cams)-min(c['infer_start_ns'] for c in cams))/1e6
        result['board_age'] = (m['publish_monotonic_ns']-min(c['capture_timestamp_us'] for c in cams)*1000)/1e6
        bounds=paired.get('latency')
        if bounds:
            finish=display_ns if display_ns is not None else paired['pc_receive_ns']
            capture=min(c['capture_timestamp_us'] for c in cams)*1000
            result['e2e_lower']=max(0,(finish+bounds['offset_min_ns']-capture)/1e6)
            result['e2e_upper']=max(0,(finish+bounds['offset_max_ns']-capture)/1e6)
    elif frame.get('benchmark'):
        result['inference']=frame['benchmark']['inference_ms']
    if skeleton:
        result['reconstruction'] = skeleton.get('compute_ms')
    if robot:
        result['mapping'] = robot.get('solve_ms')
    return result


def summarize_comparison(cpu, fpga, identity):
    for value, backend in ((cpu,'cpu'),(fpga,'pl-stream')):
        if value.get('backend') != backend or not value.get('all_full_heatmaps_exact'):
            raise ValueError('对比结果后端或数值验证不匹配')
        if value.get('measured_pairs',0)<2 or value.get('processing_pairs_per_second',0)<=0:
            raise ValueError('对比样本不足')
        calls=value.get('measured_pl_calls',0)
        if (backend=='cpu' and calls!=0) or (backend=='pl-stream' and calls<=0):
            raise ValueError('实际 PL 调用数与后端不符')
    for key in ('scope','measured_pairs','warmup_pairs','inflight_pairs','threads_per_network'):
        if cpu.get(key)!=fpga.get(key):
            raise ValueError('对比条件不一致: '+key)
    return dict(schema='motion_comparison/1', kind='measured', scope=cpu['scope'],
                input_identity=identity, cpu=cpu, fpga=fpga,
                throughput_speedup=fpga['processing_pairs_per_second']/cpu['processing_pairs_per_second'],
                acquisition_latency_measured=False)
