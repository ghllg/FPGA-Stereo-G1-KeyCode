"""Bounded, one-to-one source timestamp pairing and shared visual pulse alignment."""
from collections import deque
from dataclasses import dataclass
import threading
import time
import numpy as np


@dataclass
class Frame:
    camera: int
    index: int
    timestamp_ns: int
    receive_ns: int
    payload: bytes
    recording_index: int | None = None
    recording_id: str | None = None


class PairBuffer:
    def __init__(self, capacity=12, max_age_ms=150, byte_limit=16*1024*1024):
        self.lock = threading.Lock()
        self.buffers = [deque(), deque()]
        self.last = [-1, -1]
        self.dropped = [0, 0]
        self.capacity, self.age_ns, self.byte_limit = capacity, int(max_age_ms*1e6), byte_limit

    def add(self, frame):
        with self.lock:
            c = frame.camera
            if frame.timestamp_ns <= self.last[c]:
                raise ValueError(f'cam{c} 时间戳回退；停止流程，重新打开并校时。')
            self.last[c] = frame.timestamp_ns
            q = self.buffers[c]; q.append(frame)
            total = sum(len(x.payload) for x in q)
            while len(q) > self.capacity or total > self.byte_limit:
                total -= len(q.popleft().payload); self.dropped[c] += 1

    def clear(self):
        with self.lock:
            for c, q in enumerate(self.buffers):
                self.dropped[c] += len(q); q.clear()

    def latest(self, right_offset_ns=0, tolerance_ms=10, now_ns=None):
        now_ns = time.perf_counter_ns() if now_ns is None else now_ns
        with self.lock:
            for c, q in enumerate(self.buffers):
                while q and now_ns-q[0].receive_ns > self.age_ns:
                    q.popleft(); self.dropped[c] += 1
            # Choose newest matching pair, then consume all earlier entries.
            # No frame can participate twice, even when the inference thread is slow.
            candidates = [(min(a.timestamp_ns, b.timestamp_ns-right_offset_ns), i, j)
                          for i, a in enumerate(self.buffers[0])
                          for j, b in enumerate(self.buffers[1])
                          if abs(b.timestamp_ns-right_offset_ns-a.timestamp_ns) <= tolerance_ms*1e6]
            if not candidates: return None
            _, i, j = max(candidates)
            selected = []
            for c, n in enumerate((i, j)):
                for _ in range(n): self.buffers[c].popleft(); self.dropped[c] += 1
                selected.append(self.buffers[c].popleft())
            return selected


def align_pulses(histories, max_spread_ms=20):
    """Use at least three nonperiodic bright rising edges, never align first frames."""
    peaks = []
    for history in histories:
        if len(history) < 20: raise ValueError('校时样本不足；保持画面稳定，做至少三次共同闪光。')
        times, values = np.array(history, dtype=float).T
        diff = np.diff(values)
        threshold = max(12., float(np.median(diff)+8*np.median(abs(diff-np.median(diff)))))
        chosen = []
        for i in np.flatnonzero(diff > threshold)+1:
            t = int(times[i])
            if not chosen or t-chosen[-1] > 180_000_000: chosen.append(t)
        if len(chosen) < 3: raise ValueError('共同闪光不足三次；白纸需在两路画面中央区域同时可见。')
        peaks.append(chosen)
    if len(peaks[0]) != len(peaks[1]): raise ValueError('两路闪光次数不同，请重新校时，避免错误配对。')
    gaps = np.diff(peaks[0])/1e6
    if np.ptp(gaps) < 100: raise ValueError('闪光间隔过于均匀，请用长短不同的间隔重试以消除歧义。')
    offsets = np.array(peaks[1], dtype=np.int64)-np.array(peaks[0], dtype=np.int64)
    offset = int(np.median(offsets)); spread = float(max(abs(offsets-offset))/1e6)
    if spread > max_spread_ms: raise ValueError(f'两路校时偏移不稳定（{spread:.1f} ms），请重试。')
    return dict(method='shared_brightness_rising_edges', right_offset_ns=offset,
                pulse_count=len(offsets), max_residual_ms=spread, peaks_ns=peaks,
                hardware_synchronized=False, exposure_timestamp_verified=False,
                note='Subtract right_offset_ns from right source timestamps; frame/rolling-shutter limited visual check.')
