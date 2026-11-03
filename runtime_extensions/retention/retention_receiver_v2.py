"""Bounded raw-row retention, incremental identity proof and compact clock stats."""
import hashlib
import json
import statistics
import struct
import sys

import retention_receiver as v1
from retention_parser_v2 import run


class JournalRows(v1.JournalRows):
    def __init__(self, path):
        super().__init__(path)
        self.persisted_count = 0
        self.persistence_gap = False
        self.awaiting_persist = False
        self._identity_hash = hashlib.sha256(b'[')
        self._last_identity = None
        self.interval_count, self.interval_sum, self.probe_count = 0, 0., 0
        self._clock_file = None
        self._clock_path = self.path.with_name('receiver_clock_samples.f64le')

    def append(self, row):
        if self.awaiting_persist:
            self.persistence_gap = True
        count, last = self._state
        current_identity = [row['message']['session_id'], row['message']['pair_id'],
            [camera['capture_timestamp_us'] for camera in row['message']['cameras']]]
        if count:
            self._identity_hash.update(b',')
        self._identity_hash.update(json.dumps(current_identity, separators=(',', ':')).encode())
        self._last_identity = current_identity
        if last is not None and last['connection'] == row['connection']:
            self.interval_count += 1
            self.interval_sum += (row['pc_receive_ns']-last['pc_receive_ns'])/1e9
        super().append(row)
        self.awaiting_persist = True
        if row.get('latency'):
            try:
                if self._clock_file is None:
                    self._clock_file = self._clock_path.open('wb')
                self._clock_file.write(struct.pack('<dd', row['latency']['estimated_ms'], row['latency']['uncertainty_ms']))
                self.probe_count += 1
            except Exception as error:
                self.persistence_gap = True
                raise RuntimeError('Clock-stat sidecar write failed') from error

    def mark_persisted(self):
        if not self.awaiting_persist:
            raise RuntimeError('Unexpected duplicate received-log flush confirmation')
        self.persisted_count += 1
        self.awaiting_persist = False

    def identity_snapshot(self):
        digest = self._identity_hash.copy(); digest.update(b']')
        complete = not self.persistence_gap and not self.awaiting_persist and self.persisted_count == len(self)
        return dict(received_count=len(self), persisted_count=self.persisted_count,
            last_identity=self._last_identity, identity_sequence_sha256=digest.hexdigest(),
            persistence_complete=complete,
            persistence_error=None if complete else 'Received-log write/flush confirmation missing; historical gap cannot be cleared by later success')

    def finish_statistics(self):
        if self._clock_file is not None:
            self._clock_file.close()
            self._clock_file = None
        snapshot = self.identity_snapshot()
        if not snapshot['persistence_complete']:
            raise RuntimeError(snapshot['persistence_error'])
        estimated, uncertainty = [], []
        if self.probe_count:
            with self._clock_path.open('rb') as stream:
                for _ in range(self.probe_count):
                    raw = stream.read(16)
                    if len(raw) != 16:
                        raise RuntimeError('Clock-stat sidecar is incomplete')
                    a, b = struct.unpack('<dd', raw)
                    estimated.append(a); uncertainty.append(b)
                if stream.read(1):
                    raise RuntimeError('Clock-stat sidecar contains extra data')
        return dict(receive_pairs_per_s=self.interval_count/self.interval_sum if self.interval_count else 0,
            network_delay_estimate_ms_median=statistics.median(estimated) if estimated else None,
            clock_uncertainty_ms_median=statistics.median(uncertainty) if uncertainty else None,
            rows_with_clock_probe=self.probe_count)


class Receiver(v1.Receiver.__bases__[0]):
    run = run

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = JournalRows(self.output/'received.ndjson')

    def finish(self):
        self.stop.set(); self.thread.join(4)
        assert not self.thread.is_alive(), 'Receiver did not stop'
        result = dict(messages=len(self.rows), events=self.events,
            latency_scope='Publish to PC complete JSON receipt, midpoint of four-timestamp clock bounds; includes serialization/socket/assembly; not exposure latency')
        try:
            result.update(self.rows.finish_statistics())
        except Exception as error:
            result['summary_error'] = repr(error)
            if not self.error:
                self.error = repr(error)
        if self.error:
            result['error'] = self.error
        (self.output/'receiver_summary.json').write_text(json.dumps(result, indent=2)+'\n')
        if self.error:
            raise RuntimeError(self.error)
        return result


def activate():
    if v1.original.Receiver is Receiver:
        return Receiver
    if 'live_receiver' in sys.modules:
        raise RuntimeError('Activate retention before importing live_receiver/backend')
    v1.original.Receiver = Receiver
    return Receiver
