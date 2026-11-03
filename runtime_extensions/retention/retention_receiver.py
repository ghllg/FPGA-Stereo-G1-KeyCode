"""Isolated received-row retention change; the original wire parser is inherited."""
import itertools
import json
from pathlib import Path
import statistics
import sys

PC = Path(r'C:\Users\developer\Documents\Codex\2026-10-20\iq\work\full_pipeline_exec_20261022')
sys.path.insert(0, str(PC))
import performance_receiver as original


class JournalRows:
    """Read-only history after journal flush; append is only for Receiver.run.

    len() and [-1] are O(1). Iteration and other indexes read received.ndjson.
    Do not read history inside on_record: the original parser logs AFTER it.
    Explicit slices materialize the requested history, as the old list did.
    """
    def __init__(self, path):
        self.path = Path(path)
        self._state = (0, None)

    def append(self, row):
        self._state = (self._state[0] + 1, row)

    def __len__(self):
        return self._state[0]

    def __iter__(self):
        expected, _ = self._state
        if not expected:
            return
        seen = 0
        with self.path.open(encoding='utf8') as stream:
            for line in stream:
                if seen == expected:
                    break
                yield json.loads(line)
                seen += 1
        if seen != expected:
            raise RuntimeError('Received history requires a flushed complete journal; do not read it in on_record')

    def __getitem__(self, key):
        count, last = self._state
        if isinstance(key, slice):
            start, stop, step = key.indices(count)
            if step < 0:
                return list(self)[key]
            return list(itertools.islice(iter(self), start, stop, step))
        if not isinstance(key, int):
            raise TypeError('Received history indices must be integers or slices')
        if key < 0:
            key += count
        if not 0 <= key < count:
            raise IndexError('Received history index out of range')
        if key == count - 1:
            return last
        return next(itertools.islice(iter(self), key, key + 1))


class Receiver(original.Receiver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rows = JournalRows(self.output / 'received.ndjson')

    def finish(self):
        self.stop.set()
        self.thread.join(4)
        assert not self.thread.is_alive(), 'Receiver did not stop'
        previous = None
        intervals_count = 0
        intervals_sum = 0.
        estimated, uncertainty = [], []
        history_error = None
        try:
            for row in self.rows:
                if previous is not None and previous['connection'] == row['connection']:
                    intervals_count += 1
                    intervals_sum += (row['pc_receive_ns'] - previous['pc_receive_ns']) / 1e9
                previous = row
                if row.get('latency'):
                    estimated.append(row['latency']['estimated_ms'])
                    uncertainty.append(row['latency']['uncertainty_ms'])
        except Exception:
            # Preserve the original parser error if persistence itself failed.
            if not self.error:
                raise
            history_error = 'Summary history incomplete after receiver error'
        result = dict(messages=len(self.rows), events=self.events,
                      receive_pairs_per_s=intervals_count / intervals_sum if intervals_count else 0,
                      network_delay_estimate_ms_median=statistics.median(estimated) if estimated else None,
                      clock_uncertainty_ms_median=statistics.median(uncertainty) if uncertainty else None,
                      latency_scope='Publish to PC complete JSON receipt, midpoint of four-timestamp clock bounds; includes serialization/socket/assembly; not exposure latency',
                      rows_with_clock_probe=len(estimated))
        if self.error:
            result['error'] = self.error
            if history_error:
                result['summary_error'] = history_error
        (self.output / 'receiver_summary.json').write_text(json.dumps(result, indent=2) + '\n')
        if self.error:
            raise RuntimeError(self.error)
        return result


def activate():
    """Bind before backend/live_receiver imports; never reload a running pipeline."""
    if original.Receiver is Receiver:
        return Receiver
    if 'live_receiver' in sys.modules:
        raise RuntimeError('Activate retention before importing live_receiver/backend')
    original.Receiver = Receiver
    return Receiver
