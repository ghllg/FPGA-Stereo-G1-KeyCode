"""Optional bounded native physics/render telemetry; enabled per test process."""
import json
import os
from pathlib import Path
import queue
import threading

_folder = os.environ.get('G1_PERFORMANCE_OUTPUT')
ENABLED = bool(_folder)
_queue = queue.Queue(maxsize=4096) if _folder else None


def _write():
    folder = Path(_folder)
    try:
        with (folder/'native_events.ndjson').open('x', encoding='utf-8', buffering=1) as stream:
            while True:
                event = _queue.get()
                stream.write(json.dumps(event, separators=(',', ':'), allow_nan=False)+'\n')
    except Exception as error:
        (folder/'native_trace_error.txt').write_text(repr(error), encoding='utf-8')


def emit(kind, **event):
    if _queue is None:
        return
    try:
        _queue.put_nowait(dict(kind=kind, **event))
    except queue.Full:
        (Path(_folder)/'native_trace_error.txt').write_text('Telemetry queue overflow', encoding='utf-8')


if _queue is not None:
    threading.Thread(target=_write, name='performance_trace', daemon=True).start()
