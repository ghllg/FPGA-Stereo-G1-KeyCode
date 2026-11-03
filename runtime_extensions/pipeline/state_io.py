"""Bounded retry for the observed Windows sharing race during atomic state replacement."""
import json
from pathlib import Path
import time


def sharing_retry(action):
    deadline=time.perf_counter()+.25
    while True:
        try:return action()
        except PermissionError:
            if time.perf_counter()>=deadline:raise
            time.sleep(.005)


def read_json(path):
    return sharing_retry(lambda:json.loads(Path(path).read_text(encoding='utf8')))


def write_json(path,value):
    path=Path(path);pending=path.with_suffix('.tmp')
    pending.write_text(json.dumps(value),encoding='utf8')
    sharing_retry(lambda:pending.replace(path))
