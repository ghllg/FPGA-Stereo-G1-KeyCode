"""Saved FPGA packets at their original rate into the same monitored PC backend."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backend import Backend, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--name', required=True)
    parser.add_argument('--lag-ms', type=int, default=380)
    parser.add_argument('--speed', type=float, default=1.)
    args = parser.parse_args()
    if args.speed not in (1., 2.): raise ValueError('Only declared 1x or 2x throughput tests')
    out = ROOT/'outputs/full_pipeline_exec_20261022'/args.name
    out.mkdir(exist_ok=False)
    source = args.source.resolve()
    records = [json.loads(x) for x in (source/'pc_receiver/received.ndjson').read_text().splitlines()]
    binding = json.loads((source/'audit/binding.json').read_text())
    scope = dict(mode='paced_saved_replay', source=str(source), cameras_opened=False,
        raw_sha256=hashlib.sha256((source/'pc_receiver/received.ndjson').read_bytes()).hexdigest(),
        source_timestamps_preserved=True, pc_receipts_reissued=True, live_camera_latency_measured=False,
        lag_ms=args.lag_ms, replay_speed=args.speed)
    (out/'scope.json').write_text(json.dumps(scope, indent=2), encoding='utf-8')
    backend = None
    try:
        backend = Backend(out, binding, args.lag_ms)
        origin = records[0]['message']['cameras'][0]['capture_timestamp_us']
        start = time.perf_counter_ns()
        with (out/'replayed.ndjson').open('x', encoding='utf-8', buffering=1) as log:
            for raw in records:
                deadline = start+int((raw['message']['cameras'][0]['capture_timestamp_us']-origin)*1000/args.speed)
                time.sleep(max(0, (deadline-time.perf_counter_ns())/1e9))
                row = copy.deepcopy(raw)
                row['pc_receive_ns'] = time.perf_counter_ns()
                row['replay'] = True
                log.write(json.dumps(row, separators=(',', ':'))+'\n')
                backend.submit(row)
        backend.finish()
    finally:
        if backend: backend.close()
    from metrics import analyze
    report = analyze(out)
    print(json.dumps(report['headline'], ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
