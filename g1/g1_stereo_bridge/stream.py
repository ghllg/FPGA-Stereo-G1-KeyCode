"""JSONL (or growing receiver log) -> causal BVH -> existing platform WebSocket."""
import argparse
import asyncio
from collections import deque
import json
from pathlib import Path
import time
import sys

import aiohttp
from conversion import LiveBVH, PairedStereo
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'g1_physical'))
from windows_timing import enable_timer


async def run(args):
    close_timer = enable_timer()
    try:
        return await run_stream(args)
    finally:
        close_timer()


async def run_stream(args):
    if args.output.exists():
        raise ValueError('Use a new output directory; previous results are preserved')
    args.output.mkdir(parents=True)
    stereo = PairedStereo(args.calibration, args.camera_ids, args.allow_diagnostic) if args.calibration else None
    converter = LiveBVH(args.warmup, getattr(args, 'warmup_seconds', 0.))
    stats = dict(input=str(args.input), url=args.url, mode='live_tail' if args.tail else 'timestamped_replay',
                 source_fps_limit=args.fps, errors=[], output_frames=0, fallen=False,
                 source_contains_estimates=False, adapter=None, first_failure_source_time=None,
                 simulation_time=0, initial_floor_offset=None, minimum_root_height=None,
                 stale_source_frames_dropped=0, source_records_read=0, sent_bvh_frames=0)
    costs, acks, arrivals, ages, fits = [deque(maxlen=18000) for _ in range(5)]
    started = last_source = first_source = last_slot = None
    done, ready, ack = asyncio.Event(), asyncio.Event(), asyncio.Event()
    input_name = 'COCO17 live bridge: '+args.input.name
    last_frame = None
    first_sent_source = None
    last_sent_wall = None
    async with aiohttp.ClientSession() as client, client.ws_connect(args.url) as ws:
        hello = await ws.receive_json(timeout=10)
        if hello.get('busy'):
            raise ValueError('Platform input is busy: stop current browser/source first')
        async def receive():
            async for message in ws:
                p = json.loads(message.data)
                if p['type'] == 'ready' and p.get('name') == input_name:
                    stats['adapter'] = p.get('adapter'); ready.set()
                elif p['type'] == 'ack':
                    acks.append(p['ik_ms']); ack.set()
                elif p['type'] == 'pose':
                    arrivals.append(time.perf_counter())
                    stats['output_frames'] += 1
                    stats['simulation_time'] = p.get('simulation_time')
                    stats['initial_floor_offset'] = p.get('initial_floor_offset')
                    height = p['root_pos'][2]
                    stats['minimum_root_height'] = height if stats['minimum_root_height'] is None else min(height,stats['minimum_root_height'])
                    if p.get('client_time') is not None:
                        ages.append(time.perf_counter()*1000-p['client_time'])
                    poses.write(json.dumps(p, separators=(',', ':'))+'\n')
                    if p.get('fallen') or p.get('finished'):
                        stats['fallen'] = bool(p.get('fallen'))
                        if stats['fallen']: stats['first_failure_source_time'] = p['source_time']
                        done.set(); ack.set()
                elif p['type'] == 'error':
                    stats['errors'].append(p['message']); done.set(); ready.set(); ack.set()
        with (args.output/'poses.ndjson').open('x', encoding='utf8') as poses, (args.output/'bvh_frames.ndjson').open('x', encoding='utf8') as frames, args.input.open(encoding='utf-8-sig') as source:
            receiver = asyncio.create_task(receive())
            try:
                last_line = time.perf_counter()
                while not done.is_set():
                    if args.tail and time.perf_counter()-last_line > args.timeout:
                        raise TimeoutError('Live input stopped; no complete fresh record within timeout')
                    if last_sent_wall is not None and time.perf_counter()-last_sent_wall > args.timeout:
                        raise TimeoutError('No valid skeleton within timeout; stop stale tracking')
                    position = source.tell()
                    line = source.readline(65537)
                    if len(line) > 65536:
                        raise ValueError('Source JSONL exceeds bounded size')
                    if not line:
                        if not args.tail: break
                        if time.perf_counter()-last_line > args.timeout:
                            raise TimeoutError('Live input stopped; ending reference and pausing simulation')
                        await asyncio.sleep(.01); continue
                    if not line.endswith('\n'):
                        if args.tail:
                            source.seek(position)
                            await asyncio.sleep(.01); continue
                    row = json.loads(line)
                    stats['source_records_read'] += 1
                    if args.tail:
                        pc_stamp = row.get('pc_receive_ns') if stereo else row.get('pc_receive_ns', row.get('pc_complete_ns'))
                        if type(pc_stamp) is not int or pc_stamp <= 0 or pc_stamp > time.perf_counter_ns():
                            raise ValueError('Live tail requires a valid same-PC monotonic source timestamp')
                        if (time.perf_counter_ns()-pc_stamp)/1e9 > args.timeout:
                            stats['stale_source_frames_dropped'] += 1
                            continue
                    last_line = time.perf_counter()
                    packet = stereo.push(row) if stereo else row
                    stamp = packet['timestamp_ns']/1e9
                    if first_source is None:
                        first_source, started = stamp, time.perf_counter()
                    elapsed = stamp-first_source
                    if args.seconds and elapsed >= args.seconds: break
                    if not args.tail:
                        await asyncio.sleep(max(0., started+elapsed-time.perf_counter()))
                    slot = int(elapsed*args.fps+1e-7) if args.fps else None
                    if args.fps and slot == last_slot: continue
                    last_slot = slot
                    before = time.perf_counter()
                    frame = converter.push(packet)
                    costs.append((time.perf_counter()-before)*1000)
                    if frame is None: continue
                    if last_source is not None and stamp-last_source > args.timeout:
                        raise ValueError('Valid skeleton gap exceeds limit; refuse stale reference replay')
                    if not ready.is_set():
                        (args.output/'header.bvh').write_text(converter.header+'\n', encoding='ascii')
                        (args.output/'profile.json').write_text(json.dumps(dict(
                            source_identity=converter.identity, source_origin_ns=converter.origin_ns,
                            offsets_m=converter.offsets.tolist(), torso_length_m=converter.torso_length,
                            root_alignment='intrinsic Y +90deg; decode to source world XYZ',
                            unit='cm', up='Y', channels='root XYZ position, 22 intrinsic ZYX rotations'), indent=2))
                        await ws.send_json(converter.init_message(input_name))
                        await asyncio.wait_for(ready.wait(), 25)
                        if stats['errors']: break
                        # Initialization is not part of steady-state acquisition latency.
                        if not args.tail: started = time.perf_counter()-elapsed
                        else: last_line = time.perf_counter()
                    if args.tail and (time.perf_counter_ns()-pc_stamp)/1e9 > args.timeout:
                        stats['stale_source_frames_dropped'] += 1
                        continue
                    frame['client_time'] = time.perf_counter()*1000
                    fits.append(frame.pop('fit_p95_m'))
                    stats['source_contains_estimates'] |= frame.pop('source_estimated')
                    frame.pop('timestamp_ns')
                    logged_frame = dict(frame, source_timestamp_ns=packet['timestamp_ns'], send_pc_ns=time.perf_counter_ns())
                    if args.tail: logged_frame['source_pc_monotonic_ns'] = pc_stamp
                    frames.write(json.dumps(logged_frame)+'\n'); frames.flush()
                    ack.clear()
                    await ws.send_json(frame)
                    stats['sent_bvh_frames'] += 1
                    last_frame = frame
                    if first_sent_source is None: first_sent_source = frame['source_time']
                    await asyncio.wait_for(ack.wait(), args.timeout)
                    last_source = stamp
                    last_sent_wall = time.perf_counter()
                    last_line = last_sent_wall
                if last_source is not None and not done.is_set():
                    await ws.send_json(dict(type='end'))
                    await asyncio.wait_for(done.wait(), 5)
            except Exception as error:
                stats['errors'].append(type(error).__name__+': '+str(error))
                if last_source is not None:
                    await ws.send_json(dict(type='end'))
                    try: await asyncio.wait_for(done.wait(), 5)
                    except asyncio.TimeoutError: pass
            finally:
                await ws.send_json(dict(type='release'))
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
    stats.update(input_frames=converter.received, invalid_core_frames=converter.invalid,
                 bvh_frames=converter.emitted, warmup_valid_frames=converter.warmup_frames_used,
                 warmup_seconds=converter.warmup_seconds,
                 convert_p95_ms=float(np.percentile(costs,95)) if costs else None,
                 ik_p95_ms=float(np.percentile(acks,95)) if acks else None,
                 receive_output_hz=(len(arrivals)-1)/(arrivals[-1]-arrivals[0]) if len(arrivals)>1 else 0,
                 reference_age_p95_ms=float(np.percentile(ages,95)) if ages else None,
                 max_output_gap_ms=float(np.max(np.diff(arrivals))*1000) if len(arrivals)>1 else None,
                 fit_p95_m=float(np.percentile(fits,95)) if fits else None)
    stats['metric_window'] = 'last 18000 samples per metric; full pose/frame records on disk'
    stats['source_origin_ns'] = converter.origin_ns
    stats['source_identity'] = converter.identity
    stats['actual_bvh_source_hz'] = ((stats['sent_bvh_frames']-1)/(last_frame['source_time']-first_sent_source)
                                     if stats['sent_bvh_frames']>1 else 0)
    stats['fit_warning'] = bool(fits and stats['fit_p95_m'] > .08)
    if last_frame is None:
        stats['errors'].append('No valid BVH frames; do not claim G1 tracking success')
    (args.output/'report.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2), encoding='utf8')
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    return stats


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--url', default='http://127.0.0.1:8771/ws')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tail', action='store_true', help='Read a growing receiver JSONL instead of replay pacing')
    p.add_argument('--calibration', type=Path, help='Paired board mode; omit for human_skeleton/1.0 input')
    p.add_argument('--camera-ids', nargs=2, help='Explicit calibration identity for logical left and right')
    p.add_argument('--allow-diagnostic', action='store_true', help='Offline diagnosis only, not physical calibration acceptance')
    p.add_argument('--fps', type=float, default=0, help='Optional input rate cap; never changes source motion speed')
    p.add_argument('--seconds', type=float, default=0)
    p.add_argument('--warmup', type=int, default=10)
    p.add_argument('--warmup-seconds', type=float, default=0.,
                   help='Minimum source-time span of valid calibration frames; ABI9 launcher uses 1.4 s')
    p.add_argument('--timeout', type=float, default=.5)
    args = p.parse_args()
    if args.fps < 0 or args.seconds < 0 or args.timeout <= 0:
        p.error('Invalid rate/duration/timeout')
    if args.calibration and not args.camera_ids:
        p.error('Board mode requires explicit --camera-ids LEFT RIGHT matching calibration')
    if args.tail and args.allow_diagnostic:
        p.error('Diagnostic-only calibration cannot be used for live tail')
    result = asyncio.run(run(args))
    raise SystemExit(1 if result['errors'] or result['fallen'] else 0)
