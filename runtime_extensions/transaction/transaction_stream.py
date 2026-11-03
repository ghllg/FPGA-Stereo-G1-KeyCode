"""JSONL (or growing receiver log) -> causal BVH -> existing platform WebSocket."""
import argparse
import asyncio
import os
from collections import deque
from copy import deepcopy
import json
from pathlib import Path
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, 'C:\\Users\\developer\\Documents\\Codex\\2026-10-20\\iq\\work\\full_pipeline_exec_20261022')

from paths import BRIDGE
sys.path.insert(0, str(BRIDGE))
import aiohttp
from conversion import LiveBVH, PairedStereo
from transactional_hold import TransactionalJointHold as JointHold, recovery_eligible
from tracking_quality import SkeletonQuality
from state_io import read_json, write_json
from mocap.reconstruction import CausalFilter
import numpy as np
sys.path.insert(0, str(BRIDGE.parent/'g1_physical'))
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
    joint_hold = JointHold()
    tau=getattr(args,'smooth_ms',40)/1000
    smoothing = CausalFilter(tau)
    guarded=getattr(args,'quality_guard',False)
    quality=SkeletonQuality() if guarded else None
    producer_status=getattr(args,'producer_status',None)
    stats = dict(input=str(args.input), url=args.url, mode='live_tail' if args.tail else 'timestamped_replay',
                 source_fps_limit=args.fps, errors=[], output_frames=0, fallen=False,
                 source_contains_estimates=False, adapter=None, first_failure_source_time=None,
                 simulation_time=0, initial_floor_offset=None, minimum_root_height=None,
                 stale_source_frames_dropped=0, source_records_read=0, sent_bvh_frames=0,
                 end_reason=None,quality_rejected_frames=0,pause_count=0,recovery_count=0,
                 recovery_conditioned_frames=0,quality_guard_enabled=guarded,
                 mapping_rejected_frames=0,mapping_fit_limit_m=.08 if guarded else None)
    stats.update(held_joint_frames=0, maximum_held_joint_age_ms=0.,
                 fully_current_body_frames_sent=0, partial_body_frames_sent=0,
                 held_joint_limit_ms=100., maximum_missing_body_joints=JointHold.MAX_MISSING)
    stats['G1_input_conditioning'] = dict(type='causal_exponential', tau_s=tau,
        calibration_prefix_filtered=False, raw_3d_observations_changed=False,
        source_timestamps_changed=False, scope='G1 reference only; filtering adds motion-response lag beyond timestamp latency.')
    stats['hold_cache_transaction'] = 'commit after local quality, mapping and freshness; not server acceptance'
    startup_fresh = os.environ.get('ROUND3_STARTUP_FRESH') == '1'
    startup_budget_ns = round((float(os.environ.get('G1_REFERENCE_LAG_MS', '240'))-160)*1e6)
    if startup_budget_ns < 0:
        raise ValueError('Reference lag does not cover the 160ms policy horizon')
    stats['round3_startup'] = dict(enabled=startup_fresh, maximum_aligned_age_ms=startup_budget_ns/1e6,
        skipped_frames=0, scope='Only before first reference; steady-state quality and timeout unchanged')
    stats['stage2_calibration_protocol'] = 'v2_complete_raw_warmup_prefix_no_stale_reference_sent'
    costs, acks, arrivals, ages, fits = [deque(maxlen=18000) for _ in range(5)]
    started = last_source = first_source = last_slot = None
    done, ready, ack = asyncio.Event(), asyncio.Event(), asyncio.Event()
    input_name = 'COCO17 live bridge: '+args.input.name
    last_frame = None
    first_sent_source = None
    last_sent_wall = None
    receipt_offset_ns = source_pc_origin_ns = None
    paused=False
    end_sent=False
    stats['reference_drained']=False
    recovery_first=None
    reliable_count=0
    server_recovering=False
    status_value=None
    warmup_last_accepted_ns=None
    ack_timing={}
    async with aiohttp.ClientSession() as client, client.ws_connect(args.url) as ws:
        hello = await ws.receive_json(timeout=10)
        if hello.get('busy'):
            raise ValueError('Platform input is busy: stop current browser/source first')
        async def status(state, reason, stamp=None):
            nonlocal paused,status_value,recovery_first,reliable_count
            value=(state,reason)
            if value==status_value:return
            status_value=value
            event=dict(state=state,reason=reason,pc_ns=time.perf_counter_ns(),source_timestamp_ns=stamp)
            transitions.write(json.dumps(event)+'\n');transitions.flush()
            path=args.output.parent/'tracking_status.json'
            write_json(path,event)
            if state in ('paused','disconnected'):
                if not paused:stats['pause_count']+=1
                paused=True;recovery_first=None;reliable_count=0
                if end_sent:
                    stats['terminal_pause_without_drain']=True
                    done.set()
                if ready.is_set():await ws.send_json(dict(type='tracking_state',state='paused',reason=reason))
            elif state=='recovering':
                stats['recovery_count']+=1
                if ready.is_set():await ws.send_json(dict(type='tracking_state',state='recovering',reason=reason))
                paused=False
        async def receive():
            nonlocal server_recovering,ack_timing
            async for message in ws:
                p = json.loads(message.data)
                if p['type'] == 'ready' and p.get('name') == input_name:
                    stats['adapter'] = p.get('adapter'); ready.set()
                elif p['type'] == 'ack':
                    ack_timing=dict(ack_received_ns=time.perf_counter_ns(),server_accepted_ns=p.get('server_accepted_ns'),server_reference_saved_ns=p.get('server_reference_saved_ns'))
                    server_recovering=p.get('recovering',False)
                    stats['recovery_conditioned_frames']+=int(p.get('recovery_conditioned',False))
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
                        stats['fallen'] = stats['fallen'] or bool(p.get('fallen'))
                        if p.get('finished') and not stats['fallen']:
                            stats['reference_drained']=True
                        if stats['fallen']: stats['first_failure_source_time'] = p['source_time']
                        done.set(); ack.set()
                elif p['type'] == 'error':
                    stats['errors'].append(p['message']); done.set(); ready.set(); ack.set()
                elif p['type']=='tracking_state' and p['state']=='paused':
                    if not paused:await status('paused',p.get('reason','reference coverage insufficient'))
        with (args.output/'poses.ndjson').open('x', encoding='utf8') as poses, (args.output/'bvh_frames.ndjson').open('x', encoding='utf8') as frames, (args.output/'quality.ndjson').open('x',encoding='utf8') as diagnostics, (args.output/'state_events.ndjson').open('x',encoding='utf8') as transitions, (args.output/'timing.ndjson').open('x',encoding='utf8') as timing, args.input.open(encoding='utf-8-sig') as source:
            receiver = asyncio.create_task(receive())
            try:
                await status('preparing','waiting for reliable body input')
                last_line = time.perf_counter()
                while not done.is_set():
                    if args.tail and time.perf_counter()-last_line > args.timeout:
                        if not guarded or not producer_status:
                            raise TimeoutError('Live input stopped; no complete fresh record within timeout')
                        await status('disconnected','no fresh source records; waiting for reconnection')
                        if time.perf_counter()-last_line>5:
                            raise TimeoutError('Source reconnect deadline exceeded (5 seconds)')
                    if last_sent_wall is not None and time.perf_counter()-last_sent_wall > args.timeout:
                        if not guarded:raise TimeoutError('No valid skeleton within timeout; stop stale tracking')
                        if not paused:await status('paused','insufficient skeleton quality')
                    position = source.tell()
                    line = source.readline(65537)
                    if len(line) > 65536:
                        raise ValueError('Source JSONL exceeds bounded size')
                    if not line:
                        if not args.tail:
                            stats['end_reason']='source_eof';break
                        if producer_status and producer_status.exists():
                            producer=read_json(producer_status)
                            if producer['state']=='ended':
                                stats['end_reason']='source_eof';break
                        if not guarded and time.perf_counter()-last_line > args.timeout:
                            raise TimeoutError('Live input stopped; ending reference and pausing simulation')
                        await asyncio.sleep(.002); continue
                    if not line.endswith('\n'):
                        if args.tail:
                            if producer_status and producer_status.exists() and read_json(producer_status)['state']=='ended':
                                raise ValueError('Incomplete final JSONL record at explicit source EOF')
                            source.seek(position)
                            await asyncio.sleep(.002); continue
                    row = json.loads(line)
                    read_ns=time.perf_counter_ns()
                    stats['source_records_read'] += 1
                    if args.tail:
                        pc_stamp = row.get('pc_receive_ns') if stereo else row.get('pc_receive_ns', row.get('pc_complete_ns'))
                        if type(pc_stamp) is not int or pc_stamp <= 0 or pc_stamp > time.perf_counter_ns():
                            raise ValueError('Live tail requires a valid same-PC monotonic source timestamp')
                        if converter.offsets is not None and (time.perf_counter_ns()-pc_stamp)/1e9 > args.timeout:
                            recovery_first=None;reliable_count=0
                            stats['stale_source_frames_dropped'] += 1
                            continue
                    last_line = time.perf_counter()
                    packet = stereo.push(row) if stereo else row
                    if args.tail and source_pc_origin_ns is None:
                        offset = pc_stamp-packet['timestamp_ns']
                        receipt_offset_ns = offset if receipt_offset_ns is None else min(receipt_offset_ns, offset)
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
                    calibrated = converter.offsets is not None
                    adapted, held = joint_hold.apply(packet, enabled=calibrated)
                    admission = dict(accepted=True)
                    if quality:
                        admission=quality.assess(adapted)
                        diagnostics.write(json.dumps(dict(source_seq=packet['seq'],source_timestamp_ns=packet['timestamp_ns'],
                            joints=[dict(index=i,valid=j['valid'],estimated=j.get('estimated',False),reason=j.get('reason'),xyz=j.get('xyz')) for i,j in enumerate(packet['joints'])],
                            held_joints=held,**admission))+'\n');diagnostics.flush()
                        if not admission['accepted']:
                            joint_hold.abort()
                            stats['quality_rejected_frames']+=1
                            recovery_first=None;reliable_count=0
                            if not calibrated:
                                # Isolated missing frames contribute no bones to
                                # calibration, but need not discard its reliable
                                # observations. Reset on a gap over 200 ms.
                                if warmup_last_accepted_ns is None or packet['timestamp_ns']-warmup_last_accepted_ns>200_000_000:
                                    converter.prefix.clear();converter.prefix_start_ns=None
                                await status('preparing',admission['reasons'][0]['reason'],packet['timestamp_ns'])
                            elif not paused and last_sent_wall is not None and time.perf_counter()-last_sent_wall>args.timeout:
                                await status('paused',admission['reasons'][0]['reason'],packet['timestamp_ns'])
                            continue
                        if not calibrated:
                            if warmup_last_accepted_ns is not None and packet['timestamp_ns']-warmup_last_accepted_ns>200_000_000:
                                converter.prefix.clear();converter.prefix_start_ns=None
                            warmup_last_accepted_ns=packet['timestamp_ns']
                    conditioned = deepcopy(adapted)
                    smoothing.apply(conditioned['joints'],conditioned['timestamp_ns'])
                    position_adjustments = [np.linalg.norm(np.array(a['xyz'])-b['xyz'])*1000
                        for a,b in zip(adapted['joints'],conditioned['joints']) if a['valid']]
                    if calibrated: adapted = conditioned
                    frame = converter.push(adapted)
                    costs.append((time.perf_counter()-before)*1000)
                    if frame is None:
                        joint_hold.abort()
                        recovery_first=None;reliable_count=0
                        continue
                    if guarded and frame['fit_p95_m'] > .08:
                        joint_hold.abort()
                        stats['quality_rejected_frames'] += 1
                        stats['mapping_rejected_frames'] += 1
                        recovery_first=None;reliable_count=0
                        reason='fixed-bone mapping error exceeds 80 mm'
                        diagnostics.write(json.dumps(dict(source_seq=packet['seq'],
                            source_timestamp_ns=packet['timestamp_ns'],stage='mapping',accepted=False,
                            fit_p95_m=frame['fit_p95_m'],reasons=[dict(reason=reason)]))+'\n')
                        diagnostics.flush()
                        if calibrated and not paused and last_sent_wall is not None and time.perf_counter()-last_sent_wall>args.timeout:
                            await status('paused',reason,packet['timestamp_ns'])
                        continue
                    if args.tail and source_pc_origin_ns is None:
                        # Freeze a causal source/PC alignment from the warmup receipts.
                        # A delayed first frame must not delay the entire later motion.
                        source_pc_origin_ns = receipt_offset_ns+converter.origin_ns
                    if not guarded and last_source is not None and stamp-last_source > args.timeout:
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
                        joint_hold.abort()
                        recovery_first=None;reliable_count=0
                        stats['stale_source_frames_dropped'] += 1
                        continue
                    if args.tail and startup_fresh and first_sent_source is None:
                        aligned_ns = source_pc_origin_ns + round(frame['source_time']*1e9)
                        aligned_age_ns = time.perf_counter_ns()-aligned_ns
                        if aligned_age_ns > startup_budget_ns:
                            joint_hold.abort()
                            recovery_first=None;reliable_count=0
                            stats['round3_startup']['skipped_frames'] += 1
                            continue
                        stats['round3_startup']['first_reference_aligned_age_ms'] = aligned_age_ns/1e6
                    joint_hold.commit()
                    eligible = recovery_eligible(admission, held, packet)
                    diagnostics.write(json.dumps(dict(stage='hold_cache',source_seq=packet['seq'],
                        source_timestamp_ns=packet['timestamp_ns'],local_admitted=True,
                        recovery_eligible=eligible,server_accepted=False))+'\n');diagnostics.flush()
                    if guarded and paused:
                        if not eligible:
                            recovery_first=None;reliable_count=0
                            continue
                        if recovery_first is None:recovery_first=packet['timestamp_ns']
                        reliable_count+=1
                        if reliable_count<4 or packet['timestamp_ns']-recovery_first<200_000_000:continue
                        await status('recovering','bounded transition to fresh reliable input',packet['timestamp_ns'])
                    frame['source_pc_receive_ns'] = pc_stamp if args.tail else None
                    frame['source_pc_origin_ns'] = source_pc_origin_ns
                    frame['client_time'] = time.perf_counter()*1000
                    fits.append(frame.pop('fit_p95_m'))
                    stats['source_contains_estimates'] |= frame.pop('source_estimated')
                    frame.pop('timestamp_ns')
                    logged_frame = dict(frame, source_timestamp_ns=packet['timestamp_ns'], send_pc_ns=time.perf_counter_ns())
                    logged_frame['held_joints'] = held
                    logged_frame['conditioning'] = dict(tau_s=tau if calibrated else 0.,
                        maximum_position_adjustment_mm=float(max(position_adjustments,default=0.)) if calibrated else 0.)
                    if args.tail: logged_frame['source_pc_monotonic_ns'] = pc_stamp
                    frames.write(json.dumps(logged_frame)+'\n'); frames.flush()
                    logged_ns=time.perf_counter_ns()
                    ack.clear()
                    await ws.send_json(frame)
                    sent_ns=time.perf_counter_ns()
                    stats['sent_bvh_frames'] += 1
                    stats['partial_body_frames_sent' if held else 'fully_current_body_frames_sent'] += 1
                    if held:
                        stats['held_joint_frames'] += 1
                        stats['maximum_held_joint_age_ms'] = max(stats['maximum_held_joint_age_ms'],max(h['age_ms'] for h in held))
                    last_frame = frame
                    if first_sent_source is None: first_sent_source = frame['source_time']
                    await asyncio.wait_for(ack.wait(), args.timeout)
                    timing.write(json.dumps(dict(source_seq=packet['seq'],bvh_seq=frame['seq'],source_timestamp_ns=packet['timestamp_ns'],
                        read_ns=read_ns,source_pc_receive_ns=pc_stamp if args.tail else None,
                        built_ns=logged_frame['send_pc_ns'],logged_ns=logged_ns,sent_ns=sent_ns,**ack_timing))+'\n')
                    if not server_recovering and not paused and not done.is_set():
                        await status('tracking','following current admitted input',packet['timestamp_ns'])
                    last_source = stamp
                    last_sent_wall = time.perf_counter()
                    last_line = last_sent_wall
                if last_source is not None and not done.is_set():
                    end_sent=True
                    await ws.send_json(dict(type='end'))
                    if not paused:await asyncio.wait_for(done.wait(), 5)
                if paused and not stats['reference_drained']:
                    stats['terminal_pause_without_drain']=True
                if stats['fallen']:
                    stats['end_reason']='physical_failure'
                    await status('error','G1 simulation fell; tracking stopped, source diagnostics continue')
                elif stats['errors']:
                    stats['end_reason']='source_or_processing_error'
                    await status('error',stats['errors'][-1])
                if stats['end_reason']=='source_eof':
                    await status('ended',('source ended while paused; references not drained' if paused and not stats['reference_drained'] else 'source ended; diagnostics complete') if last_frame else
                        'source ended; G1 never initialized because body quality was insufficient')
            except Exception as error:
                joint_hold.abort()
                stats['end_reason']='source_or_processing_error'
                stats['errors'].append(type(error).__name__+': '+str(error))
                await status('error',type(error).__name__+': '+str(error))
                if last_source is not None and not ws.closed:
                    try:
                        end_sent=True
                        await ws.send_json(dict(type='end'))
                        if not paused:await asyncio.wait_for(done.wait(), 5)
                    except (asyncio.TimeoutError,ConnectionError,aiohttp.ClientError):pass
                if paused and not stats['reference_drained']:
                    stats['terminal_pause_without_drain']=True
                await status('error',type(error).__name__+': '+str(error))
            finally:
                joint_hold.abort()
                if not ws.closed:
                    try:await ws.send_json(dict(type='release'))
                    except (ConnectionError,aiohttp.ClientError):pass
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
    stats['reference_drained']=bool(stats['reference_drained'] and not stats['fallen'] and not stats['errors'])
    stats['terminal_pause_without_drain']=bool(paused and not stats['reference_drained'])
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
    stats['warmup_max_valid_gap_ms']=200 if guarded else None
    stats['source_pc_origin_ns'] = source_pc_origin_ns
    stats['source_clock_alignment'] = 'minimum receipt minus source offset during causal warmup, then frozen'
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
    p.add_argument('--producer-status',type=Path,help='Explicit producer lifecycle marker; EOF is not inferred from a timeout')
    p.add_argument('--quality-guard',action='store_true')
    p.add_argument('--smooth-ms',type=float,default=40)
    args = p.parse_args()
    if args.fps < 0 or args.seconds < 0 or args.timeout <= 0 or not 0<=args.smooth_ms<=200:
        p.error('Invalid rate/duration/timeout')
    if args.calibration and not args.camera_ids:
        p.error('Board mode requires explicit --camera-ids LEFT RIGHT matching calibration')
    if args.tail and args.allow_diagnostic:
        p.error('Diagnostic-only calibration cannot be used for live tail')
    result = asyncio.run(run(args))
    raise SystemExit(1 if result['errors'] or result['fallen'] else 0)
