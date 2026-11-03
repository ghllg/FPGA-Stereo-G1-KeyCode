"""Feed original observed skeleton rows into the explicit standing upper-body adapter."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import aiohttp

SCOPE = 'standing_upper_body'


def write_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding='utf8')
    deadline = time.perf_counter()+.25
    while True:
        try:
            temporary.replace(path); return
        except PermissionError:
            if time.perf_counter() >= deadline: raise
            time.sleep(.005)


async def run(args):
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    report = dict(motion_mode=SCOPE, reference_scope=SCOPE, fixed_lower_body_reference=True, lower_body_measured=False,
                  source_frames=0, sent_reference_frames=0, observed_arm_updates=dict(left=0, right=0),
                  cached_basis_arm_updates=dict(left=0, right=0), preparation_frames=0, preparation_ready=False,
                  output_frames=0, fallen=False, errors=[], reference_drained=False,
                  terminal_pause_without_drain=False, end_reason=None)
    status_path = output.parent/'tracking_status.json'
    pending = {}; ready = asyncio.get_running_loop().create_future(); done = asyncio.Event()
    latest_upper_body = None
    poses = (output/'poses.ndjson').open('w', encoding='utf8', buffering=1)
    acks = (output/'upper_body_quality.ndjson').open('w', encoding='utf8', buffering=1)
    def status(state, reason, **fields):
        if latest_upper_body is not None and 'upper_body' not in fields:
            fields['upper_body'] = latest_upper_body
        write_json(status_path, dict(state=state, reason=reason, reference_scope=SCOPE,
            fixed_lower_body_reference=True, lower_body_measured=False, pc_ns=time.perf_counter_ns(), **fields))
    async def receive(ws):
        async for message in ws:
            if message.type != aiohttp.WSMsgType.TEXT: continue
            packet = json.loads(message.data); kind = packet.get('type')
            if kind == 'ready' and not ready.done(): ready.set_result(packet)
            elif kind == 'ack':
                acks.write(json.dumps(packet, allow_nan=False)+'\n')
                future = pending.pop(packet['seq'], None)
                if future and not future.done(): future.set_result(packet)
            elif kind == 'pose':
                packet['pc_receive_ns'] = time.perf_counter_ns()
                poses.write(json.dumps(packet, allow_nan=False)+'\n'); report['output_frames'] += 1
                report['fallen'] |= bool(packet.get('fallen'))
                if packet.get('finished'):
                    report['reference_drained'] = not report['fallen']; done.set()
            elif kind == 'preparation_pose':
                packet['pc_receive_ns'] = time.perf_counter_ns()
                report['preparation_frames'] += 1
                write_json(output/'preparation_pose.json', packet)
                if packet.get('preparation_ready') and not report['preparation_ready']:
                    report['preparation_ready'] = True
                    write_json(output.parent/'upper_body_ready.json', dict(ready=True, session=packet['session'],
                        preparation_steps=packet['preparation_steps'], human_observation_count=0,
                        pc_ns=time.perf_counter_ns(), pose_path='g1/preparation_pose.json'))
                if packet.get('fallen'):
                    report['fallen'] = True; report['errors'].append('Preparation physics fell'); done.set()
            elif kind == 'tracking_state':
                status(packet['state'], packet.get('reason', 'upper-body tracking state'))
                if packet.get('terminal_undrained'):
                    report['terminal_pause_without_drain'] = True; done.set()
            elif kind == 'error':
                report['errors'].append(packet.get('message', str(packet))); done.set()
                for future in pending.values():
                    if not future.done(): future.set_exception(RuntimeError(report['errors'][-1]))
                if not ready.done(): ready.set_exception(RuntimeError(report['errors'][-1]))
    try:
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(args.url, heartbeat=20) as ws:
                reader = asyncio.create_task(receive(ws))
                try:
                    await ws.send_json(dict(type='init', motion_mode=SCOPE, smooth_ms=args.smooth_ms,
                                            engine='protomotions', name='真人上肢跟随 · 固定站姿参考'))
                    metadata = await asyncio.wait_for(ready, 30)
                    write_json(output/'metadata.json', metadata)
                    origin = clock_origin = start = None
                    while not Path(args.input).exists():
                        if reader.done() or report['errors']: raise RuntimeError('Upper-body preparation stopped')
                        await asyncio.sleep(.005)
                    with Path(args.input).open(encoding='utf-8-sig') as source:
                        while True:
                            if report['fallen'] or report['errors']: break
                            position = source.tell(); line = source.readline()
                            if not line or not line.endswith('\n'):
                                source.seek(position)
                                producer_ended = (args.producer_status and Path(args.producer_status).exists()
                                    and json.loads(Path(args.producer_status).read_text(encoding='utf8')).get('state') == 'ended')
                                if not args.tail or producer_ended: break
                                if reader.done(): raise RuntimeError('Upper-body websocket closed before source ended')
                                await asyncio.sleep(.005); continue
                            row = json.loads(line); report['source_frames'] += 1
                            if origin is None:
                                origin = row['timestamp_ns']; clock_origin = row['pc_receive_ns']; start = time.perf_counter()
                                report['source_origin_ns'] = origin
                            stamp = (row['timestamp_ns']-origin)/1e9
                            if args.fps > 0:
                                await asyncio.sleep(max(0, start+(report['source_frames']-1)/args.fps-time.perf_counter()))
                            seq = report['source_frames']-1
                            future = asyncio.get_running_loop().create_future(); pending[seq] = future
                            await ws.send_json(dict(type='frame', seq=seq, source_time=stamp,
                                source_pc_receive_ns=row['pc_receive_ns'], source_pc_origin_ns=clock_origin, values=row))
                            ack = await asyncio.wait_for(future, 5)
                            report['sent_reference_frames'] += int(ack['reference_accepted'])
                            arms = ack['upper_body']['arms']
                            latest_upper_body = ack['upper_body']
                            for side, state in arms.items():
                                report['observed_arm_updates'][side] += int(state['observed'])
                                report['cached_basis_arm_updates'][side] += int(state.get('shoulder_basis_cached', False))
                            mode = ('tracking' if any(s['reference_updated'] for s in arms.values()) else
                                    'holding' if any(s['state'] == 'holding' for s in arms.values()) else 'paused')
                            if ack.get('control_state') in ('paused', 'recovering', 'holding_standing'): mode = ack['control_state']
                            basis_note = '；肩线方向短时保持（最多500毫秒）' if any(s.get('shoulder_basis_cached') for s in arms.values()) else ''
                            status(mode, '上肢关节参考'+basis_note+'；机器人下肢使用固定站姿参考', upper_body=ack['upper_body'])
                            if report['fallen'] or report['errors']: break
                    report['end_reason'] = 'fallen' if report['fallen'] else 'error' if report['errors'] else 'source_eof'
                    await ws.send_json(dict(type='end'))
                    await asyncio.wait_for(done.wait(), 12)
                    await ws.send_json(dict(type='release'))
                finally:
                    reader.cancel(); await asyncio.gather(reader, return_exceptions=True)
    except Exception as error:
        report['errors'].append(repr(error)); report['end_reason'] = report['end_reason'] or 'error'
        status('failed', str(error))
    finally:
        poses.close(); acks.close(); write_json(output/'report.json', report)
    return 0 if (report['reference_drained'] and not report['fallen'] and not report['errors']
                 and report['output_frames'] and all(report['observed_arm_updates'].values())) else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--input', required=True); result.add_argument('--output', required=True)
    result.add_argument('--tail', action='store_true'); result.add_argument('--producer-status')
    result.add_argument('--fps', type=float, default=0)
    result.add_argument('--url', default='http://127.0.0.1:'+os.environ.get('G1_WEB_PORT', '8771')+'/ws')
    result.add_argument('--smooth-ms', type=float, default=80)
    result.add_argument('--quality-guard', action='store_true')
    return result


if __name__ == '__main__':
    raise SystemExit(asyncio.run(run(parser().parse_args())))
