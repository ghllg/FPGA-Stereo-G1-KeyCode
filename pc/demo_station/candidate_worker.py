"""Isolated latest-candidate service. TCP input starts only on an explicit command."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
import traceback

from .candidate import G1, PC, ROOT, RETENTION, SERVER, STREAM, pairing_settings, snapshot_candidate, validate_live_identity
from .core import digest, read_json, write_json


class OwnedProcesses:
    """Manage only this worker's descendants, never every process using GPY."""
    def __init__(self, server=SERVER, stream=STREAM, *, full_body_check=False):
        import psutil
        self.psutil = psutil; self.original = subprocess.Popen; self.processes = {}; self.next_collect = 0.
        self.routes = {str(PC/'performance_server.py'):str(server),str(SERVER):str(server),
                       str(PC/'performance_stream.py'):str(stream),str(STREAM):str(stream)}
        if full_body_check:
            self.routes[str(G1/'g1_unitree_official/standalone/simulate_python/unitree_mujoco.py')] = str(
                Path(__file__).parent/'native_runtime/unitree_mujoco.py')

    def collect(self, force=False):
        now = time.monotonic()
        if not force and now < self.next_collect: return
        self.next_collect = now+1.
        try:
            for process in self.psutil.Process().children(recursive=True):
                self.processes[(process.pid,process.create_time())] = process
        except self.psutil.Error: pass

    def popen(self, command, *args, **kwargs):
        if isinstance(command,list): command = [self.routes.get(str(x),str(x)) for x in command]
        result = self.original(command,*args,**kwargs)
        self.collect(force=True)
        return result

    def close(self):
        self.collect(force=True); active = []
        for (pid,created),process in reversed(list(self.processes.items())):
            try:
                if process.is_running() and process.create_time() == created:
                    process.terminate(); active.append(process)
            except self.psutil.Error: pass
        _,remaining = self.psutil.wait_procs(active,timeout=3)
        for process in remaining:
            try: process.kill()
            except self.psutil.Error: pass
        self.psutil.wait_procs(remaining,timeout=3)


class SavedSkeleton:
    """Explicit saved-3D replay; new PC receipts, preserved source times, no wire identity."""
    def __init__(self, source, output):
        self.source = Path(source); self.output = Path(output); self.output.mkdir(exist_ok=False)
        (self.output/'skeleton3d.jsonl').touch()
        self.latest = self.error = None; self.count = self.skipped = 0
        self.stop = threading.Event(); self.done = threading.Event(); self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self.run,daemon=True); self.thread.start()

    def run(self):
        try:
            with self.source.open(encoding='utf-8-sig') as source: rows=[json.loads(line) for line in source if line.strip()]
            if not rows: raise ValueError('Saved 3D source is empty')
            if any(b['timestamp_ns']<=a['timestamp_ns'] for a,b in zip(rows,rows[1:])):raise ValueError('Saved source time is not increasing')
            start = time.perf_counter(); origin=rows[0]['timestamp_ns'];index=last=-1
            with (self.output/'skeleton3d.jsonl').open('a',encoding='utf8',buffering=1) as output:
                for tick in range(math.ceil((rows[-1]['timestamp_ns']-origin)/1e9*14)+1):
                    if self.stop.wait(max(0,start+tick/14-time.perf_counter())): break
                    while index+1<len(rows) and rows[index+1]['timestamp_ns']<=origin+tick/14*1e9:index+=1
                    if index==last:continue
                    self.skipped+=index-last-1;last=index
                    row = dict(rows[index],pc_receive_ns=time.perf_counter_ns(),pc_complete_ns=time.perf_counter_ns(),
                        source_role='UI saved reconstructed3D replay; PC receipts reissued; no new camera or board measurement')
                    output.write(json.dumps(row,allow_nan=False)+'\n'); self.count += 1; self.latest = row
        except Exception:
            self.error = traceback.format_exc()
        finally: self.done.set()

    def finish(self):
        self.stop.set()
        if self.thread:
            self.thread.join(5)
            if self.thread.is_alive(): raise RuntimeError('Saved 3D producer did not stop')
        report = dict(frames=self.count,dropped_queue_packets=0,source_records_rate_capped=self.skipped,
            error=self.error,source=str(self.source),source_sha256=digest(self.source),pc_receipts_reissued=True)
        write_json(self.output/'summary.json',report)
        if self.error: raise RuntimeError(self.error)
        return report


def check_completion(report):
    bridge = report['bridge']
    if not (bridge.get('reference_drained') is True and not bridge.get('terminal_pause_without_drain')
            and not bridge.get('fallen') and not bridge.get('errors') and bridge.get('output_frames',0)>0):
        raise RuntimeError('候选未完整排空或发生失败；保留原始记录，不标为成功')
    if report.get('reference_scope') == 'standing_upper_body':
        updates = bridge.get('observed_arm_updates',{})
        if not all(updates.get(side,0)>0 for side in ('left','right')):
            raise RuntimeError('双臂尚未都收到可靠观测；不能把单纯站立输出记为上肢测试完成')


def candidate_error_message(error, pair20, output=None):
    if 'actual_horizontal_tracking_error_exceeds_existing_0.2m' in str(error):
        return '机器人与人体的实际水平位移偏差超过 20 cm，仿真已保护停止；本组未通过，请回到站立位置后开始新会话。'
    if pair20 and 'Check the 10ms stereo gate' in str(error):
        if output is not None and (Path(output)/'g1/report.json').exists():
            from .full_body_stream import startup_failure_message
            detail=startup_failure_message(read_json(Path(output)/'g1/report.json'))
            if detail:return detail
        return '当前候选配对上限为 20 ms；没有有效三维骨架到达 G1，跟踪未启动。请检查原始时差和身体有效性；原始原因保留在 candidate_error.json。'
    return str(error)


def bridge_exit_error(output):
    output=Path(output)
    status_path=output/'tracking_status.json'
    if status_path.exists():
        status=read_json(status_path)
        if status.get('state')=='error' and status.get('reason'):return status['reason']
    report_path=output/'g1/report.json'
    if report_path.exists():
        report=read_json(report_path)
        if report.get('fallen'):return 'G1 仿真跌倒，跟随已停止；本组未通过。'
        if report.get('errors'):return str(report['errors'][-1])
    return '输入未明确停止时 G1 桥已退出；不作为正常完成'


def standing_small_motion_enabled(config):
    enabled = config.get('standing_small_motion_check', False)
    if type(enabled) is not bool:
        raise ValueError('standing_small_motion_check must be an explicit boolean')
    if enabled and (config.get('full_body_check') is not True or config.get('motion_mode','full_body') != 'full_body'):
        raise ValueError('Standing small-motion check requires the full-body check path')
    return enabled


def run(config, output):
    output = Path(output); status_path = output/'candidate_status.json'; control_path = output/'candidate_control.json'
    def status(state, **fields): write_json(status_path,dict(state=state,pc_ns=time.perf_counter_ns(),**fields))
    status('preparing')
    motion_mode=config.get('motion_mode','full_body')
    if motion_mode not in ('full_body','standing_upper_body'):raise ValueError('未知跟随模式')
    upper=motion_mode=='standing_upper_body'
    full_check=config.get('full_body_check',False)
    standing_check=standing_small_motion_enabled(config)
    os.environ['G1_STANDING_SMALL_MOTION_CHECK']='1' if standing_check else '0'
    hybrid=bool(config.get('hybrid_refinement'))
    from .hybrid_candidate import hip_flip_enabled, latency_settings, reprojection_limit
    hip_flip_check=hip_flip_enabled(config)
    reconstruction_limit=reprojection_limit(config)
    latency=latency_settings(config)
    write_json(output/'latency_settings.json',latency)
    if full_check and motion_mode!='full_body':raise ValueError('全身短测不能使用固定下肢的上肢模式')
    server=Path(__file__).with_name('upper_body_server.py' if upper else 'full_body_server.py') if upper or full_check else SERVER
    stream=Path(__file__).with_name('upper_body_stream.py' if upper else 'full_body_stream.py') if upper or full_check else STREAM
    proof,assets = snapshot_candidate(config,output)
    pairing=pairing_settings(config);pair20=pairing['transport_pair_limit_us']==20000
    template = read_json(assets['binding'])
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[name] = '1'
    os.environ.update(ROUND3_ACK_FIX='1',ROUND3_STARTUP_FRESH='0',G1_REFERENCE_LAG_MS=str(latency['reference_lag_ms']),
                     G1_QUALITY_GUARD='1',G1_RECEIPT_CLOCK='1',G1_WEB_PORT='8771',G1_DDS_DOMAIN='74')
    sys.path[:0] = [str(RETENTION),str(PC),str(ROOT/'windows_vision'),str(ROOT)]
    if pair20:
        from . import pair20_performance_receiver
        sys.modules['performance_receiver']=pair20_performance_receiver
    # Keep genuine reconnect semantics. Never import the saved-fixture EOF wrapper.
    from retention_receiver_v2 import activate
    raw_receiver=activate()
    if hybrid:
        from .hybrid_receiver import run as responsive_run
    else:
        from .candidate_receiver import run as responsive_run
    import performance_receiver
    class ResponsiveReceiver(raw_receiver):
        run=responsive_run
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if hybrid:self.raw_image_max_bytes=int(config.get('max_mib',512))*1024*1024
        def finish(self):
            report=super().finish()
            if hybrid:
                report['latency_scope']='Publish to complete FPGA metadata and both original JPEGs; excludes refinement, 3D, and display'
                write_json(self.output/'receiver_summary.json',report)
            return report
    performance_receiver.Receiver=ResponsiveReceiver
    from live_receiver import LiveReceiver
    import backend as production
    import launcher
    owned = OwnedProcesses(server,stream,full_body_check=full_check); production.subprocess.Popen = owned.popen; launcher.stop = owned.close
    if full_check:
        from .startup_health import StartupHealth
        launcher.health=StartupHealth(owned,launcher.health)
    original_online = production.OnlineSession
    if pair20:
        from pose3d_train.online_pair20 import OnlineSession as selected_online
    else:selected_online=original_online
    def online(calibration, checkpoint, binding, destination):
        if config.get('recording_source'): return SavedSkeleton(config['recording_source'],destination)
        if hybrid:
            from .hybrid_online import HybridOnlineSession
            return HybridOnlineSession(assets['calibration'],assets['checkpoint'],binding,destination,
                                       max_delta_ms=20.,backend=config.get('hybrid_backend','cpu'),hip_flip_check=hip_flip_check,
                                       max_reprojection_px=reconstruction_limit)
        if pair20:
            return selected_online(assets['calibration'],assets['checkpoint'],binding,destination,max_delta_ms=20.)
        return selected_online(assets['calibration'],assets['checkpoint'],binding,destination)
    production.OnlineSession = online
    runtime = output/'candidate_runtime_sources'; runtime.mkdir(exist_ok=False)
    sources = [Path(__file__),Path(__file__).with_name('core.py'),Path(__file__).with_name('candidate.py'),Path(__file__).with_name('candidate_receiver.py'),
        Path(__file__).with_name('pipeline.py'),ROOT/'pose3d_train/online.py',ROOT/'pose3d_train/board_adapter.py',
        SERVER,STREAM,STREAM.with_name('transactional_hold.py'),
        STREAM.with_name('paths.py'),PC/'backend.py',PC/'live_receiver.py',PC/'performance_receiver.py',
        RETENTION/'retention_receiver_v2.py',RETENTION/'retention_parser_v2.py',RETENTION/'retention_receiver.py']
    sources += [p for p in SERVER.parent.glob('*.py') if p not in sources]
    sources += [Path(__file__).with_name(name) for name in ('app.py','services.py','core.py','robot_view.py','render_robot.py')]
    if pair20:sources += [ROOT/'pose3d_train/online_pair20.py',ROOT/'pose3d_train/geometry.py',ROOT/'pose3d_train/geometry_reprojection.py',Path(__file__).with_name('pair20_performance_receiver.py'),ROOT/'demo_station/profiles/pairing20_candidate.json']
    if upper:sources += [Path(__file__).with_name('upper_body_'+name+'.py') for name in ('backend','server','stream','tracker')]
    if full_check:
        sources += [Path(__file__).with_name(name) for name in ('full_body_backend.py','full_body_preparation.py','full_body_stream.py','full_body_quality.py','full_body_server.py','actual_tracking_guard.py','native_viewer_snapshot.py','startup_health.py')]
        sources.append(Path(__file__).parent/'native_runtime/unitree_mujoco.py')
    if standing_check:
        sources += [Path(__file__).with_name(name) for name in ('standing_scope.py','standing_small_motion_stream.py')]
    if hybrid:
        from .hybrid_candidate import runtime_directory
        sources += [Path(__file__).with_name(name) for name in ('hybrid_candidate.py','hybrid_receiver.py','hybrid_online.py','pose_refiner.py','person_detector.py','jpeg_pair_decoder.py')]+[
            ROOT/'demo_station/models/rtmpose_m/source_manifest.json',
            ROOT/'demo_station/models/rtmdet_nano/source_manifest.json',
            runtime_directory(config.get('hybrid_backend','cpu'))/'qualification.json']
    if hip_flip_check: sources.append(Path(__file__).with_name('hip_flip_refiner.py'))
    manifest = []
    for index,source in enumerate(sources):
        target = runtime/(str(index)+'_'+source.name); shutil.copy2(source,target)
        manifest.append(dict(path=str(source),snapshot=target.name,sha256=digest(source)))
    write_json(output/'candidate_runtime_manifest.json',manifest)
    write_json(output/'candidate_config.json',dict(variant=proof['variant']+('_standing_upper_v2' if upper else '_full_body_check_v2' if full_check else '')+('_fpga_guided_rtmpose_v1' if hybrid else ''),lag_ms=latency['reference_lag_ms'],smooth_ms=80,
        pairing=pairing,pair20_candidate=proof.get('pair20_candidate'),exposure_sync_verified=False,
        quality_guard=True,max_prediction_ms=160,server=str(server),stream=str(stream),
        motion_mode=motion_mode,fixed_lower_body_reference=upper,full_body_check=full_check,standing_small_motion_check=standing_check,
        hybrid_refinement=proof.get('hybrid_refinement'),hybrid_hip_flip_check=hip_flip_check,
        fixture_eof_wrapper=False,live_recv_poll_ms=20,stop_scope='Only processes descended from this isolated worker',
        calibration_provisional=proof['calibration_provisional']))
    backend = receiver = None; input_started = False; successful = False
    try:
        if upper:
            from .upper_body_backend import UpperBodyBackend
            backend_class=UpperBodyBackend
        elif full_check:
            from .full_body_backend import FullBodyCheckBackend
            backend_class=FullBodyCheckBackend
        else:backend_class=production.Backend
        backend = backend_class(output,template,lag_ms=latency['reference_lag_ms'],smooth_ms=80,quality_guard=True)
        status('ready',input_started=False)
        last_seen = time.monotonic(); count = 0; command_seq = -1
        while True:
            owned.collect()
            command = read_json(control_path) if control_path.exists() else {}
            if command.get('seq',-1) != command_seq:
                command_seq = command.get('seq',-1); action = command.get('action')
                if action == 'stop':
                    if not input_started:
                        status('stopped',input_started=False,reference_drained=False); break
                    status('stopping'); break
                if action == 'start':
                    if input_started: raise ValueError('输入已经开始，不能在原会话中再次启动')
                    if config.get('recording_source'):
                        backend.worker.start()
                        scope = dict(mode='saved_3d_ui_replay',pc_receipts_reissued=True,hardware_access=False,
                            source=str(config['recording_source']),wire_identity_present=False)
                    else:
                        binding = validate_live_identity(command.get('binding'),command.get('preflight'),template,config)
                        backend.worker.binding = binding
                        write_json(output/'live_preflight.json',command['preflight'])
                        (output/'audit').mkdir(exist_ok=True); write_json(output/'audit/binding.json',binding)
                        simulated = bool(command['preflight'].get('simulated_board'))
                        scope = dict(mode='simulated_board' if simulated else 'live',pc_receipts_reissued=simulated,
                            hardware_access=not simulated,source=str(output),wire_identity_present=True)
                        receiver = LiveReceiver(output/'pc_receiver',host=config['host'],port=int(config['board_port']),
                            expected_source_size=(1920,1080),expected_pair_limit_us=pairing['transport_pair_limit_us'],backend=backend)
                        receiver.start()
                    scope.update(lag_ms=latency['reference_lag_ms'],smooth_ms=80,source_timestamps_preserved=True,
                        motion_mode=motion_mode,fixed_lower_body_reference=upper,full_body_check=full_check,standing_small_motion_check=standing_check,
                        pairing=pairing,exposure_sync_verified=False,
                        hybrid_refinement=proof.get('hybrid_refinement'),hybrid_hip_flip_check=hip_flip_check,
                        source_ceiling_hz=14,stop_semantics='Explicit UI stop drains received complete records; socket EOF is not completion',
                        calibration_provisional=proof['calibration_provisional'])
                    write_json(output/'scope.json',scope)
                    input_started = True; last_seen = time.monotonic(); status('running',input_started=True)
            if input_started:
                if receiver:
                    if receiver.error or receiver.relay_error: raise RuntimeError(receiver.error or receiver.relay_error)
                    if len(receiver.rows) != count: count = len(receiver.rows); last_seen = time.monotonic()
                    if time.monotonic()-last_seen > 5: raise TimeoutError('真实输入持续断流超过 5 秒；不作为正常 EOF')
                elif backend.worker.done.is_set():
                    status('stopping',reason='saved_3d_source_complete'); break
                if backend.error or backend.worker.error: raise RuntimeError(backend.error or backend.worker.error)
                if backend.child and backend.child.poll() is not None:
                    raise RuntimeError(bridge_exit_error(output))
            time.sleep(.02)
        if input_started:
            if receiver:
                receiver.finish()
                if receiver.relay_error: raise RuntimeError(receiver.relay_error)
                if receiver.forwarded != len(receiver.rows) or not receiver.rows.identity_snapshot()['persistence_complete']:
                    raise RuntimeError('完整收到的记录未全部持久化并转交三维处理')
                write_json(output/'live_receiver_completion.json',dict(explicit_ui_stop=True,forwarded=receiver.forwarded,
                    received=len(receiver.rows),persistence=receiver.rows.identity_snapshot(),socket_eof_is_completion=False))
                receiver = None
            report = backend.finish(); check_completion(report); successful = True
        backend.close(); backend = None
        if successful: status('finished',reference_drained=True,input_started=True)
    except Exception as error:
        message=candidate_error_message(error,pair20,output)
        write_json(output/'candidate_error.json',dict(error=repr(error),display_error=message,traceback=traceback.format_exc()))
        status('failed',error=message,reference_drained=False)
        if message!=str(error):raise RuntimeError(message) from error
        raise
    finally:
        if receiver:
            try: receiver.finish()
            except Exception as error: write_json(output/'receiver_cleanup_error.json',dict(error=repr(error)))
        try:
            if backend: backend.close()
        finally: owned.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    try: run(read_json(args.config),args.output)
    except Exception as error:
        write_json(args.output/'candidate_status.json',dict(state='failed',error=str(error),pc_ns=time.perf_counter_ns(),reference_drained=False))
        raise


if __name__ == '__main__': main()
