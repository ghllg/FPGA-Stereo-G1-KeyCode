"""PC half of the live chain. No hardware access; also used by paced replay."""
import json
import gc
import hashlib
import os
from pathlib import Path
import subprocess
import shutil
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
G1 = Path(r'E:\FPGAchina\G1_1080p_Candidate_20261020')
sys.path.insert(0, r'E:\FPGAchina\FPGA2026_1080p_20261020')
sys.path.insert(0, str(G1))
sys.path.insert(0, str(G1/'g1_physical'))
from pose3d_train.online import OnlineSession
import launcher
from windows_timing import enable_timer
from state_io import write_json


class Backend:
    def __init__(self, output, binding, lag_ms=380, smooth_ms=40, quality_guard=True):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.worker = self.child = None
        self.logs = []
        self.closed = self.stopping = False
        self.gc_was_enabled = None
        self.source_first_ns = None
        self.start_ns = time.perf_counter_ns()
        if launcher.health(8770) or launcher.health(8771):
            raise RuntimeError('G1 candidate service already occupied; preserve its input')
        self.owns_services = False
        self.error = None
        self.bridge_error = None
        self.smooth_ms, self.quality_guard = smooth_ms, quality_guard
        self.producer_status = self.output/'producer_status.json'
        self.producer_status.write_text(json.dumps(dict(state='running')),encoding='utf8')
        (self.output/'tracking_status.json').write_text(json.dumps(dict(state='preparing',reason='waiting for reliable body input')),encoding='utf8')
        self.close_timer = enable_timer()
        os.environ['G1_PERFORMANCE_OUTPUT'] = str(self.output)
        os.environ['G1_REFERENCE_LAG_MS'] = str(lag_ms)
        os.environ['G1_RECEIPT_CLOCK'] = '1'
        os.environ['G1_QUALITY_GUARD'] = '1' if quality_guard else '0'
        try:
            files = list(HERE.glob('*.py'))
            files += [G1/p for p in ('g1_stereo_bridge/stream.py','g1_stereo_bridge/conversion.py',
                'g1_stereo_bridge/upstream/windows_vision/mocap/reconstruction.py',
                'g1_physical/server.py','g1_physical/physical.py','g1_physical/reference_trajectory.py',
                'g1_unitree_official/standalone/windows_compat/policy_bridge.py',
                'g1_unitree_official/standalone/windows_compat/viewer_snapshot.py',
                'g1_unitree_official/standalone/windows_compat/performance_trace.py')]
            archive = self.output/'runtime_sources'
            archive.mkdir()
            manifest = []
            for index, path in enumerate(files):
                target = archive/(str(index)+'_'+path.name)
                shutil.copy2(path,target)
                manifest.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),snapshot=target.name))
            (self.output/'runtime_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
            self.owns_services = True
            process = launcher.spawn(HERE/'performance_server.py')
            for _ in range(200):
                if launcher.health(8771): break
                if process.poll() is not None: raise RuntimeError('Performance service exited; inspect G1 logs')
                time.sleep(.15)
            else: raise TimeoutError('Performance service startup timed out')
            launcher.start(native=True, browser=False)
            train = ROOT/'outputs/training_1080p_20261021'
            self.worker = OnlineSession(train/'calibration.json', train/'model/checkpoint.pt', binding, self.output/'online_3d')
            # Full cyclic scans of retained raw packets stall both receiver and
            # decoder. Collect before input, then defer cyclic GC for this
            # bounded capture only; ordinary reference counting remains active.
            self.gc_was_enabled = gc.isenabled()
            if self.gc_was_enabled:
                gc.collect()
                gc.disable()
            self.prepared_ns = time.perf_counter_ns()
            self.monitor = threading.Thread(target=self._monitor, name='G1_bridge_start', daemon=True)
            self.monitor.start()
        except Exception:
            self.close()
            raise

    def _monitor(self):
        try:
            while self.worker.latest is None:
                if self.stopping: return
                if self.worker.error: raise RuntimeError(self.worker.error)
                time.sleep(.002)
            self.logs = [(self.output/'bridge_stdout.txt').open('wb'), (self.output/'bridge_stderr.txt').open('wb')]
            command = [str(G1/'runtime/python.exe'), '-B', '-X', 'utf8',
                str(HERE/'performance_stream.py'), '--input', str(self.output/'online_3d/skeleton3d.jsonl'),
                '--tail', '--fps', '0', '--warmup-seconds', '1.4', '--output', str(self.output/'g1'),
                '--producer-status',str(self.producer_status),'--smooth-ms',str(self.smooth_ms)]
            if self.quality_guard:command.append('--quality-guard')
            self.child = subprocess.Popen(command,
                cwd=G1, env=launcher.environment(), stdout=self.logs[0], stderr=self.logs[1],
                creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as error:
            self.error = repr(error)

    def submit(self, record):
        if self.error or self.worker.error:
            raise RuntimeError(self.error or self.worker.error)
        if self.child and self.child.poll() is not None:
            self.bridge_error = 'G1 bridge stopped while source is still running; 3D diagnostics continued'
        if self.source_first_ns is None: self.source_first_ns = record['pc_receive_ns']
        self.worker.submit(record)

    def finish(self):
        processing = self.worker.finish()
        self.worker = None
        status_path=self.output/'producer_status.json'
        write_json(status_path,dict(state='ended',processed_frames=processing.get('frames',0)))
        self.stopping = True
        self.monitor.join(2)
        if self.monitor.is_alive(): raise RuntimeError('Bridge start worker did not stop')
        if self.error: raise RuntimeError(self.error)
        if self.child is None: raise RuntimeError('No valid 3D output reached G1')
        code = self.child.wait(timeout=15)
        bridge = json.loads((self.output/'g1/report.json').read_text(encoding='utf-8'))
        expected_end = code == 0 and bridge.get('end_reason')=='source_eof' and not bridge['errors']
        expected_end &= bool(bridge['sent_bvh_frames'] and bridge['output_frames'])
        report = dict(processing=processing, bridge=bridge, producer_end_expected=expected_end,
            bridge_error=getattr(self,'bridge_error',None),source_processing_finished=True,
            realtime_gc=dict(automatic_enabled_before=getattr(self,'gc_was_enabled',None),
                policy='cyclic collection deferred until capture closes; reference counting remains active'),
            startup=dict(start_ns=self.start_ns, prepared_ns=self.prepared_ns, first_receipt_ns=self.source_first_ns))
        (self.output/'backend.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        if not bridge['sent_bvh_frames'] or not bridge['output_frames']:
            raise RuntimeError('No valid skeleton reached G1; native tracking did not start. Check the 10ms stereo gate and body validity; see backend.json.')
        if not expected_end or bridge['fallen']:
            raise RuntimeError('G1 tracking did not finish normally; see backend.json.')
        return report

    def close(self):
        if self.closed: return
        self.closed = self.stopping = True
        try:
            monitor = getattr(self, 'monitor', None)
            if monitor: monitor.join(2)
            processing_error = None
            if self.worker:
                try: self.worker.finish()
                except Exception as error: processing_error = error
                finally: self.worker = None
            if self.child and self.child.poll() is None:
                self.child.terminate()
                self.child.wait(timeout=5)
            for log in self.logs: log.close()
            if self.owns_services:
                time.sleep(.2)  # Let the bounded native telemetry writer finish its last records.
                launcher.stop()
            self.close_timer()
            if processing_error: raise RuntimeError('3D worker failed during cleanup') from processing_error
        finally:
            if getattr(self,'gc_was_enabled',False):gc.enable()

