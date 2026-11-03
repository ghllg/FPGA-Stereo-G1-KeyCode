"""Original live 3D worker and owned G1 lifecycle, with an explicit upper-body bridge."""
import json
from pathlib import Path
import subprocess
import time
import backend as production


class UpperBodyBackend(production.Backend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            deadline = time.monotonic()+30
            while time.monotonic() < deadline:
                ready_path = self.output/'upper_body_ready.json'
                if ready_path.exists():
                    ready = json.loads(ready_path.read_text(encoding='utf8'))
                    if (ready.get('ready') is True and ready.get('preparation_steps', 0) >= 25
                            and ready.get('human_observation_count') == 0):
                        self.prepared_ns = time.perf_counter_ns()
                        return
                if self.error: raise RuntimeError(self.error)
                if self.child and self.child.poll() is not None: raise RuntimeError('Upper-body preparation bridge exited')
                time.sleep(.02)
            raise TimeoutError('Upper-body standing physics preparation timed out')
        except Exception:
            self.close()
            raise

    def _monitor(self):
        try:
            if self.stopping: return
            self.logs = [(self.output/'bridge_stdout.txt').open('wb'), (self.output/'bridge_stderr.txt').open('wb')]
            command = [str(production.G1/'runtime/python.exe'), '-B', '-X', 'utf8',
                str(Path(__file__).with_name('upper_body_stream.py')), '--input', str(self.output/'online_3d/skeleton3d.jsonl'),
                '--tail', '--fps', '0', '--output', str(self.output/'g1'), '--producer-status', str(self.producer_status),
                '--smooth-ms', str(self.smooth_ms), '--quality-guard']
            self.child = production.subprocess.Popen(command, cwd=production.G1, env=production.launcher.environment(),
                stdout=self.logs[0], stderr=self.logs[1], creationflags=subprocess.CREATE_NO_WINDOW)
        except Exception as error:
            self.error = repr(error)

    def finish(self):
        processing = self.worker.finish(); self.worker = None
        production.write_json(self.producer_status, dict(state='ended', processed_frames=processing.get('frames', 0)))
        self.stopping = True; self.monitor.join(2)
        if self.monitor.is_alive(): raise RuntimeError('Upper-body bridge start worker did not stop')
        if self.error: raise RuntimeError(self.error)
        if self.child is None: raise RuntimeError('No original 3D output reached the upper-body bridge')
        code = self.child.wait(timeout=15)
        bridge = json.loads((self.output/'g1/report.json').read_text(encoding='utf8'))
        expected = (code == 0 and bridge.get('end_reason') == 'source_eof' and not bridge['errors']
                    and bridge['reference_drained'] and bridge['sent_reference_frames'] > 0
                    and bridge['output_frames'] > 0 and all(bridge['observed_arm_updates'].values()))
        report = dict(processing=processing, bridge=bridge, producer_end_expected=expected,
            source_processing_finished=True, bridge_error=self.bridge_error, reference_scope='standing_upper_body',
            fixed_lower_body_reference=True, lower_body_measured=False,
            startup=dict(start_ns=self.start_ns, prepared_ns=self.prepared_ns, first_receipt_ns=self.source_first_ns))
        production.write_json(self.output/'backend.json', report)
        if not expected or bridge['fallen']:
            raise RuntimeError('上肢受限模式未正常结束；查看 backend.json 的逐臂观测、暂停和仿真结果')
        return report
