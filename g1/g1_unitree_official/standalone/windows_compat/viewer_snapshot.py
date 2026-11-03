"""Display actual simulation snapshots without holding the physics lock."""
import copy
import sys
import time
import os
import json
from pathlib import Path
import mujoco
from performance_trace import ENABLED, emit


class SnapshotViewer:
    def __init__(self, handle, model, source, display):
        self.handle, self.model = handle, model
        self.source, self.display = source, display

    def __getattr__(self, name):
        return getattr(self.handle, name)

    def sync(self, *args, **kwargs):
        # The unchanged official PhysicsViewerThread owns this lock on entry.
        locker = sys.modules["__main__"].locker
        bridge = getattr(sys.modules.get('policy_bridge'), 'BRIDGE', None)
        if ENABLED and bridge and getattr(self, 'last_identity', None)==(bridge.reset_id, bridge.completed_seq):
            bridge.render_ready.clear()
            locker.release()
            try:
                bridge.render_ready.wait(.02)
            finally:
                locker.acquire()
        mujoco.mj_copyData(self.display, self.model, self.source)
        if bridge is None or bridge.reset_id == 0:
            # DDS intentionally pauses physics before the first valid source.
            # MjData has no derived geometry yet; draw its initial standing pose.
            mujoco.mj_forward(self.model, self.display)
        if ENABLED:
            copied_ns = time.perf_counter_ns()
            identity = (bridge.reset_id, bridge.completed_seq) if bridge else (0, 0)
            self.last_identity = identity
        locker.release()
        try:
            if time.perf_counter()-getattr(self,'last_status_check',0)>.1:
                self.last_status_check=time.perf_counter()
                folder=os.environ.get('G1_PERFORMANCE_OUTPUT')
                if folder:
                    try:
                        status=json.loads((Path(folder)/'tracking_status.json').read_text(encoding='utf8'))
                        self.status_read_failure_since=None
                        text=status['state'].upper()+': '+status['reason']
                        if text!=getattr(self,'last_status_text',None):
                            self.handle.set_texts((None,None,'Tracking status',text))
                            self.last_status_text=text
                    except (FileNotFoundError,json.JSONDecodeError):pass
                    except PermissionError:
                        # Atomic replacement briefly conflicts with Windows readers.
                        # Keep the last label for at most one second, then disclose it.
                        if getattr(self,'status_read_failure_since',None) is None:
                            self.status_read_failure_since=time.perf_counter()
                        elif time.perf_counter()-self.status_read_failure_since>1:
                            self.handle.set_texts((None,None,'Tracking status','STATUS UNAVAILABLE: state file cannot be read'))
                            self.last_status_text=None
            self.handle.sync(*args, **kwargs)
            if ENABLED:
                emit('render', reset_id=identity[0], sequence=identity[1], copied_ns=copied_ns,
                     submitted_ns=time.perf_counter_ns(), simulation_time=float(self.display.time),
                     qpos=self.display.qpos.tolist())
        finally:
            locker.acquire()


def launch_snapshot_viewer(launch, model, data, **kwargs):
    display_model = copy.copy(model)
    display_data = mujoco.MjData(display_model)
    mujoco.mj_copyData(display_data, display_model, data)
    mujoco.mj_forward(display_model, display_data)
    handle = launch(display_model, display_data, **kwargs)
    return SnapshotViewer(handle, display_model, data, display_data)
