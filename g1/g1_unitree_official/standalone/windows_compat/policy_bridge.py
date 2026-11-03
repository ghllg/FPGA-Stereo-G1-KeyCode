"""Explicit simulation adapter: upstream implicit PD, session reset and clock telemetry.

The official entry and bridge source files remain intact. This subclass adapts
their motor command semantics to the released policy's simulation contract.
It is enabled only in the local simulation process on isolated DDS domain 74.
"""
import json
import sys
import time
import threading
import faulthandler
from pathlib import Path
import mujoco
import numpy as np
import unitree_sdk2py_bridge as upstream

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'g1_physical'))
from physical import PhysicsTracker, read_robot_state
from windows_timing import enable_timer
from performance_trace import ENABLED, emit

RESET = Path(__file__).resolve().parents[1]/'session_reset.json'
sleep = time.sleep
original_step = mujoco.mj_step
BRIDGE = None


class PolicyBridge(upstream.UnitreeSdk2Bridge):
    def __init__(self, model, data):
        global BRIDGE
        self.stop_timer = enable_timer()
        self.reset_id = 0
        self.command_seq = self.completed_seq = 0
        self.command_ready = threading.Event()
        self.render_ready = threading.Event()
        self.reset_mtime = RESET.stat().st_mtime_ns if RESET.exists() else None
        self.helper = object.__new__(PhysicsTracker)
        self.helper.model,self.helper.data = model,data
        self.helper.initial_floor_offset = 0.
        BRIDGE = self
        super().__init__(model,data)
        viewer = sys.modules['__main__'].viewer
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = model.body('pelvis').id
            viewer.cam.distance = 3.
            viewer.cam.elevation = -20.
            viewer.cam.azimuth = 130.
        print('PolicyBridge: implicit PD active',model.actuator_gainprm[0].tolist(),flush=True)
        self.last_state_publish = time.perf_counter()
        def watch_publication():
            last_dump = None
            with (ROOT/'logs/native_stalls.log').open('a', encoding='utf8') as log:
                while viewer.is_running():
                    sleep(.1)
                    published = self.last_state_publish
                    if time.perf_counter()-published > .25 and published != last_dump:
                        log.write(f'Publication stalled at {time.perf_counter():.6f}; last={published:.6f}\n')
                        log.flush()
                        faulthandler.dump_traceback(file=log, all_threads=True)
                        last_dump = published
        threading.Thread(target=watch_publication, name='publication_watchdog', daemon=True).start()

    def LowCmdHandler(self,msg):
        # Configured affine actuators perform implicit PD every physics substep.
        with sys.modules['__main__'].locker:
            if msg.reserve[0] != self.reset_id:return
            if msg.reserve[2] > self.completed_seq:
                self.mj_data.ctrl[:] = [m.q for m in msg.motor_cmd[:self.num_motor]]
                self.command_seq = msg.reserve[2]
                self.command_ready.set()

    def PublishLowState(self):
        with sys.modules['__main__'].locker:
            if RESET.exists():
                mtime = RESET.stat().st_mtime_ns
                if mtime != self.reset_mtime:
                    request = json.loads(RESET.read_text())
                    if request['id'] != self.reset_id:
                        PhysicsTracker.reset(self.helper,np.array(request['qpos']))
                        self.mj_data.ctrl[:] = self.mj_data.qpos[7:]
                        self.reset_id = request['id']
                        self.command_seq = self.completed_seq = 0
                        self.command_ready.clear()
                        print('Session initialized',self.reset_id,flush=True)
                    self.reset_mtime = mtime
            self.low_state.tick = int(round(self.mj_data.time*1000))
            self.low_state.reserve[0] = self.reset_id
            self.low_state.reserve[1] = int(PhysicsTracker.fallen(self.helper))
            self.low_state.reserve[2] = self.completed_seq
            self.low_state.reserve[3] = int(round(self.helper.initial_floor_offset*1e6))
            state = read_robot_state(self.mj_data,16)
            for i in range(self.num_motor):
                self.low_state.motor_state[i].q = float(self.mj_data.qpos[7+i])
                self.low_state.motor_state[i].dq = float(self.mj_data.qvel[6+i])
                self.low_state.motor_state[i].tau_est = float(self.mj_data.actuator_force[i])
            self.low_state.imu_state.quaternion[:] = self.mj_data.qpos[3:7].tolist()
            self.low_state.imu_state.gyroscope[:] = state['root_local_ang_vel'].tolist()
            self.low_state_puber.Write(self.low_state)

    def PublishHighState(self):
        with sys.modules['__main__'].locker:
            self.high_state.stamp.sec = int(self.mj_data.time)
            self.high_state.stamp.nanosec = int((self.mj_data.time%1)*1e9)
            self.high_state.progress = float(self.completed_seq)
            self.high_state.error_code = self.reset_id  # Local simulation epoch, not a hardware error code.
            self.high_state.position[:] = self.mj_data.qpos[:3].tolist()
            self.high_state.velocity[:] = self.mj_data.qvel[:3].tolist()
            self.high_state.imu_state.quaternion[:] = self.mj_data.body('torso_link').xquat.tolist()
            self.high_state_puber.Write(self.high_state)
            self.last_state_publish = time.perf_counter()


upstream.UnitreeSdk2Bridge = PolicyBridge


def paced_sleep(seconds):
    # Control requests pace each complete 20-substep block; sleeping again for
    # each substep would add scheduling delay to the upstream control contract.
    main = sys.modules.get('__main__')
    if threading.current_thread().name.endswith('(SimulationThread)') and hasattr(main,'mj_data'):
        return
    if ENABLED and BRIDGE is not None and threading.current_thread().name.endswith('(PhysicsViewerThread)'):
        return  # SnapshotViewer waits for the next completed native state without holding the physics lock.
    return sleep(seconds)


time.sleep = paced_sleep


def synchronized_step(model,data,*args,**kwargs):
    """One DDS action advances exactly 20 substeps, as the upstream runner does."""
    bridge = BRIDGE
    if bridge is None or data is not bridge.mj_data:
        return original_step(model,data,*args,**kwargs)
    locker = sys.modules['__main__'].locker
    if bridge.command_seq <= bridge.completed_seq:
        bridge.command_ready.clear()
        locker.release()
        bridge.command_ready.wait(.1)
        locker.acquire()
        if bridge.command_seq <= bridge.completed_seq:return
    original_step(model,data,nstep=20)
    bridge.completed_seq = bridge.command_seq
    bridge.render_ready.set()
    if ENABLED:
        emit('physics', reset_id=bridge.reset_id, sequence=bridge.completed_seq,
             completed_ns=time.perf_counter_ns(), simulation_time=float(data.time), qpos=data.qpos.tolist())
    locker.release()
    try:
        bridge.PublishLowState()
        bridge.PublishHighState()
    finally:locker.acquire()


mujoco.mj_step = synchronized_step
