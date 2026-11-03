"""External policy controller. No viewer, simulator import, or mj_step here."""
from pathlib import Path
import sys
import threading
import multiprocessing
from multiprocessing.managers import BaseManager
import os
import time
import json
import uuid
import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent/'g1_physical'))
from physical import PhysicsTracker, build_tracker_onnx_inputs
from windows_timing import enable_timer
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC


class DDSController:
    def __init__(self):
        self.stop_timer = enable_timer()
        self.policy = PhysicsTracker()
        self.low = self.high = None
        self.received_at = 0.
        self.received = self.sent = 0
        self.command_retries = 0
        self.lock = threading.RLock()
        self.advance_lock = threading.RLock()
        self.stop = threading.Event()
        self.command_seq = 0
        self.reset_id = 0
        self.refs = None
        self.last_error = None
        self.targets = np.zeros(29)
        self.advance_cost = {}
        self.started = time.perf_counter()
        self.kp = np.array(self.policy.meta['control']['stiffness'])
        self.kd = np.array(self.policy.meta['control']['damping'])
        self.crc = CRC()
        model = mujoco.MjModel.from_xml_path(str(ROOT/'unitree_mujoco-main/unitree_robots/g1/scene_29dof.xml'))
        motor_names = [model.joint(int(model.actuator_trnid[i, 0])).name for i in range(model.nu)]
        assert motor_names == self.policy.joints, 'DDS motor order must equal policy order'
        ChannelFactoryInitialize(int(os.environ.get('G1_DDS_DOMAIN','74')))
        self.publisher = ChannelPublisher('rt/lowcmd', LowCmd_)
        self.publisher.Init()
        def low(msg):
            with self.lock:
                self.low = msg
                self.received_at = time.perf_counter()
                self.received += 1
        def high(msg):
            with self.lock: self.high = msg
        self.low_sub = ChannelSubscriber('rt/lowstate', LowState_)
        self.low_sub.Init(low, 1)
        self.high_sub = ChannelSubscriber('rt/sportmodestate', SportModeState_)
        self.high_sub.Init(high, 1)
        self.control_thread = threading.Thread(target=self.control_loop, daemon=True)
        self.control_thread.start()

    def read_state(self):
        deadline = time.perf_counter()+.5
        while True:
            with self.lock:
                low,high = self.low,self.high
                if low is not None and high is not None and time.perf_counter()-self.received_at < .5 and low.reserve[2] == int(high.progress) and low.reserve[0] == high.error_code:break
            if time.perf_counter()>deadline:raise ValueError('未收到同一控制周期的官方仿真 DDS 状态')
            time.sleep(.001)
        q = np.r_[high.position, low.imu_state.quaternion, [m.q for m in low.motor_state[:29]]]
        v = np.r_[high.velocity, low.imu_state.gyroscope, [m.dq for m in low.motor_state[:29]]]
        return q, v

    def control_loop(self):
        # Watch the parent lifetime; inference runs only for received references.
        while not self.stop.is_set():
            parent = multiprocessing.parent_process()
            if parent is not None and not parent.is_alive(): os._exit(0)
            self.stop.wait(.1)

    def advance(self,refs):
        with self.advance_lock:
            start = time.perf_counter()
            self.refs = refs
            q,v = self.read_state()
            with self.lock:
                if self.low.reserve[0] != self.reset_id:raise ValueError('等待本次仿真初始化')
            observed = time.perf_counter()
            p = self.policy
            with self.lock:anchor = np.array(self.high.imu_state.quaternion)[[1,2,3,0]]
            inputs = build_tracker_onnx_inputs(onnx_input_names=p.names,dof_pos=q[7:],dof_vel=v[6:],
                anchor_rot=anchor,root_local_ang_vel=v[3:6],future_refs=refs,
                onnx_name_to_key=p.meta['_runtime']['onnx_name_to_in_key'],
                anchor_body_index=p.meta['robot']['anchor_body_index'],num_dofs=29,prev_actions=p.prev)
            out = p.session.run(None,inputs)
            inferred = time.perf_counter()
            p.prev = out[1].reshape(29).copy()
            with self.lock:
                self.targets = p.prev.copy()
                self.command_seq += 1
                sequence = self.command_seq
            self.publish_command()
            published = time.perf_counter()
            deadline = published+.5
            while time.perf_counter()<deadline:
                with self.lock:
                    if self.low is not None and self.low.reserve[2] == sequence:break
                # A lost command must not strand a synchronized simulator.
                # Reuse the epoch and sequence: the bridge executes it once.
                if time.perf_counter()-published >= .04:
                    self.publish_command()
                    self.command_retries += 1
                    published = time.perf_counter()
                time.sleep(.001)
            else:
                with self.lock:
                    self.last_error = dict(kind='control_cycle_timeout',requested_sequence=sequence,
                        low_sequence=self.low.reserve[2] if self.low else None,
                        high_sequence=int(self.high.progress) if self.high else None,
                        state_age_ms=(time.perf_counter()-self.received_at)*1000,
                        elapsed_ms=(time.perf_counter()-start)*1000)
                print(json.dumps(self.last_error),flush=True)
                raise ValueError('官方仿真未完成本次物理控制周期')
            q,v = self.read_state()
            self.advance_cost = dict(state_ms=(observed-start)*1000,infer_ms=(inferred-observed)*1000,ack_ms=(time.perf_counter()-inferred)*1000)
            return q,v,self.telemetry()

    def publish_command(self):
        with self.lock:
            targets = self.targets.copy()
            reset_id,sequence = self.reset_id,self.command_seq
        msg = unitree_hg_msg_dds__LowCmd_()
        msg.mode_pr = 0
        msg.reserve[0],msg.reserve[2] = reset_id,sequence
        for i in range(29):
            motor = msg.motor_cmd[i]
            motor.mode, motor.q = 1, float(targets[i])
            motor.kp, motor.kd = float(self.kp[i]), float(self.kd[i])
        msg.crc = self.crc.Crc(msg)
        self.publisher.Write(msg)
        self.sent += 1

    def stop_reference(self):
        with self.lock:
            self.refs = None

    def prepare_session(self,qpos):
        # Match the original simulator's once-per-session initialization. This
        # never overwrites floating-base state during tracking or hides falls.
        self.advance_lock.acquire()
        try:
            with self.lock:
                self.refs = {}
                self.policy.prev = None
                self.command_seq = 0
                self.targets = np.asarray(qpos[7:]).copy()
                self.reset_id = uuid.uuid4().int & 0x7fffffff
            path = Path(__file__).parent/'session_reset.json'
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps({'id':self.reset_id,'qpos':np.asarray(qpos).tolist()}))
            temporary.replace(path)
            deadline = time.perf_counter()+2
            while time.perf_counter()<deadline:
                with self.lock:
                    confirmed = self.low is not None and self.low.reserve[0] == self.reset_id
                if confirmed:return self.read_state()
                time.sleep(.002)
            raise ValueError('官方仿真未确认本次初始化')
        finally:
            self.advance_lock.release()

    def telemetry(self):
        with self.lock:
            return dict(simulation_time=self.low.tick/1000 if self.low else 0.,
                        fallen=bool(self.low.reserve[1]) if self.low else False,
                        session_reset=self.low.reserve[0] if self.low else 0,
                        initial_floor_offset=self.low.reserve[3]/1e6 if self.low else 0.)

    def reset_controller(self):
        self.command_seq = 0
        with self.lock:
            self.refs = None
            self.targets[:] = 0
            self.policy.prev = None
            self.low = self.high = None

    def metrics(self):
        return dict(dds_sent=self.sent,dds_received=self.received,command_retries=self.command_retries,error=self.last_error,policy_pid=os.getpid(),simulation_paused=self.refs is None,advance_cost=self.advance_cost,**self.telemetry())

    def model_spec(self):
        return self.policy.meta

    def close(self):
        self.stop.set()
        self.control_thread.join(2)
        self.low_sub.Close()
        self.high_sub.Close()
        self.stop_timer()


CONTROLLER = None
FK_MODEL = FK_META = None


class ControllerManager(BaseManager):
    pass


ControllerManager.register('DDSController', DDSController)


class ExternalTracker(PhysicsTracker):
    def __init__(self):
        self.controller = CONTROLLER
        # Reuse the loaded model; this adapter never performs policy inference.
        self.model, self.meta = FK_MODEL, FK_META
        self.data = mujoco.MjData(self.model)
        self.dt = self.meta['timing']['control_dt']
        self.offsets = np.array(self.meta['motion']['future_dt_seconds'])
        self.joints = self.meta['joint_names']
        self.initial_floor_offset = 0.

    def reset(self, qpos):
        actual, velocity = self.controller.prepare_session(qpos)
        self.initial_floor_offset = self.controller.telemetry()['initial_floor_offset']
        self.data.qpos[:] = actual
        self.data.qvel[:] = velocity
        mujoco.mj_kinematics(self.model, self.data)

    def step(self, refs):
        start = time.perf_counter()
        q,v,state = self.controller.advance(refs)
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        mujoco.mj_kinematics(self.model, self.data)
        self.data.time = state['simulation_time']
        return (time.perf_counter()-start)*1000

    def fallen(self):
        return self.controller.telemetry()['fallen']
