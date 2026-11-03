"""G1 physics adapter; observation and PD model setup use upstream code unchanged."""
from pathlib import Path
import sys
import time
import numpy as np
import mujoco
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import PchipInterpolator

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'upstream'))
from deployment.test_tracker_mujoco import load_mujoco_model, read_robot_state
from deployment.tracker_inputs import build_tracker_onnx_inputs

MODEL = ROOT / 'upstream/data/pretrained_models/motion_tracker/g1-bones-deploy/compiled_models/unified_pipeline.onnx'


class PhysicsTracker:
    def __init__(self):
        self.meta = yaml.safe_load(MODEL.with_suffix('.yaml').read_text())
        c, t = self.meta['control'], self.meta['timing']
        self.model, self.data = load_mujoco_model(self.meta['robot']['mjcf_path'], c['stiffness'], c['damping'], t['physics_dt'])
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(MODEL), sess_options=opts, providers=['CPUExecutionProvider'])
        self.names = [x.name for x in self.session.get_inputs()]
        self.dt = t['control_dt']
        self.offsets = np.array(self.meta['motion']['future_dt_seconds'])
        self.prev = None
        self.joints = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, self.model.njnt)]
        assert self.joints == self.meta['joint_names'], 'Joint ordering must match released model'
        assert c['action_ema_alpha'] == 1 and c['pd_target_max_accel'] is None

    def reset(self, qpos):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        # Recovered BVH can have a different floor origin. Correct substantial
        # initial penetration once; never overwrite floating-base state in step().
        feet = [i for i in range(self.model.ngeom)
                if 'foot' in (mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or '')
                and self.model.geom_contype[i]]
        assert all(self.model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE for i in feet)
        bottom = min(self.data.geom_xpos[i, 2] - abs(self.data.geom_xmat[i, 8])*self.model.geom_size[i, 1]
                     - self.model.geom_size[i, 0] for i in feet)
        self.initial_floor_offset = .002-bottom if bottom < -.02 else 0.
        self.data.qpos[2] += self.initial_floor_offset
        mujoco.mj_forward(self.model, self.data)
        self.prev = None

    def step(self, refs):
        start = time.perf_counter()
        r = read_robot_state(self.data, self.meta['robot']['anchor_body_index'])
        inputs = build_tracker_onnx_inputs(
            onnx_input_names=self.names, dof_pos=r['dof_pos'], dof_vel=r['dof_vel'],
            anchor_rot=r['body_rot'][self.meta['robot']['anchor_body_index']],
            root_local_ang_vel=r['root_local_ang_vel'], future_refs=refs,
            onnx_name_to_key=self.meta['_runtime']['onnx_name_to_in_key'],
            anchor_body_index=self.meta['robot']['anchor_body_index'], num_dofs=29, prev_actions=self.prev)
        out = self.session.run(None, inputs)
        self.prev = out[1].reshape(29).copy()
        self.data.ctrl[:] = self.prev
        for _ in range(self.meta['timing']['decimation']):
            mujoco.mj_step(self.model, self.data)
        return (time.perf_counter() - start) * 1000

    def fallen(self):
        q = self.data.qpos
        if not np.isfinite(q).all() or q[2] < .18 or (q[2] < .3 and self.data.xmat[1, 8] < .5):
            return True
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if self.model.geom_type[a] == mujoco.mjtGeom.mjGEOM_PLANE:
                a, b = b, a
            if self.model.geom_type[b] == mujoco.mjtGeom.mjGEOM_PLANE:
                body = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[a]))
                if body in ('pelvis', 'torso_link', 'head'):
                    return True
        return False


class ReferenceBuffer:
    """Timestamped poses; interpolate only between already received samples."""
    def __init__(self, model, interpolation='linear'):
        self.model = model
        self.fk = mujoco.MjData(model)
        self.times, self.poses = [], []
        self.interpolation = interpolation
        self.curve = None

    def push(self, timestamp, qpos):
        if self.times and timestamp <= self.times[-1]:
            raise ValueError('Input timestamps must increase')
        self.times.append(float(timestamp))
        self.poses.append(np.asarray(qpos).copy())
        if len(self.times) > 32:
            self.times.pop(0)
            self.poses.pop(0)
        if self.interpolation == 'pchip' and len(self.times) >= 3:
            self.curve = PchipInterpolator(self.times, np.array(self.poses)[:, 7:], axis=0, extrapolate=False)

    def sample(self, timestamp):
        if not self.times or timestamp < self.times[0]-1e-8 or timestamp > self.times[-1]+1e-8:
            raise ValueError('Reference window unavailable')
        hi = min(max(1, int(np.searchsorted(self.times, timestamp, side='right'))), len(self.times)-1)
        if hi == 0:
            return self.poses[0].copy(), np.zeros(29)
        lo = hi-1
        a,b = self.poses[lo],self.poses[hi]
        dt = self.times[hi]-self.times[lo]
        w = np.clip((timestamp-self.times[lo])/dt,0,1)
        q = a*(1-w)+b*w
        rot = Slerp([0,1], Rotation.from_quat(np.array([a[3:7],b[3:7]])[:,[1,2,3,0]]))([w]).as_quat()[0]
        q[3:7] = rot[[3,0,1,2]]
        velocity = (b[7:]-a[7:])/dt
        if self.curve is not None:
            q[7:] = self.curve(timestamp)
            velocity = self.curve(timestamp, 1)
        return q, velocity

    def references(self, timestamps):
        joints,vel,rots = [],[],[]
        for stamp in timestamps:
            q,v = self.sample(stamp)
            self.fk.qpos[:] = q
            mujoco.mj_forward(self.model,self.fk)
            joints.append(q[7:]);vel.append(v)
            rots.append(self.fk.xquat[1:, [1,2,3,0]].copy())
        return dict(dof_pos=np.array(joints,dtype=np.float32), dof_vel=np.array(vel,dtype=np.float32), body_rot=np.array(rots,dtype=np.float32))
