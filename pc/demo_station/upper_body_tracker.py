"""Real observed arm directions -> G1 references; legs are declared fixed, not measured."""
from pathlib import Path
import time
import mujoco
import numpy as np
from scipy.optimize import least_squares

G1 = Path(r'E:\FPGAchina\G1_1080p_Candidate_20261020')
SCOPE = 'standing_upper_body'
HOLD_SECONDS = .2
SHOULDER_BASIS_SECONDS = .5
MAX_ARM_SPEED = 2.0


def observed(joint):
    xyz = joint.get('xyz')
    reprojection = joint.get('reprojection_px')
    return (joint.get('valid') is True and joint.get('estimated') is False
            and xyz is not None and np.asarray(xyz).shape == (3,) and np.isfinite(xyz).all()
            and reprojection is not None and len(reprojection) == 2
            and np.isfinite(reprojection).all() and max(reprojection) <= 5.)


def packet_joints(packet):
    if packet.get('schema') != 'human_skeleton/1.0' or packet.get('unit') != 'm':
        raise ValueError('Upper-body input requires the original skeleton schema in metres')
    if packet.get('coordinate_system') != 'world_right_handed_X_forward_Y_left_Z_up':
        raise ValueError('Unsupported upper-body coordinate system')
    return dict(zip(packet['joint_names'], packet['joints']))


def shoulder_basis(joints):
    if not all(observed(joints[name]) for name in ('left_shoulder', 'right_shoulder')):
        return None
    left, right = [np.asarray(joints[name]['xyz'], dtype=float) for name in ('left_shoulder', 'right_shoulder')]
    lateral = left-right; lateral[2] = 0.
    width = np.linalg.norm(lateral)
    if not .12 <= width <= .65:
        return None
    lateral /= width
    up = np.array([0., 0., 1.]); forward = np.cross(lateral, up)
    return np.column_stack((forward, lateral, up))


def directions(packet, side, basis=None):
    joints = packet_joints(packet)
    if basis is None: basis = shoulder_basis(joints)
    if basis is None: return None, 'missing_or_expired_true_shoulder_basis'
    names = [side+'_'+name for name in ('shoulder', 'elbow', 'wrist')]
    if not all(observed(joints[name]) for name in names):
        return None, 'missing_true_own_shoulder_elbow_or_wrist'
    shoulder, elbow, wrist = [np.asarray(joints[name]['xyz'], dtype=float) for name in names]
    segments = np.array([elbow-shoulder, wrist-elbow])
    lengths = np.linalg.norm(segments, axis=1)
    if not np.all((lengths >= .10) & (lengths <= .55)):
        return None, 'invalid_observed_arm_lengths'
    return (segments/lengths[:, None]) @ basis, 'observed'


class UpperBodyTracker:
    def __init__(self, message):
        if message.get('motion_mode') != SCOPE:
            raise ValueError('Explicit standing_upper_body mode is required')
        self.model = mujoco.MjModel.from_xml_path(str(G1/'g1_realtime/vendor/gmr/unitree_g1/g1_mocap_29dof.xml'))
        self.data = mujoco.MjData(self.model)
        self.qpos = self.model.qpos0.copy()
        self.fixed_reference = self.qpos.copy()
        self.last_stamp = None
        self.smooth_seconds = float(message.get('smooth_ms', 80))/1000
        if not 0 <= self.smooth_seconds <= .25: raise ValueError('Upper-body smoothing must be between 0 and 250ms')
        self.last_observed = dict(left=None, right=None)
        self.last_updated = dict(left=None, right=None)
        self.basis = self.basis_stamp = None
        self.arms = {}
        for side in ('left', 'right'):
            ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, side+'_'+name+'_joint')
                   for name in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow')]
            bodies = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, side+'_'+name+'_link')
                      for name in ('shoulder_roll', 'elbow', 'wrist_roll')]
            self.arms[side] = (self.model.jnt_qposadr[ids], self.model.jnt_range[ids], bodies)
        self.metadata = dict(type='ready', name='真人上肢跟随 · 固定站姿参考', motion_mode=SCOPE,
            reference_scope=SCOPE, fixed_lower_body_reference=True, lower_body_measured=False,
            recovered_skeleton=False, channel_count=0, human_names=[], human_parents=[],
            shoulder_basis_max_age_ms=SHOULDER_BASIS_SECONDS*1000,
            adapter='true observed arm directions only; fixed robot standing root/waist/legs',
            body_names=[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(self.model.nbody)],
            body_parents=self.model.body_parentid.tolist(),
            joint_names=[mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(1, self.model.njnt)],
            joint_limits=self.model.jnt_range[1:].tolist(), quaternion_order='wxyz', position_unit='m')

    def arm_directions(self, side, values):
        indexes, _, bodies = self.arms[side]
        self.data.qpos[:] = self.qpos
        self.data.qpos[indexes] = values
        mujoco.mj_kinematics(self.model, self.data)
        segments = np.diff(self.data.xpos[bodies], axis=0)
        return segments/np.linalg.norm(segments, axis=1)[:, None]

    def solve(self, packet):
        start = time.perf_counter()
        stamp = int(packet['timestamp_ns'])/1e9
        if self.last_stamp is not None and stamp <= self.last_stamp:
            raise ValueError('Upper-body observation times must increase')
        dt = min(.1, stamp-self.last_stamp) if self.last_stamp is not None else .02
        current_basis = shoulder_basis(packet_joints(packet))
        if current_basis is not None:
            self.basis, self.basis_stamp = current_basis, stamp
        basis_age = None if self.basis_stamp is None else stamp-self.basis_stamp
        basis = self.basis if basis_age is not None and basis_age <= SHOULDER_BASIS_SECONDS+1e-9 else None
        states = {}
        for side in ('left', 'right'):
            target, reason = directions(packet, side, basis) if basis is not None else (None, 'missing_or_expired_true_shoulder_basis')
            indexes, limits, _ = self.arms[side]
            if target is not None:
                previous = self.qpos[indexes].copy()
                def residual(values):
                    return np.r_[(self.arm_directions(side, values)-target).ravel(), .035*(values-previous)]
                result = least_squares(residual, previous, bounds=(limits[:, 0]+1e-6, limits[:, 1]-1e-6),
                                       max_nfev=24, ftol=1e-5, xtol=1e-5, gtol=1e-5)
                error = float(np.linalg.norm((self.arm_directions(side, result.x)-target), axis=1).max())
                if np.isfinite(result.x).all() and error <= .65:
                    delta = (result.x-previous)*dt/(self.smooth_seconds+dt)
                    self.qpos[indexes] = previous+np.clip(delta, -MAX_ARM_SPEED*dt, MAX_ARM_SPEED*dt)
                    self.last_updated[side] = stamp
                    if current_basis is not None: self.last_observed[side] = stamp
                else:
                    target = None; reason = 'arm_direction_ik_rejected'
            age = None if self.last_observed[side] is None else max(0., stamp-self.last_observed[side])
            update_age = None if self.last_updated[side] is None else max(0., stamp-self.last_updated[side])
            state = 'tracking' if target is not None else ('holding' if update_age is not None and update_age <= HOLD_SECONDS+1e-9 else 'paused')
            states[side] = dict(state=state, reason=reason, observed=target is not None and current_basis is not None,
                                reference_updated=target is not None, arm_points_observed=target is not None,
                                shoulder_basis_observed=current_basis is not None,
                                shoulder_basis_age_ms=None if basis_age is None else basis_age*1000,
                                shoulder_basis_cached=target is not None and current_basis is None,
                                reference_update_age_ms=None if update_age is None else update_age*1000,
                                observation_age_ms=None if age is None else age*1000,
                                last_observed_timestamp_ns=None if self.last_observed[side] is None else round(self.last_observed[side]*1e9),
                                source_seq=packet['seq'] if target is not None else None)
        self.last_stamp = stamp
        return dict(qpos=self.qpos.tolist(), human_pos=[], solve_ms=(time.perf_counter()-start)*1000,
                    upper_body=dict(scope=SCOPE, fixed_lower_body_reference=True, lower_body_measured=False,
                                    arms=states, source_seq=packet['seq'], timestamp_ns=packet['timestamp_ns']))
