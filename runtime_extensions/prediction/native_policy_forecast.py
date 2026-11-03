"""Experimental relaxation: forecast policy lookahead, never current observation.

User-authorized P3 practical experiment, frozen 2026-10-24.
Not eligible for the original already-observed-coverage acceptance rule.
Arrival revision is audited rather than rejected; forecast speeds saturate at unchanged bounds; source gap/limits remain hard gates.
No native libraries, physical resets, time retiming, or implicit hold fallback.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation

PARAMETERS = dict(max_prediction_s=.160, decay_s=.080,
    joint_limit_tolerance_rad=1e-10,
    maximum_observation_gap_s=.200, root_speed_mps=.5, root_angular_speed_radps=1.5, joint_speed_radps=2.5,
    current_target='must remain in already observed coverage',
    admission='saturate only forecast velocity; reject source gap above existing quality 200ms or limits; revision is diagnostic',
    provenance='prediction is not observation and never reliable recovery')


class ForecastUnavailable(ValueError):
    pass


def rotation(q):
    return Rotation.from_quat(np.asarray(q)[[1,2,3,0]])


@dataclass
class Snapshot:
    stamp: float
    pose: np.ndarray
    root_velocity: np.ndarray
    local_angular_velocity: np.ndarray
    joint_velocity: np.ndarray

    def sample(self, timestamp):
        h=float(timestamp)-self.stamp
        if h < -1e-9 or h > PARAMETERS['max_prediction_s']+1e-9:
            raise ForecastUnavailable('prediction_horizon_exceeded')
        h=max(0.,h)
        decay=np.exp(-h/PARAMETERS['decay_s'])
        distance=PARAMETERS['decay_s']*(-np.expm1(-h/PARAMETERS['decay_s']))
        q=self.pose.copy()
        q[:3]+=self.root_velocity*distance
        q[3:7]=(rotation(q[3:7])*Rotation.from_rotvec(self.local_angular_velocity*distance)).as_quat()[[3,0,1,2]]
        q[7:]+=self.joint_velocity*distance
        return q,self.joint_velocity*decay,self.root_velocity*decay,self.local_angular_velocity*decay


class PolicyForecast:
    """Observe an existing buffer; failed admission is explicit, never clamped away."""
    def __init__(self):
        self.snapshot=None
        self.evidence=dict(admitted=False,reason='uninitialized')

    def observe(self, buffer):
        if len(buffer.times)<2:
            self.evidence=dict(admitted=False,reason='two_current_observations_required')
            return self.evidence
        stamp=float(buffer.times[-1])
        if self.snapshot is not None and stamp<=self.snapshot.stamp:
            raise ValueError('Observation source times must strictly increase')
        previous_stamp=float(buffer.times[-2]);dt=stamp-previous_stamp
        if dt<=0:raise ValueError('Observation source times must strictly increase')
        previous,_=buffer.sample(previous_stamp)
        pose,velocity=buffer.sample(stamp)
        pose=np.asarray(pose,dtype=float).copy();velocity=np.asarray(velocity,dtype=float).copy()
        root_velocity=(pose[:3]-previous[:3])/dt
        omega=(rotation(previous[3:7]).inv()*rotation(pose[3:7])).as_rotvec()/dt
        raw_root=root_velocity.copy();raw_omega=omega.copy();raw_velocity=velocity.copy()
        root_velocity*=min(1.,PARAMETERS['root_speed_mps']/max(np.linalg.norm(root_velocity),1e-12))
        omega*=min(1.,PARAMETERS['root_angular_speed_radps']/max(np.linalg.norm(omega),1e-12))
        velocity=np.clip(velocity,-PARAMETERS['joint_speed_radps'],PARAMETERS['joint_speed_radps'])
        speed_velocity=velocity.copy();joint_limits=None;invalid_endpoint=False
        if hasattr(buffer,'model'):
            limited=buffer.model.jnt_limited[1:].astype(bool);bounds=buffer.model.jnt_range[1:]
            excess=np.maximum(bounds[limited,0]-pose[7:][limited],pose[7:][limited]-bounds[limited,1])
            invalid_endpoint=bool(np.any(excess>PARAMETERS['joint_limit_tolerance_rad']))
            if not invalid_endpoint:
                distance=PARAMETERS['decay_s']*(-np.expm1(-PARAMETERS['max_prediction_s']/PARAMETERS['decay_s']))
                lower=(bounds[limited,0]-pose[7:][limited])/distance
                upper=(bounds[limited,1]-pose[7:][limited])/distance
                bounded=np.clip(velocity[limited],lower,upper)
                changed=bounded!=velocity[limited]
                bounded[changed]=np.nextafter(bounded[changed],0.)  # Conservative floating-point endpoint.
                velocity[limited]=bounded
            joint_limits=dict(limited=limited.tolist(),bounds=bounds.tolist(),endpoint_joint_pose=pose[7:].tolist(),
                endpoint_valid=not invalid_endpoint,maximum_prediction_s=PARAMETERS['max_prediction_s'],
                numerical_tolerance_rad=PARAMETERS['joint_limit_tolerance_rad'],
                endpoint_maximum_excess_rad=max(0.,float(np.max(excess))) if len(excess) else 0.)
        new=Snapshot(stamp,pose,root_velocity,omega,velocity)
        evidence=dict(admitted=True,reason=None,rejection_reasons=[],source_dt_s=dt,observation_stamp=stamp,
            speed_mps=float(np.linalg.norm(root_velocity)),angular_speed_radps=float(np.linalg.norm(omega)),
            maximum_joint_speed_radps=float(np.max(abs(velocity))),
            prediction_never_reliable=True,observation_modified=False,revision_checked=False)
        evidence.update(raw_root_velocity=raw_root.tolist(),raw_local_angular_velocity=raw_omega.tolist(),
            raw_joint_velocity=raw_velocity.tolist(),prediction_root_velocity=root_velocity.tolist(),
            prediction_local_angular_velocity=omega.tolist(),prediction_joint_velocity=velocity.tolist(),
            velocity_saturated=bool(np.any(raw_root!=root_velocity) or np.any(raw_omega!=omega) or np.any(raw_velocity!=velocity)),
            endpoint_joint_velocity_change_radps=float(np.max(abs(raw_velocity-velocity))),
            prediction_endpoint_pose_unchanged=True,
            velocity_scope='Only forecast derivative is saturated; observed buffer and actual recovery poses are unchanged.')
        evidence.update(speed_limited_joint_velocity=speed_velocity.tolist(),joint_limit_envelope=joint_limits,
            joint_limit_velocity_changed=bool(np.any(speed_velocity!=velocity)))
        if invalid_endpoint:
            evidence.update(admitted=False,reason='observed_endpoint_joint_limit')
            evidence['rejection_reasons'].append('observed_endpoint_joint_limit')
        if dt>PARAMETERS['maximum_observation_gap_s']+1e-9:
            evidence.update(admitted=False,reason='observation_gap_requires_repriming')
            evidence['rejection_reasons'].append('observation_gap_requires_repriming')
        old=self.snapshot
        overlap_end=min(stamp+.160,old.stamp+.160) if old is not None else stamp
        if old is not None and overlap_end>stamp+1e-9:
            errors=[]
            for query in np.linspace(stamp,overlap_end,21):
                a=old.sample(query)[0];b=new.sample(query)[0]
                errors.append((np.linalg.norm(b[:3]-a[:3]),np.linalg.norm((rotation(a[3:7]).inv()*rotation(b[3:7])).as_rotvec()),np.max(abs(b[7:]-a[7:]))))
            maximum=np.max(errors,axis=0)
            upper=maximum.copy()
            # Translation/joint differences share one exponential time basis:
            # norms are convex in exp(-t/tau), so endpoint maxima bound them.
            # SO(3) distance is Lipschitz in the sum of angular path speeds.
            upper[1]+=(np.linalg.norm(old.local_angular_velocity)+np.linalg.norm(omega))*(overlap_end-stamp)/40
            limits=np.array([.5,1.5,2.5])*dt
            evidence.update(revision_checked=True,revision_maximum=maximum.tolist(),revision_upper_bound=upper.tolist(),revision_limits=limits.tolist(),
                revision_scope='Entire common forecast interval: convex exponential difference for position/joints; angular grid plus path-speed Lipschitz bound for SO3')
            if np.any(upper>limits+1e-9):
                evidence.update(arrival_revision_exceeded=True,
                    revision_diagnostic='P3 authorized practical test: old strict revision admission would reject; no measurement modified')
        self.snapshot=new;self.evidence=evidence
        return evidence

    def sample(self, buffer, current_target, timestamp):
        current_target=float(current_target);timestamp=float(timestamp)
        if current_target<buffer.times[0]-1e-9 or current_target>buffer.times[-1]+1e-9:
            raise ForecastUnavailable('current_target_not_observed')
        if timestamp<current_target-1e-9:
            raise ValueError('Policy lookahead cannot precede current target')
        if timestamp<=buffer.times[-1]+1e-9:
            q,v=buffer.sample(min(timestamp,buffer.times[-1]))
            return q,v,dict(source='observed_interpolation',prediction_horizon_s=0.,reliable_recovery=False)
        if not self.evidence['admitted']:
            raise ForecastUnavailable(self.evidence['reason'])
        q,v,root_v,omega=self.snapshot.sample(timestamp)
        if hasattr(buffer,'model'):
            limited=buffer.model.jnt_limited[1:].astype(bool)
            bounds=buffer.model.jnt_range[1:]
            excess=np.maximum(bounds[limited,0]-q[7:][limited],q[7:][limited]-bounds[limited,1])
            if np.any(excess>PARAMETERS['joint_limit_tolerance_rad']):
                self.evidence['rejected_query']=dict(timestamp=float(timestamp),joint_pose=q[7:].tolist(),
                    maximum_excess_rad=float(np.max(excess)),numerical_tolerance_rad=PARAMETERS['joint_limit_tolerance_rad'])
                raise ForecastUnavailable('predicted_joint_limit')
        return q,v,dict(source='bounded_causal_prediction',prediction_horizon_s=timestamp-self.snapshot.stamp,
            root_velocity=root_v.tolist(),root_local_angular_velocity=omega.tolist(),reliable_recovery=False)
