"""Native tick with observed-future guard; original source SHA256 241bd5490577fd27ec01d99321175fcaa2915c686edeee371bd4961ab09d47c1"""
import time,os
import numpy as np
import mujoco
from future_prediction import predicted_references
from native_policy_forecast import ForecastUnavailable

def tick_locked(self):
    terminal=getattr(self,'_tick_terminal',bool(self.ended))
    if self.wall_origin is None or self.finished:
        return None
    start=time.perf_counter()
    if not self.started:
        if self.buffer.times[-1]<max(self.physics.offsets) and not terminal:return None
        self.physics.reset(self.buffer.sample(self.buffer.times[0])[0]);self.started=True
    # Initialize within the reference buffer instead of adding reset/DDS
    # handshake time after it. The first physics step still waits for lag.
    if time.perf_counter()<self.wall_origin+self.lag:return None
    last=self.buffer.times[-1]
    # Follow the timestamped source clock. A missing reference or a delayed
    # control tick must not permanently shift every later reference frame.
    source_now=max(0.,time.perf_counter()-self.wall_origin-self.lag)
    waiting=bool(source_now+max(self.physics.offsets)>last+1e-8 and not terminal)
    # P3 explicitly predicts only the policy future, at most 160ms.
    # Current references and terminal drainage remain observed.
    # P3 applies the same 200ms deficit to current real coverage; future is explicitly predicted.
    if not terminal and os.environ.get('G1_QUALITY_GUARD')=='1' and source_now-last>.2:
        raise ForecastUnavailable('observed_reference_coverage_more_than_200ms_late')
    if source_now<self.buffer.times[0]-1e-9:
        self.current_coverage_wait_count=getattr(self,'current_coverage_wait_count',0)+1
        return None  # Real recovery priming may finish one tick before its source clock.
    target_t=min(source_now,last)
    unguarded_missing_future=waiting
    # P3 records a short current clamp only at the latest real point, not last-160ms.
    # Missing current observations or forecast bounds raise an explicit pause.
    refs,forecast_evidence=predicted_references(self,target_t,terminal=terminal)
    forecast_evidence.update(nominal_current_target_time=source_now+self.source_origin,
        current_reference_extra_age_ms=(source_now-target_t)*1000,
        current_real_coverage_wait_count=getattr(self,'current_coverage_wait_count',0))
    waiting=False
    reference=self.buffer.sample(target_t)[0]
    if self.position_feedback:
        refs=self.position_feedback.apply(refs,reference,self.physics.data.qpos,self.physics.data.qvel,self.physics.dt)
        # The legacy filter state is unchanged; record the actual world rotvec
        # separately from that state and from the final policy quaternion.
        filtered=self.position_feedback.filtered
        forecast_evidence['position_feedback_ablation']=dict(
            variant='rollhalf',lateral_rotation_scale=.5,
            unchanged_filtered_xy_rad=filtered.tolist(),
            legacy_world_rotvec_rad=[float(-filtered[1]),float(filtered[0]),0.],
            applied_world_rotvec_rad=[float(-.5*filtered[1]),float(filtered[0]),0.],
            legacy_tilt_field_scope='reference_tilt_correction_deg remains the unscaled XY filter state; use applied_world_rotvec_rad for this candidate')
    height_target=None
    if self.height_feedback:
        height=float(np.interp(target_t,self.buffer.times,self.heights))
        refs,height_target=self.height_feedback.apply(refs,height,self.physics.data.qpos,self.physics.data.qvel,self.physics.dt)
    anchor=int(self.physics.meta['robot']['anchor_body_index'])
    forecast_evidence['policy_input_after_existing_feedback']=dict(
        dof_pos=np.asarray(refs['dof_pos']).tolist(),
        dof_vel=np.asarray(refs['dof_vel']).tolist(),
        anchor_body_index=anchor,
        anchor_body_quat_xyzw=np.asarray(refs['body_rot'])[:,anchor,:].tolist())
    self.physics.step(refs)
    self.t=target_t+self.physics.dt
    self.processed+=1
    if terminal and target_t>=last-1e-9:self.finished=True
    q=self.physics.data.qpos.copy()
    fallen=self.physics.fallen()
    if fallen:self.finished=True
    tilt=float(np.degrees(np.arccos(np.clip(self.physics.data.xmat[1,8],-1,1))))
    # Render the original G1 meshes using the actual simulated state, never the reference.
    render_q=q.copy();render_q[7:]=q[7:][self.reverse]
    self.render_data.qpos[:]=render_q
    mujoco.mj_forward(self.ik.model,self.render_data)
    i=min(max(1,int(np.searchsorted(self.buffer.times,target_t))),len(self.humans)-1)
    lo=max(0,i-1);span=self.buffer.times[i]-self.buffer.times[lo]
    w=np.clip((target_t-self.buffer.times[lo])/span,0,1) if span else 0
    human=self.humans[lo]*(1-w)+self.humans[i]*w
    elapsed=(time.perf_counter()-start)*1000
    return dict(type='pose',session=self.session,seq=self.processed,source_time=target_t+self.source_origin,
        client_time=self.client_origin+target_t*1000 if self.client_origin is not None else None,
        qpos=render_q.tolist(),root_pos=q[:3].tolist(),root_quat=q[3:7].tolist(),joint_pos=render_q[7:].tolist(),
        body_pos=self.render_data.xpos.tolist(),body_quat=self.render_data.xquat.tolist(),human_pos=human.tolist(),
        geom_pos=self.render_data.geom_xpos.tolist(),geom_mat=self.render_data.geom_xmat.tolist(),
        solve_ms=elapsed,process_ms=(time.perf_counter()-self.wall_origin-target_t)*1000,
        received=self.received,processed=self.processed,dropped=0,waiting=waiting,
        unguarded_missing_future=unguarded_missing_future,
        source_clock_guard_active=bool(not terminal and target_t+1e-8<source_now),
        source_clock_delay_ms=float((source_now-target_t)*1000),
        future_reference_latest_time=float(target_t+max(self.physics.offsets)+self.source_origin),
        reference_available_through=float(last+self.source_origin),
        reference_terminal_drain=terminal,fallen=fallen,finished=self.finished,
        predicted_future=forecast_evidence['maximum_prediction_horizon_s']>0.,
        future_prediction=forecast_evidence,
        physics=True,simulation_time=self.physics.data.time,initial_floor_offset=self.physics.initial_floor_offset,
        pelvis_tilt_deg=tilt,reference_root_pos=reference[:3].tolist(),
        position_feedback=bool(self.position_feedback),
        height_target_m=height_target,
        knee_reference_correction_rad=self.height_feedback.filtered if self.height_feedback else None,
        horizontal_error_m=float(np.linalg.norm(reference[:2]+self.position_feedback.origin-q[:2])) if self.position_feedback else None,
        reference_tilt_correction_deg=np.rad2deg(self.position_feedback.filtered).tolist() if self.position_feedback else None)
