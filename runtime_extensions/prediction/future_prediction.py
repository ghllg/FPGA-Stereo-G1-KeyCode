"""P3: explicitly forecast policy lookahead; current reference remains observed."""
import numpy as np
import copy,json
from native_policy_forecast import PolicyForecast, ForecastUnavailable, PARAMETERS


def clear_forecast(stream):
    stream.policy_forecast = None


def pause_for_forecast(stream, reason):
    with stream.frame_lock:
        stream.paused = True
        sink=getattr(stream,'forecast_rejections',None)
        if sink is not None:
            forecast=getattr(stream,'policy_forecast',None)
            rejection=dict(reason=str(reason),tick_terminal=getattr(stream,'_tick_terminal',None),
                forecast_evidence=copy.deepcopy(forecast.evidence) if forecast is not None else None,
                observed_reference_snapshot=stream.freeze_reference_queries([]))
            sink.write(json.dumps(rejection,separators=(',',':'),allow_nan=False)+'\n')
        clear_forecast(stream)
        stream.physics.controller.stop_reference()
    return dict(type='tracking_state', state='paused',
                reason='P4 prediction unavailable: '+str(reason),
                prediction_failure=str(reason), predicted_future=False)


def prepare_rows(stream, target, *, terminal=None):
    """Called under the real frame lock, immediately before one physical step."""
    terminal=bool(stream.ended) if terminal is None else bool(terminal)
    buffer = stream.buffer
    first, last = float(buffer.times[0]), float(buffer.times[-1])
    if target < first-1e-8 or target > last+1e-8:
        raise ForecastUnavailable('current_target_not_observed')
    if max(stream.physics.offsets) > .160+1e-8:
        raise ForecastUnavailable('policy_offset_exceeds_160ms')
    stamps = target+stream.physics.offsets
    if terminal:
        stamps = np.minimum(stamps,last)  # Existing explicit terminal drainage only.
    queries=[dict(kind='current',requested_source_time=target+stream.source_origin,
                  effective_source_time=target+stream.source_origin)]
    for offset,query in zip(stream.physics.offsets,stamps):
        entry=dict(kind='policy_future_prediction' if query>last+1e-9 else 'policy_future_observed',
            offset_s=float(offset),requested_source_time=float(target+offset+stream.source_origin),
            effective_source_time=float(query+stream.source_origin))
        if query>last+1e-9:
            entry['observation_source_times']=[float(t+stream.source_origin) for t in buffer.times]
        queries.append(entry)
    stream._reference_use_snapshot=stream.freeze_reference_queries(queries)
    cq,cv=buffer.sample(target)
    stream._reference_use_snapshot['reference_queries'][0].update(
        qpos_before_existing_feedback=np.asarray(cq).tolist(),
        joint_velocity_before_existing_feedback=np.asarray(cv).tolist())
    forecast = getattr(stream, 'policy_forecast', None)
    if forecast is None:
        forecast = stream.policy_forecast = PolicyForecast()
    if forecast.snapshot is None or forecast.snapshot.stamp != last:
        forecast.observe(buffer)
    rows=[]
    for query in stamps:
        if query > last+1e-9 and (
                not stream.started or getattr(stream,'resume_pending',False)
                or getattr(stream,'recovery_priming',False)):
            raise ForecastUnavailable('real_initialization_or_recovery_priming_required')
        q,v,meta = forecast.sample(buffer,target,float(query))
        rows.append(dict(query_time=float(query+stream.source_origin),
            query_relative_time=float(query),source=meta['source'],
            prediction_horizon_s=meta['prediction_horizon_s'],
            last_real_source_time=last+stream.source_origin,
            qpos=np.asarray(q).tolist(),joint_velocity=np.asarray(v).tolist(),
            reference_stage='before_legacy_position_and_height_feedback',
            qpos_qd_curve='existing_buffer' if meta['prediction_horizon_s']==0 else 'same_exponential_decay_trajectory',
            reliable_measurement=False,reliable_recovery=False,
            root_velocity=meta.get('root_velocity'),
            root_local_angular_velocity=meta.get('root_local_angular_velocity')))
    for frozen,row in zip(stream._reference_use_snapshot['reference_queries'][1:],rows):
        frozen.update(qpos_before_existing_feedback=row['qpos'],
                      joint_velocity_before_existing_feedback=row['joint_velocity'])
    horizons=[r['prediction_horizon_s'] for r in rows]
    evidence=dict(experiment='P4_practical_future_prediction',
        current_target_time=float(target+stream.source_origin),
        current_target_source='observed_buffer_only',last_real_source_time=last+stream.source_origin,
        reference_snapshot_ns=stream._reference_use_snapshot['reference_snapshot_ns'],
        last_real_relative_time=last,maximum_prediction_horizon_s=max(horizons),
        forecast_observation_evidence=forecast.evidence,
        real_buffer_size=len(buffer.times),predictions_appended_to_buffer=0,
        prediction_counts_for_recovery=False,
        bounded_recovering=bool(getattr(stream,'recovering',False)),
        terminal_drain=terminal,references=rows)
    return rows,evidence


def predicted_references(stream, target, *, terminal=None):
    import mujoco
    rows,evidence=prepare_rows(stream,target,terminal=terminal)
    positions,velocities,rotations=[],[],[]
    for row in rows:
        q=np.asarray(row['qpos']);v=np.asarray(row['joint_velocity'])
        stream.buffer.fk.qpos[:]=q
        mujoco.mj_forward(stream.buffer.model,stream.buffer.fk)
        positions.append(q[7:]);velocities.append(v)
        rotations.append(stream.buffer.fk.xquat[1:,[1,2,3,0]].copy())
    return dict(dof_pos=np.asarray(positions,dtype=np.float32),
                dof_vel=np.asarray(velocities,dtype=np.float32),
                body_rot=np.asarray(rotations,dtype=np.float32)),evidence
